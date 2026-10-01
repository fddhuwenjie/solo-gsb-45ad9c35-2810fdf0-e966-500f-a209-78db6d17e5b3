"""整棵 DOM 样本上的结构化样式声明清理审计。"""
from __future__ import annotations

from dataclasses import dataclass

from .cascade import Candidate, cascade
from .engine import _resolve_layer_order
from .models import (
    CleanupAuditRequest,
    CleanupAuditResponse,
    CleanupNode,
    DeclarationRefOut,
    ElementInfo,
    RetainedDeclarationOut,
    Rule,
)
from .selector import (
    SelectorSyntaxError,
    parse_selector,
    selector_matches,
    selector_specificity,
)
from .variables import Unresolvable, resolve_custom_properties, substitute_vars


class CleanupAuditError(ValueError):
    """清理审计请求语义非法。"""


@dataclass(frozen=True)
class TreeNode:
    id: object
    element: ElementInfo
    ancestors: list[ElementInfo]


@dataclass
class NodeSnapshot:
    computed: dict[str, str | None]
    custom_all: dict[str, tuple[str, str | None]]
    normal_winners: dict[str, Candidate]
    custom_winners: dict[str, Candidate]


@dataclass
class ParsedRule:
    rule: Rule
    source_order: int
    compounds: list
    combinators: list
    specificity: tuple[int, int, int]


def _to_element(node: CleanupNode) -> ElementInfo:
    return ElementInfo(
        tag=node.tag,
        id=str(node.id),
        classes=list(node.classes),
        attributes=dict(node.attributes),
    )


def _flatten_tree(root: CleanupNode) -> list[TreeNode]:
    nodes: list[TreeNode] = []
    seen: set[object] = set()

    def walk(node: CleanupNode | None, ancestors: list[ElementInfo]) -> None:
        if node is None:
            raise CleanupAuditError("节点不能为空")
        if node.id in seen:
            raise CleanupAuditError(f"节点 id 重复：{node.id!r}")
        seen.add(node.id)
        element = _to_element(node)
        nodes.append(TreeNode(id=node.id, element=element, ancestors=list(ancestors)))
        next_ancestors = ancestors + [element]
        for child in node.children:
            walk(child, next_ancestors)

    walk(root, [])
    return nodes


def _validate_rules(rules: list[Rule], layers: list[str]) -> None:
    if any(not isinstance(layer, str) or not layer.strip() for layer in layers):
        raise CleanupAuditError("层名必须为非空字符串")
    for ri, rule in enumerate(rules):
        if not rule.selector.strip():
            raise CleanupAuditError(f"规则 {ri} 的 selector 不能为空")
        for di, decl in enumerate(rule.declarations):
            if not decl.property.strip():
                raise CleanupAuditError(f"规则 {ri} 第 {di} 条声明的 property 不能为空")
            if not decl.value.strip():
                raise CleanupAuditError(f"规则 {ri} 第 {di} 条声明的 value 不能为空")
        # 提前复用选择器解析，非法选择器统一转成 422。
        try:
            compounds, combinators = parse_selector(rule.selector)
        except SelectorSyntaxError:
            raise
        selector_specificity(compounds)
        if rule.layer is not None and not rule.layer.strip():
            raise CleanupAuditError(f"规则 {ri} 的 layer 不能为空；未分层时显式传 null")


def _parse_rules(rules: list[Rule]) -> list[ParsedRule]:
    parsed: list[ParsedRule] = []
    for ri, rule in enumerate(rules):
        compounds, combinators = parse_selector(rule.selector)
        parsed.append(
            ParsedRule(
                rule=rule,
                source_order=rule.order if rule.order is not None else ri,
                compounds=compounds,
                combinators=combinators,
                specificity=selector_specificity(compounds),
            )
        )
    return parsed


def _solve_tree(
    nodes: list[TreeNode],
    parsed_rules: list[ParsedRule],
    layers: list[str],
    removed: frozenset[tuple[int, int]],
) -> list[NodeSnapshot]:
    layer_index = _resolve_layer_order(layers, [parsed.rule for parsed in parsed_rules])

    def candidates_for(
        element: ElementInfo,
        ancestors: list[ElementInfo],
        custom: bool,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for ri, parsed in enumerate(parsed_rules):
            rule = parsed.rule
            if not selector_matches(
                parsed.compounds, parsed.combinators, element, ancestors
            ):
                continue
            for di, decl in enumerate(rule.declarations):
                if (ri, di) in removed:
                    continue
                is_custom = decl.property.startswith("--")
                if is_custom != custom:
                    continue
                candidates.append(
                    Candidate(
                        selector=rule.selector,
                        origin=rule.origin,
                        layer=rule.layer,
                        layer_index=layer_index.get(rule.layer)
                        if rule.layer is not None
                        else None,
                        important=decl.important,
                        specificity=parsed.specificity,
                        source_order=parsed.source_order,
                        rule_index=ri,
                        decl_index=di,
                        property=decl.property if is_custom else decl.property.lower(),
                        value=decl.value,
                    )
                )
        return candidates

    snapshots: list[NodeSnapshot] = []
    inherited: dict[str, tuple[str, str | None]] = {}

    for node in nodes:
        custom_winners_map = cascade(
            candidates_for(node.element, node.ancestors, custom=True)
        )
        own = {
            prop: winner.value for prop, (winner, _eliminated) in custom_winners_map.items()
        }
        resolved, _notes = resolve_custom_properties(own, inherited)

        normal_winners_map = cascade(
            candidates_for(node.element, node.ancestors, custom=False)
        )
        computed: dict[str, str | None] = {}

        def lookup(name: str):
            return resolved.get(name, ("missing", None))

        for prop in sorted(normal_winners_map):
            winner, _eliminated = normal_winners_map[prop]
            try:
                value = (
                    substitute_vars(winner.value, lookup)
                    if "var(" in winner.value
                    else winner.value
                )
                computed[prop] = value
            except Unresolvable:
                computed[prop] = None

        snapshots.append(
            NodeSnapshot(
                computed=computed,
                custom_all=dict(resolved),
                normal_winners={
                    prop: winner for prop, (winner, _) in normal_winners_map.items()
                },
                custom_winners={
                    prop: winner for prop, (winner, _) in custom_winners_map.items()
                },
            )
        )
        inherited = resolved

    return snapshots


def _status_label(status: str | None, value: str | None) -> str:
    if status == "invalid" or value is None:
        return "invalid"
    if status == "missing":
        return "missing"
    return "computed" if status is None else "ok"


def _value_text(value: str | None) -> str:
    return "null" if value is None else repr(value)


def _candidate_text(candidate: Candidate | None) -> str:
    if candidate is None:
        return "无声明"
    return (
        f"规则 {candidate.rule_index} 第 {candidate.decl_index} 条"
        f"（{candidate.selector} {candidate.property}: {candidate.value}）"
    )


def _find_conflict(
    nodes: list[TreeNode],
    baseline: list[NodeSnapshot],
    candidate_world: list[NodeSnapshot],
    candidate: tuple[int, int],
) -> tuple[object, str, str] | None:
    for node, before, after in zip(nodes, baseline, candidate_world):
        custom_names = sorted(set(before.custom_all) | set(after.custom_all))
        for name in custom_names:
            b = before.custom_all.get(name, ("missing", None))
            a = after.custom_all.get(name, ("missing", None))
            before_winner = before.custom_winners.get(name)
            after_winner = after.custom_winners.get(name)
            before_ref = (
                (before_winner.rule_index, before_winner.decl_index)
                if before_winner
                else None
            )
            after_ref = (
                (after_winner.rule_index, after_winner.decl_index)
                if after_winner
                else None
            )
            if before_ref != after_ref or b != a:
                direct = "；该声明本身是原点的层叠胜出者" if before_ref == candidate else ""
                if before_ref != after_ref:
                    outcome = (
                        f"层叠胜出者由 {_candidate_text(before_winner)} 变为 "
                        f"{_candidate_text(after_winner)}"
                    )
                else:
                    outcome = (
                        f"状态/计算值由 {_status_label(b[0], b[1])} {_value_text(b[1])} "
                        f"变为 {_status_label(a[0], a[1])} {_value_text(a[1])}"
                    )
                reason = (
                    f"自定义属性 {name} 的 {outcome}"
                    f"，会改变本节点或后代继承的 custom_properties、invalid 状态或 var() fallback 求值"
                    f"{direct}"
                )
                return node.id, name, reason

        props = sorted(set(before.computed) | set(after.computed))
        for prop in props:
            bv = before.computed.get(prop)
            av = after.computed.get(prop)
            b_exists = prop in before.computed
            a_exists = prop in after.computed
            before_winner = before.normal_winners.get(prop)
            after_winner = after.normal_winners.get(prop)
            before_ref = (
                (before_winner.rule_index, before_winner.decl_index)
                if before_winner
                else None
            )
            after_ref = (
                (after_winner.rule_index, after_winner.decl_index)
                if after_winner
                else None
            )
            if before_ref != after_ref or bv != av or b_exists != a_exists:
                if before_ref == candidate:
                    lead = "该声明是原层叠胜出者，删除后"
                elif before_winner == after_winner:
                    lead = "层叠胜出者未变，但删除经自定义属性、循环或 fallback 依赖传导后"
                else:
                    lead = "删除后层叠胜出者发生变化，"
                if not b_exists and a_exists:
                    detail = f"该属性从无胜出声明变为 {_status_label(None, av)} {_value_text(av)}"
                elif not a_exists:
                    detail = "该属性不再有胜出声明"
                elif before_ref != after_ref:
                    detail = (
                        f"胜出者由 {_candidate_text(before_winner)} 变为 "
                        f"{_candidate_text(after_winner)}，取值为 "
                        f"{_status_label(None, av)} {_value_text(av)}"
                    )
                else:
                    detail = (
                        f"该属性由 {_status_label(None, bv)} {_value_text(bv)} "
                        f"变为 {_status_label(None, av)} {_value_text(av)}"
                    )
                reason = f"{lead}{detail}"
                return node.id, prop, reason
    return None


def audit_cleanup(req: CleanupAuditRequest) -> CleanupAuditResponse:
    nodes = _flatten_tree(req.root)
    _validate_rules(req.rules, req.layers)
    parsed_rules = _parse_rules(req.rules)

    baseline = _solve_tree(nodes, parsed_rules, req.layers, frozenset())
    removed: set[tuple[int, int]] = set()
    retained: list[RetainedDeclarationOut] = []

    for ri, rule in enumerate(req.rules):
        for di, _decl in enumerate(rule.declarations):
            ref = (ri, di)
            trial_world = _solve_tree(
                nodes, parsed_rules, req.layers, frozenset(removed | {ref})
            )
            conflict = _find_conflict(nodes, baseline, trial_world, ref)
            if conflict is None:
                removed.add(ref)
            else:
                node_id, prop, reason = conflict
                retained.append(
                    RetainedDeclarationOut(
                        rule_index=ri,
                        declaration_index=di,
                        node_id=node_id,
                        property=prop,
                        reason=reason,
                    )
                )

    removable = [
        DeclarationRefOut(rule_index=ri, declaration_index=di)
        for ri, rule in enumerate(req.rules)
        for di, _ in enumerate(rule.declarations)
        if (ri, di) in removed
    ]
    return CleanupAuditResponse(removable=removable, retained=retained)
