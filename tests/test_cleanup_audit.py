"""端到端测试：POST /cleanup-audit 整树声明清理审计。"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def audit(payload, status=200):
    resp = client.post("/cleanup-audit", json=payload)
    assert resp.status_code == status, resp.text
    return resp.json()


def node(node_id, tag="div", children=None, classes=None, attributes=None):
    return {
        "id": node_id,
        "tag": tag,
        "classes": classes or [],
        "attributes": attributes or {},
        "children": children or [],
    }


def rule(selector, *decls, origin="author", layer=None, order=None):
    result = {
        "selector": selector,
        "origin": origin,
        "layer": layer,
        "declarations": [
            {"property": prop, "value": value, "important": important}
            for prop, value, important in decls
        ],
    }
    if order is not None:
        result["order"] = order
    return result


def decl(prop, value, important=False):
    return prop, value, important


def refs(items):
    return [(item["rule_index"], item["declaration_index"]) for item in items]


def retained_map(data):
    return {
        (item["rule_index"], item["declaration_index"]): item
        for item in data["retained"]
    }


def test_removes_unmatched_and_losing_declarations_together_but_keeps_winner():
    root = node("root", "section", [node("child", "p")])
    payload = {
        "root": root,
        "rules": [
            rule("p", decl("color", "red")),
            rule("p", decl("color", "green")),
            rule(".never", decl("margin", "10px"), decl("padding", "2px")),
        ],
    }
    data = audit(payload)
    assert refs(data["removable"]) == [(0, 0), (2, 0), (2, 1)]
    kept = retained_map(data)
    assert set(kept) == {(1, 0)}
    assert kept[(1, 0)]["node_id"] == "child"
    assert kept[(1, 0)]["property"] == "color"
    assert "原层叠胜出者" in kept[(1, 0)]["reason"]


def test_custom_property_inheritance_and_fallback_block_deletion():
    tree = node("root", "div", [node("child", "p")])
    payload = {
        "root": tree,
        "rules": [
            rule("div", decl("--x", "red")),
            rule("p", decl("color", "var(--x)")),
            rule("div", decl("--unused", "1px")),
        ],
    }
    data = audit(payload)
    # --unused 仍改变 root 的 custom_properties，因此不能删除。
    assert data["removable"] == []
    kept = retained_map(data)
    assert kept[(0, 0)]["node_id"] == "root"
    assert kept[(0, 0)]["property"] == "--x"
    assert kept[(1, 0)]["node_id"] == "child"
    assert kept[(1, 0)]["property"] == "color"
    assert kept[(2, 0)]["node_id"] == "root"
    assert kept[(2, 0)]["property"] == "--unused"


def test_fallback_only_used_after_invalid_variable_still_retains_invalid_state():
    # root 的 --bad 失效；即使 child 的普通属性总是走 fallback，
    # 删除 root 声明也会让 root 的自定义属性 invalid 状态消失。
    tree = node("root", "div", [node("child", "p")])
    payload = {
        "root": tree,
        "rules": [
            rule("div", decl("--bad", "var(--missing)")),
            rule(
                "p",
                decl("--bad", "var(--missing)"),
                decl("color", "var(--bad, blue)"),
            ),
        ],
    }
    data = audit(payload)
    assert data["removable"] == []
    blocked = retained_map(data)[(0, 0)]
    assert blocked["node_id"] == "root"
    assert blocked["property"] == "--bad"
    assert "invalid" in blocked["reason"]


def test_existing_invalid_declaration_is_retained_but_identical_invalid_loser_removable():
    tree = node("only", "p")
    payload = {
        "root": tree,
        "rules": [
            rule("p", decl("color", "var(--missing)")),
            rule("p", decl("color", "var(--missing)", True)),
        ],
    }
    data = audit(payload)
    # important 声明胜出且失效；删除它会让普通同值声明胜出，invalid 状态不变、
    # 但胜出者改变，按要求必须阻断。普通声明可随批次删除。
    assert refs(data["removable"]) == [(0, 0)]
    blocked = retained_map(data)[(1, 0)]
    assert blocked["node_id"] == "only"
    assert blocked["property"] == "color"
    assert "胜出者" in blocked["reason"]


def test_declaration_that_would_create_new_invalid_is_not_added_to_batch():
    tree = node("root", "div", [node("child", "p")])
    payload = {
        "root": tree,
        "rules": [
            rule("div", decl("--x", "red"), decl("color", "green")),
            rule("p", decl("color", "var(--x)")),
        ],
    }
    data = audit(payload)
    # root 上的 color 声明未被任何最终结果使用，可删除；
    # --x 一旦删除，child 的 color 会新失效，必须保留。
    assert refs(data["removable"]) == [(0, 1)]
    blocked = retained_map(data)[(0, 0)]
    assert blocked["node_id"] == "child"
    assert blocked["property"] == "color"
    assert "computed 'red'" in blocked["reason"]
    assert "invalid null" in blocked["reason"]


def test_duplicate_id_returns_422():
    payload = {
        "root": node("same", "div", [node("same", "p")]),
        "rules": [],
    }
    data = audit(payload, status=422)
    assert "重复" in data["detail"]


def test_invalid_selector_returns_422():
    payload = {
        "root": node("root", "p"),
        "rules": [rule("> p", decl("color", "red"))],
    }
    data = audit(payload, status=422)
    assert "选择器语法错误" in data["detail"]


def test_empty_declaration_value_returns_422():
    payload = {
        "root": node("root", "p"),
        "rules": [rule("p", decl("color", ""))],
    }
    data = audit(payload, status=422)
    assert "value 不能为空" in data["detail"]


def test_empty_node_returns_422():
    payload = {"root": None, "rules": []}
    data = audit(payload, status=422)
