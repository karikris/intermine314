"""Offline contracts adapted from upstream 1.13.0's public logic tests."""

from dataclasses import FrozenInstanceError
from io import StringIO
from xml.etree import ElementTree as ET

import pytest

import intermine314.constraints as public_constraints
import intermine314.query.constraints as constraints
from intermine314.query.builder import ConstraintError, Query, QueryError
from intermine314.query.executor import QueryExecutor
from intermine314.query.pathfeatures import Join, PathFeature, SortOrder, SortOrderList


@pytest.fixture(params=["native", "legacy"])
def query(request):
    return Query(root="Gene", compatibility=request.param)


def add_codes(query, count=4):
    return [query.add_constraint("symbol", "=", str(i)) for i in range(count)]


def test_default_logic_is_dynamic_and_single_nodes_work(query):
    assert query.get_logic() == query.logic == ""
    assert query.set_logic("  ") is query
    query.validate_logic()
    a = add_codes(query, 1)[0]
    assert query.get_logic() is a
    assert a.get_codes() == ["A"]
    assert str(a) == "A"
    assert constraints.CodedConstraint.to_string(a) == "Gene.symbol ="
    assert a.to_dict() == {"path": "Gene.symbol", "op": "=", "code": "A", "value": "0"}
    assert query.set_logic(a) is query
    assert query.logic is a
    query.logic = "A"
    assert query.logic is a
    with pytest.raises(constraints.EmptyLogicError):
        query.set_logic("")
    fresh = Query(root="Gene", compatibility=query.compatibility)
    add_codes(fresh, 2)
    assert str(fresh.logic) == "A and B"
    fresh.add_constraint("symbol", "IS NULL", None)
    assert str(fresh.logic) == "A and B and C"
    fresh.validate_logic()


@pytest.mark.parametrize("expression, expected", [
    ("(B or C) and (A or D)", "(B or C) and (A or D)"),
    ("B and C or A and D", "B and (C or A) and D"),
    ("(A and B) or (A and C and D)", "(A and B) or (A and C and D)"),
    ("((A or B) and (C or D))", "(A or B) and (C or D)"),
    ("a&&(b|c)||d", "(A and (B or C)) or D"),
    ("A and (B) or C or D", "(A and B) or C or D"),
])
def test_historical_string_precedence_grouping_aliases(query, expression, expected):
    add_codes(query)
    assert query.set_logic(expression) is query
    assert str(query.logic) == expected
    assert str(query._logic_parser.parse(str(query.logic))) == expected


def test_node_operations_parent_links_and_public_identity(query):
    a, b, c, d = add_codes(query)
    for name in ("LogicNode", "LogicGroup", "LogicParser", "LogicParseError", "EmptyLogicError"):
        assert getattr(public_constraints, name) is getattr(constraints, name)
    for expression, expected in [
        (a + b + c + d, "A and B and C and D"),
        (a & b & c & d, "A and B and C and D"),
        (a | b | c | d, "A or B or C or D"),
        (a + b & c | d, "(A and B and C) or D"),
    ]:
        query.set_logic(expression)
        assert query.logic is expression
        assert str(expression) == expected
        assert expression.get_codes() == ["A", "B", "C", "D"]
        assert repr(expression) == f"<LogicGroup: {expected}>"
    expression = (a | b) & (a | c | d)
    assert expression.left.parent is expression
    assert expression.right.parent is expression
    assert expression.get_codes() == ["A", "B", "A", "C", "D"]
    for method in (a.__add__, a.__and__, a.__or__):
        assert method(1) is NotImplemented
    with pytest.raises(TypeError):
        _ = expression + 1
    with pytest.raises(TypeError):
        constraints.LogicGroup(a, "bar", b)
    with pytest.raises(TypeError):
        constraints.LogicGroup(a, "AND", 1)


def test_parser_helpers_and_multi_letter_codes(query):
    add_codes(query, 27)
    parser = constraints.LogicParser(query)
    assert parser.get_constraint("AA") is query.get_constraint("AA")
    assert [parser.get_priority(x) for x in ("AND", "OR", "(", ")", "?")] == [2, 1, 3, 3, None]
    assert parser.check_syntax(["A", "AND", "(", "B", "OR", "AA", ")"]) is None
    with pytest.raises(constraints.LogicParseError):
        parser.check_syntax(["A", "AND"])
    postfix = parser.infix_to_postfix(["B", "AND", "C", "OR", "AA", "AND", "D"])
    assert postfix == ["B", "C", "AA", "OR", "AND", "D", "AND"]
    assert str(parser.postfix_to_tree(postfix)) == "B and (C or AA) and D"
    query.set_logic(" or ".join(con.code for con in query.coded_constraints))
    assert "AA" in query.logic.get_codes()
    assert ET.fromstring(query.to_xml()).get("constraintLogic") == str(query.logic)


def test_closing_group_completes_preceding_operation(query):
    add_codes(query)
    parser = query._logic_parser
    assert str(parser.parse("A and (B) or C")) == "(A and B) or C"
    assert str(parser.parse(str(parser.parse("A and (B) or C")))) == "(A and B) or C"
    expected = "(A and (B or C)) or D"
    assert str(parser.parse("A && (B | C) || D")) == expected
    assert str(parser.parse("A&&((B|C))||D")) == expected
    assert str(parser.parse(expected)) == expected


def test_nested_group_markers_preserve_grouping_and_truth_value(query):
    add_codes(query)
    query.set_logic("B OR ((A OR C) AND D)")
    expected = "B or ((A or C) and D)"
    tree = query.logic
    assert tree.op == "OR" and tree.left is query.get_constraint("B")
    assert tree.right.op == "AND" and tree.right.right is query.get_constraint("D")
    assert tree.right.left.op == "OR"
    assert tree.right.left.get_codes() == ["A", "C"]
    assert str(tree) == expected

    def evaluate(node, values):
        if isinstance(node, constraints.LogicGroup):
            left = evaluate(node.left, values)
            right = evaluate(node.right, values)
            return left and right if node.op == "AND" else left or right
        return values[node.code]

    values = {"A": False, "B": True, "C": False, "D": False}
    assert evaluate(tree, values) is True
    # Upstream successfully parsed this expression into the wrong grouping.
    historical_tree = query._logic_parser.parse("(B or A or C) and D")
    assert evaluate(historical_tree, values) is False
    xml_logic = ET.fromstring(query.to_xml()).get("constraintLogic")
    assert xml_logic == expected
    reparsed = query._logic_parser.parse(xml_logic)
    assert str(reparsed) == expected
    assert evaluate(reparsed, values) is True


@pytest.mark.parametrize("expression", [
    "AND A", "A AND", "A OR OR B", "A B", "()", "A(B)", "A and ()",
    "(A", "A)", ")A(", "A and (B and )C", "A and (B) C", "A and (B) (C)",
    "A not B", "A ^ B", "A &&& B", "A or 1", "A.B", "A_1",
])
def test_invalid_logic_raises_public_parse_error(query, expression):
    add_codes(query)
    with pytest.raises(constraints.LogicParseError):
        query.set_logic(expression)


def test_logic_validation_missing_unknown_and_wrong_inputs(query):
    a, b, _, _ = add_codes(query)
    with pytest.raises(QueryError, match="not mentioned"):
        query.set_logic(a | b)
    with pytest.raises(ConstraintError, match="code 'E'"):
        query.set_logic("E and C or A and D")
    outsider = constraints.BinaryConstraint("Gene.symbol", "=", "x", "E")
    with pytest.raises(QueryError, match="Unknown constraint code"):
        query.validate_logic(a | b | outsider)
    with pytest.raises(QueryError, match="Unknown constraint code"):
        query.set_logic(a | b | outsider)
    with pytest.raises(QueryError, match="not mentioned"):
        query.validate_logic("")
    with pytest.raises(TypeError):
        query.set_logic(123)
    query.set_logic("A and B and C and D")
    query.add_constraint("symbol", "=", "new")
    with pytest.raises(QueryError, match="not mentioned"):
        query.validate_logic()
    unchecked = Query(root="Gene", validate=False, compatibility=query.compatibility)
    a, _ = add_codes(unchecked, 2)
    unchecked.set_logic(a)
    with pytest.raises(QueryError, match="not mentioned"):
        unchecked.validate_logic()


def test_public_parser_helpers_reject_malformed_sequences(query):
    add_codes(query)
    parser = constraints.LogicParser(query)
    for tokens in (["A", "AND"], ["(", "A"], ["A", ")"], ["A", "B"]):
        with pytest.raises(constraints.LogicParseError):
            parser.infix_to_postfix(tokens)
    for tokens in (["A", "AND"], ["A", "B"], ["A", "("]):
        with pytest.raises(constraints.LogicParseError):
            parser.postfix_to_tree(tokens)
    with pytest.raises(constraints.EmptyLogicError):
        parser.parse(" ")
    with pytest.raises(constraints.EmptyLogicError):
        parser.postfix_to_tree([])


def test_logic_serialization_spec_executor_and_clone_are_independent(query):
    add_codes(query)
    query.add_view("symbol")
    query.set_logic("(A or B) and (C or D)")
    spec = query.to_spec()
    assert spec.constraint_logic == "(A or B) and (C or D)"
    with pytest.raises(FrozenInstanceError):
        spec.constraint_logic = "A"
    executor = QueryExecutor(None, spec)
    for xml in (query.to_xml(), query.to_formatted_xml(), query.to_query_params()["query"],
                executor.to_query_params()["query"]):
        assert ET.fromstring(xml).get("constraintLogic") == spec.constraint_logic
    clone = query.clone()
    assert clone._logic_parser._query is clone
    assert clone.logic is not query.logic
    assert clone.logic.left.parent is clone.logic
    assert clone.logic.left.left is clone.get_constraint("A")
    assert clone.logic.left.left is not query.get_constraint("A")
    clone.get_constraint("A").value = "changed"
    assert query.get_constraint("A").value == "0"
    clone.set_logic("A or B or C or D")
    assert str(query.logic) == spec.constraint_logic
    clone.add_constraint("symbol", "=", "fifth")
    query.add_constraint("symbol", "=", "fifth")
    assert clone.get_constraint("E") is not query.get_constraint("E")
    query.set_logic("A or B or C or D or E")
    assert ET.fromstring(executor.to_query_params()["query"]).get("constraintLogic") == spec.constraint_logic


def test_xml_default_logic_only_with_multiple_coded_constraints(query):
    assert "constraintLogic" not in ET.fromstring(query.to_xml()).attrib
    add_codes(query, 1)
    assert "constraintLogic" not in ET.fromstring(query.to_xml()).attrib
    query.add_constraint(constraints.SubClassConstraint("Gene", "SpecialGene"))
    assert "constraintLogic" not in ET.fromstring(query.to_xml()).attrib
    query.add_constraint("symbol", "=", "x")
    assert ET.fromstring(query.to_xml()).get("constraintLogic") == "A and B"


def test_clone_rebinds_valid_external_nodes_to_its_own_constraints(query):
    a, b = add_codes(query, 2)
    external_a = constraints.BinaryConstraint(a.path, "=", "external", a.code)
    query.set_logic(external_a | b)
    clone = query.clone()
    assert clone.logic.left is clone.get_constraint("A")
    assert clone.logic.right is clone.get_constraint("B")
    assert clone.logic.parent is None
    # A lone coded node is a valid expression too.
    single = Query(root="Gene", compatibility=query.compatibility)
    single.add_constraint("symbol", "=", "owned")
    single.set_logic(external_a)
    assert single.clone().logic is not external_a
    cloned_single = single.clone()
    assert cloned_single.logic is cloned_single.get_constraint("A")


def test_csv_cannot_silently_ignore_explicit_unverified_logic(query, tmp_path):
    query.do_verification = False
    query.set_logic(constraints.BinaryConstraint("Gene.symbol", "=", "x", "A"))
    with pytest.raises(ValueError, match="logic"):
        query.to_parquet(tmp_path / "rows.parquet", csv_input=StringIO("symbol\nx\n"))


def test_path_features_and_sort_container_behavior():
    feature = PathFeature("Gene.symbol")
    assert feature.to_dict() == {"path": "Gene.symbol"}
    assert feature.to_string() == "Gene.symbol"
    assert repr(feature) == "<PathFeature: Gene.symbol>"
    with pytest.raises(AttributeError):
        _ = feature.child_type
    join = Join("Gene.organism", "outer")
    assert join.child_type == "join"
    assert repr(join) == "<Join: Gene.organism OUTER>"
    assert join.to_dict() == {"path": "Gene.organism", "style": "OUTER"}
    order = SortOrder("Gene.symbol", "DESC")
    assert order.to_string() == str(order) == "Gene.symbol desc"
    orders = SortOrderList(order, ("Gene.id", "asc"))
    assert len(orders) == 2
    assert not orders.is_empty()
    assert list(orders) == [order, orders.sort_orders[1]]
    assert str(orders) == "Gene.symbol desc Gene.id asc"
    assert repr(orders) == "<SortOrderList: [Gene.symbol desc Gene.id asc]>"
    # Existing native repair: next() returns the first element without consuming.
    assert next(orders) is orders.next() is order
    orders.clear()
    assert len(orders) == 0 and orders.is_empty()
    orders.append(("Gene.id", "DESC"))
    assert str(orders) == "Gene.id desc"
    assert not orders.is_empty()


def test_query_children_prefix_default_sort_and_validation(query):
    with pytest.raises(QueryError, match="view is empty"):
        query.get_default_sort_order()
    query.add_view("organism.name", "symbol")
    a = query.add_constraint("symbol", "=", "x")
    subclass = query.add_constraint(constraints.SubClassConstraint("Gene", "SpecialGene"))
    assert query.constraints == [a, subclass]
    assert query.coded_constraints == [a]
    assert query.get_constraint("A") is a
    assert str(query.get_sort_order()) == "Gene.organism.name asc"
    assert query.outerjoin("organism") is query
    assert query.get_default_sort_order() == ""
    assert query.add_join("proteins", "inner") is query
    assert query.children() == [*query.joins, a, subclass]
    query.add_sort_order("symbol", "DESC")
    query.validate_sort_order()
    assert str(query.get_sort_order()) == "Gene.symbol desc"
    with pytest.raises(QueryError, match="not in the query"):
        query.validate_sort_order(SortOrder("Gene.absent.name", "asc"))
    xml = ET.fromstring(query.to_formatted_xml())
    assert xml.get("sortOrder") == "Gene.symbol desc"
    assert [(node.tag, node.get("path")) for node in xml] == [
        ("join", "Gene.organism"), ("join", "Gene.proteins"),
        ("constraint", "Gene.symbol"), ("constraint", "Gene"),
    ]
