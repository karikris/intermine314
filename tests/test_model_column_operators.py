import pytest

from intermine314.model import Column, Model


MODEL_XML = """
<model name="mock" package="org.mock">
  <class name="Gene">
    <reference name="organism" referenced-type="Organism"/>
    <attribute name="symbol" type="java.lang.String"/>
    <attribute name="length" type="java.lang.Integer"/>
  </class>
  <class name="Organism">
    <attribute name="name" type="java.lang.String"/>
  </class>
</model>
""".strip()


class _ConstraintSource:
    def __init__(self):
        self.calls = []

    def make_list_constraint(self, path, op):
        self.calls.append((path, op))
        return ("constraint", path, op)


def _build_model():
    return Model(MODEL_XML)


def _snapshot(node):
    if hasattr(node, "vargs"):
        return node.vargs
    return node


def test_eq_and_ne_operator_snapshots():
    model = _build_model()
    symbol = model.column("Gene.symbol")
    length = model.column("Gene.length")
    source = _ConstraintSource()

    assert _snapshot(symbol == None) == ("Gene.symbol", "IS NULL")
    assert _snapshot(symbol == length) == ("Gene.symbol", "IS", "Gene.length")
    assert _snapshot(symbol == ["A", "B"]) == ("Gene.symbol", "ONE OF", ["A", "B"])
    assert _snapshot(symbol == "A1") == ("Gene.symbol", "=", "A1")
    assert (symbol == source) == ("constraint", "Gene.symbol", "IN")

    assert _snapshot(symbol != None) == ("Gene.symbol", "IS NOT NULL")
    assert _snapshot(symbol != length) == ("Gene.symbol", "IS NOT", "Gene.length")
    assert _snapshot(symbol != ["A", "B"]) == ("Gene.symbol", "NONE OF", ["A", "B"])
    assert _snapshot(symbol != "A1") == ("Gene.symbol", "!=", "A1")
    assert (symbol != source) == ("constraint", "Gene.symbol", "NOT IN")


def test_membership_operators_snapshot_and_errors():
    model = _build_model()
    symbol = model.column("Gene.symbol")
    source = _ConstraintSource()

    assert _snapshot(symbol.in_(["A"])) == ("Gene.symbol", "ONE OF", ["A"])
    assert symbol.in_(source) == ("constraint", "Gene.symbol", "IN")
    assert _snapshot(symbol ^ ["A"]) == ("Gene.symbol", "NONE OF", ["A"])
    assert (symbol ^ source) == ("constraint", "Gene.symbol", "NOT IN")

    with pytest.raises(TypeError, match="unsupported operand for operator in_"):
        symbol.in_(42)
    with pytest.raises(TypeError, match="unsupported operand for operator xor"):
        symbol ^ 42


def test_comparison_lookup_and_shift_operator_snapshots():
    model = _build_model()
    symbol = model.column("Gene.symbol")
    root = model.column("Gene")
    organism = root.organism
    organism_class = model.column("Organism")

    root._branches["stale"] = object()
    assert _snapshot(organism < organism_class) == ("Gene.organism", "Organism")
    assert root._subclasses["Gene.organism"] == "Organism"
    assert root._branches == {}

    assert _snapshot(symbol < ["A"]) == ("Gene.symbol", "ONE OF", ["A"])
    assert _snapshot(symbol < 10) == ("Gene.symbol", "<", 10)
    assert _snapshot(symbol <= ["A"]) == ("Gene.symbol", "ONE OF", ["A"])
    assert _snapshot(symbol <= 10) == ("Gene.symbol", "<=", 10)
    assert _snapshot(symbol > 10) == ("Gene.symbol", ">", 10)
    assert _snapshot(symbol >= 10) == ("Gene.symbol", ">=", 10)

    assert _snapshot(symbol % "A1") == ("Gene.symbol", "LOOKUP", "A1")
    assert _snapshot(symbol % ("A1", "Gene")) == ("Gene.symbol", "LOOKUP", "A1", "Gene")
    assert _snapshot(symbol >> "Gene") == ("Gene.symbol", "Gene")
    assert _snapshot(symbol << "Gene") == ("Gene.symbol", "Gene")


def test_filter_alias_is_class_level_and_not_instance_bound():
    model = _build_model()
    symbol = model.column("Gene.symbol")

    assert Column.filter is Column.where
    assert symbol.filter.__func__ is symbol.where.__func__
    assert "filter" not in getattr(symbol, "__dict__", {})


def test_branch_cache_is_bounded_lru_with_stable_semantics():
    model_xml = """
    <model name="mock" package="org.mock">
      <class name="Gene">
        <attribute name="a1" type="java.lang.String"/>
        <attribute name="a2" type="java.lang.String"/>
        <attribute name="a3" type="java.lang.String"/>
      </class>
    </model>
    """.strip()
    model = Model(model_xml)
    root = Column("Gene", model, branch_cache_maxsize=2)

    first_a1 = root.a1
    first_a2 = root.a2
    _ = root.a1  # Mark a1 as recently used.
    first_a3 = root.a3  # Evicts a2 as LRU entry.

    assert str(first_a1) == "Gene.a1"
    assert str(first_a2) == "Gene.a2"
    assert str(first_a3) == "Gene.a3"
    assert "a1" in root._branches
    assert "a2" not in root._branches
    assert "a3" in root._branches

    second_a2 = root.a2
    assert str(second_a2) == "Gene.a2"
    assert root.branch_cache_stats["size"] <= 2
    assert root.branch_cache_stats["evictions"] >= 1
    assert root.branch_cache_stats["hits"] >= 1
    assert root.branch_cache_stats["misses"] >= 3


def test_branch_cache_rejects_non_positive_maxsize():
    model = _build_model()
    with pytest.raises(ValueError, match="branch_cache_maxsize must be a positive integer"):
        Column("Gene", model, branch_cache_maxsize=0)
