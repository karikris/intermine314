import pytest

from intermine314.model import Model, ModelError


DIAMOND_MODEL_XML = """
<model name="mock" package="org.mock">
  <class name="A" />
  <class name="B" extends="A" />
  <class name="C" extends="A" />
  <class name="D" extends="B C" />
</model>
""".strip()


CYCLE_MODEL_XML = """
<model name="mock" package="org.mock">
  <class name="A" extends="B" />
  <class name="B" extends="A" />
</model>
""".strip()


def test_to_ancestry_uses_unique_ancestors_for_diamond_graph():
    model = Model(DIAMOND_MODEL_XML)
    d = model.get_class("D")

    names = [ancestor.name for ancestor in model.to_ancestry(d)]
    assert names == ["B", "A", "C"]
    assert names.count("A") == 1


def test_to_ancestry_populates_and_reuses_memoized_cache():
    model = Model(DIAMOND_MODEL_XML)
    d = model.get_class("D")

    first = model.to_ancestry(d)
    cache_after_first = dict(model._ancestry_cache)
    second = model.to_ancestry(d)

    assert "D" in cache_after_first
    assert tuple(first) == cache_after_first["D"]
    assert [c.name for c in second] == [c.name for c in first]


def test_to_ancestry_internal_tuple_cache_is_reused():
    model = Model(DIAMOND_MODEL_XML)
    d = model.get_class("D")

    first = model._to_ancestry_tuple(d)
    second = model._to_ancestry_tuple(d)
    public = model.to_ancestry(d)

    assert isinstance(first, tuple)
    assert first is second
    assert [c.name for c in public] == [c.name for c in first]


def test_to_ancestry_detects_cycles_with_clear_error():
    with pytest.raises(ModelError, match="Cycle detected in class ancestry"):
        Model(CYCLE_MODEL_XML)
