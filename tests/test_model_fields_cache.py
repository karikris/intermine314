import pytest

from intermine314.model import Attribute, Model


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


def _build_model():
    return Model(MODEL_XML)


def test_fields_cache_records_hits_and_misses():
    model = _build_model()
    gene = model.get_class("Gene")

    initial_hits = gene.fields_cache_hits
    initial_misses = gene.fields_cache_misses

    first = gene.fields
    second = gene.fields

    assert isinstance(first, tuple)
    assert isinstance(second, tuple)
    assert gene.fields_cache_hits >= initial_hits + 1
    assert gene.fields_cache_misses >= initial_misses


def test_fields_cache_invalidation_on_add_field():
    model = _build_model()
    gene = model.get_class("Gene")
    _ = gene.fields
    misses_before = gene.fields_cache_misses

    gene.add_field(Attribute("alias", "String", gene))
    fields_after = gene.fields

    assert any(field.name == "alias" for field in fields_after)
    assert gene.fields_cache_misses >= misses_before + 1


def test_attribute_reference_collection_caches_refresh_after_field_update():
    model = _build_model()
    gene = model.get_class("Gene")
    _ = gene.attributes

    gene.add_field(Attribute("secondaryIdentifier", "String", gene))

    attributes = gene.attributes
    assert any(field.name == "secondaryIdentifier" for field in attributes)


def test_composed_class_field_dict_is_cached_and_immutable():
    model = _build_model()
    composed = model.get_class("Gene,Organism")

    first = composed.field_dict
    second = composed.field_dict

    assert first is second
    assert first["id"] is second["id"]
    assert "symbol" in first
    assert "name" in first

    with pytest.raises(TypeError):
        first["x"] = first["id"]
