import pytest

from intermine314.model import Model, ModelParseError
from intermine314.model.xmlparse import _strip_java_prefix


PATH_SUBCLASS_MODEL_XML = """
<model name="mock" package="org.mock">
  <class name="Publication">
    <reference name="subject" referenced-type="BioEntity"/>
  </class>
  <class name="BioEntity" />
  <class name="Gene" extends="BioEntity">
    <reference name="organism" referenced-type="Organism"/>
  </class>
  <class name="Protein" extends="BioEntity">
    <reference name="organism" referenced-type="Organism"/>
  </class>
  <class name="Organism">
    <attribute name="name" type="java.lang.String"/>
  </class>
</model>
""".strip()


PARENT_AGGREGATION_MODEL_XML = """
<model name="mock" package="org.mock">
  <class name="A" />
  <class name="D" />
  <class name="B" extends="A" />
  <class name="C" extends="D" />
</model>
""".strip()


def test_path_hash_and_equality_include_subclass_overrides():
    model = Model(PATH_SUBCLASS_MODEL_XML)
    path_gene_a = model.make_path("Publication.subject.organism", {"Publication.subject": "Gene"})
    path_gene_b = model.make_path("Publication.subject.organism", {"Publication.subject": "Gene"})
    path_protein = model.make_path("Publication.subject.organism", {"Publication.subject": "Protein"})

    assert path_gene_a == path_gene_b
    assert hash(path_gene_a) == hash(path_gene_b)
    assert path_gene_a != path_protein

    cache = {path_gene_a: "gene", path_protein: "protein"}
    assert cache[path_gene_b] == "gene"
    assert len(cache) == 2


def test_path_hash_and_equality_are_order_independent_for_subclass_mapping():
    model = Model(PATH_SUBCLASS_MODEL_XML)
    subclasses_a = {
        "Publication.subject": "Gene",
        "Publication.subject.organism": "Organism",
    }
    subclasses_b = {
        "Publication.subject.organism": "Organism",
        "Publication.subject": "Gene",
    }

    path_a = model.make_path("Publication.subject.organism", subclasses_a)
    path_b = model.make_path("Publication.subject.organism", subclasses_b)

    assert path_a == path_b
    assert hash(path_a) == hash(path_b)


def test_composed_class_parent_aggregation_includes_all_parts():
    model = Model(PARENT_AGGREGATION_MODEL_XML)
    composed = model.get_class("B,C")
    names = [cls.name for cls in composed.parent_classes]

    assert "A" in names
    assert "D" in names
    assert names[-2:] == ["B", "C"]


def test_composed_class_parent_aggregation_is_stable_once_computed():
    model = Model(PARENT_AGGREGATION_MODEL_XML)
    composed = model.get_class("B,C")

    first = composed.parent_classes
    second = composed.parent_classes

    assert first == second
    assert first is not second
    assert [cls.name for cls in first] == [cls.name for cls in second]


def test_parser_reports_missing_model_name_or_package_without_assertion_cause():
    invalid_xml = "<model><class name='Gene'/></model>"
    with pytest.raises(ModelParseError) as exc_info:
        Model(invalid_xml)

    message = str(exc_info.value)
    assert "<model>" in message
    assert "name and package" in message
    assert "AssertionError" not in message


def test_parser_reports_missing_class_name_without_assertion_cause():
    invalid_xml = "<model name='mock' package='org.mock'><class /></model>"
    with pytest.raises(ModelParseError) as exc_info:
        Model(invalid_xml)

    message = str(exc_info.value)
    assert "Missing name in <class>" in message
    assert "AssertionError" not in message


def test_model_constructor_rejects_none_source_without_assertion():
    with pytest.raises(ModelParseError) as exc_info:
        Model(None)

    message = str(exc_info.value)
    assert "Model source cannot be None" in message
    assert "AssertionError" not in message


def test_strip_java_prefix_uses_split_semantics_for_edge_cases():
    assert _strip_java_prefix("java.lang.String") == "String"
    assert _strip_java_prefix("String") == "String"
    assert _strip_java_prefix("java.lang.") == ""
    assert _strip_java_prefix(".String") == "String"
    assert _strip_java_prefix("") == ""
    assert _strip_java_prefix(None) == ""
