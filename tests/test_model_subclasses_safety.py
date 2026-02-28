from intermine314.model import Column, Model, Path
from intermine314.model.constants import _PATH_SEGMENT_CACHE_MAXSIZE
from intermine314.model import helpers as model_helpers


MODEL_XML = """
<model name="mock" package="org.mock">
  <class name="Gene">
    <reference name="organism" referenced-type="Organism"/>
    <attribute name="symbol" type="java.lang.String"/>
  </class>
  <class name="Organism">
    <attribute name="name" type="java.lang.String"/>
  </class>
</model>
""".strip()


def _build_model():
    return Model(MODEL_XML)


def test_path_default_subclasses_dict_is_not_shared_between_instances():
    model = _build_model()
    path_a = Path("Gene.organism", model)
    path_b = Path("Gene.organism", model)

    assert path_a.subclasses is not path_b.subclasses
    path_a.subclasses["Gene.organism"] = "Organism"
    assert "Gene.organism" not in path_b.subclasses


def test_column_default_subclasses_dict_is_not_shared_between_instances():
    model = _build_model()
    column_a = Column("Gene", model)
    column_b = Column("Gene", model)

    assert column_a._subclasses is not column_b._subclasses
    column_a._subclasses["Gene.organism"] = "Organism"
    assert "Gene.organism" not in column_b._subclasses


def test_path_and_column_copy_caller_subclasses_mapping():
    model = _build_model()
    shared = {"Gene.organism": "Organism"}

    path = Path("Gene.organism.name", model, shared)
    column = Column("Gene", model, shared)

    assert path.subclasses is not shared
    assert column._subclasses is not shared

    shared["Gene"] = "Organism"
    assert "Gene" not in path.subclasses
    assert "Gene" not in column._subclasses


def test_column_subclass_mutation_does_not_leak_to_caller_mapping():
    model = _build_model()
    shared = {}

    root = Column("Gene", model, shared)
    _ = root.organism < Column("Organism", model)

    assert shared == {}
    assert root._subclasses.get("Gene.organism") == "Organism"


def test_make_path_copies_caller_subclasses_mapping():
    model = _build_model()
    shared = {"Gene.organism": "Organism"}

    path = model.make_path("Gene.organism.name", shared)
    shared["Gene.organism"] = "Gene"

    assert path.subclasses["Gene.organism"] == "Organism"


def test_parse_path_string_supports_nested_subclass_overrides():
    model_xml = """
    <model name="mock" package="org.mock">
      <class name="Publication">
        <reference name="subject" referenced-type="BioEntity"/>
      </class>
      <class name="BioEntity" />
      <class name="Gene" extends="BioEntity">
        <reference name="organism" referenced-type="Organism"/>
      </class>
      <class name="Organism">
        <attribute name="name" type="java.lang.String"/>
      </class>
    </model>
    """.strip()
    model = Model(model_xml)

    parts = model.parse_path_string(
        "Publication.subject.organism.name",
        {"Publication.subject": "Gene"},
    )

    assert [part.name for part in parts] == ["Publication", "subject", "organism", "name"]


def test_prefix_and_append_reuse_existing_parts_without_reparse(monkeypatch):
    model = _build_model()
    path = model.make_path("Gene.organism")

    def _fail_reparse(*_args, **_kwargs):
        raise AssertionError("parse_path_string should not be called")

    monkeypatch.setattr(model, "parse_path_string", _fail_reparse)
    prefixed = path.prefix()
    appended = path.append("name")

    assert str(prefixed) == "Gene"
    assert [part.name for part in prefixed.parts] == ["Gene"]
    assert str(appended) == "Gene.organism.name"
    assert [part.name for part in appended.parts] == ["Gene", "organism", "name"]


def test_append_honors_subclass_override_without_reparse(monkeypatch):
    model_xml = """
    <model name="mock" package="org.mock">
      <class name="Publication">
        <reference name="subject" referenced-type="BioEntity"/>
      </class>
      <class name="BioEntity" />
      <class name="Gene" extends="BioEntity">
        <reference name="organism" referenced-type="Organism"/>
      </class>
      <class name="Organism">
        <attribute name="name" type="java.lang.String"/>
      </class>
    </model>
    """.strip()
    model = Model(model_xml)
    path = model.make_path("Publication.subject", {"Publication.subject": "Gene"})

    def _fail_reparse(*_args, **_kwargs):
        raise AssertionError("parse_path_string should not be called")

    monkeypatch.setattr(model, "parse_path_string", _fail_reparse)
    appended = path.append("organism", "name")

    assert str(appended) == "Publication.subject.organism.name"
    assert [part.name for part in appended.parts] == ["Publication", "subject", "organism", "name"]


def test_parse_path_string_segment_cache_is_bounded_and_records_hits():
    model_helpers._split_path_segments.cache_clear()
    model = _build_model()

    model.parse_path_string("Gene.organism.name")
    model.parse_path_string("Gene.organism.name")

    info = model_helpers._split_path_segments.cache_info()
    assert info.maxsize == _PATH_SEGMENT_CACHE_MAXSIZE
    assert info.hits >= 1

    for idx in range(info.maxsize + 200):
        model_helpers._split_path_segments(f"Gene{idx}.symbol")

    bounded_info = model_helpers._split_path_segments.cache_info()
    assert bounded_info.currsize <= bounded_info.maxsize
