from intermine314.model import Column, Model, Path


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
