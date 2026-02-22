from intermine314.model import Model


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


def test_field_slots_keep_extension_flexibility_without_instance_dict():
    model = _build_model()
    field = model.get_class("Gene").get_field("symbol")

    field.custom_label = "primary"

    assert field.custom_label == "primary"
    assert field.extensions["custom_label"] == "primary"
    assert not hasattr(field, "__dict__")


def test_path_slots_keep_extension_flexibility_without_instance_dict():
    model = _build_model()
    path = model.make_path("Gene.organism")

    path.custom_tag = "interactive-note"

    assert path.custom_tag == "interactive-note"
    assert path.extensions["custom_tag"] == "interactive-note"
    assert not hasattr(path, "__dict__")
