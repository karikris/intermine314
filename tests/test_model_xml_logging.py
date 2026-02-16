import logging

import pytest

from intermine314.model import Model, ModelParseError


def test_parse_error_includes_line_and_column_without_full_xml_payload():
    broken_xml = '<model name="mock" package="org.mock"><class name="Gene"></model>'

    with pytest.raises(ModelParseError) as exc_info:
        Model(broken_xml)

    message = str(exc_info.value)
    assert "line=" in message
    assert "column=" in message
    assert "<inline-xml:" in message
    assert broken_xml not in message


def test_debug_logging_uses_metadata_not_full_model_xml(caplog):
    secret = "SENSITIVE_MODEL_XML_TOKEN"
    large_padding = "x" * 600
    model_xml = (
        '<model name="mock" package="org.mock">'
        '<class name="Gene"><attribute name="symbol" type="java.lang.String"/></class>'
        f"<!--{large_padding}{secret}-->"
        "</model>"
    )

    with caplog.at_level(logging.DEBUG, logger="intermine314.model"):
        model = Model(model_xml)

    assert model.name == "mock"
    assert model.package_name == "org.mock"

    messages = "\n".join(record.getMessage() for record in caplog.records if record.name == "intermine314.model")
    assert "Parsing model XML source=" in messages
    assert "sha256=" in messages
    assert secret not in messages
    assert model_xml not in messages
