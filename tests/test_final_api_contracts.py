"""Scoped public-call checks found missing during the final symbol audit.

Expected HTTP callback messages and Template paths follow pinned 1.13.0
results.py:688-789 and query.py:1954. Destruction is an explicit departure.
"""
from io import BytesIO
from urllib.parse import parse_qs, urlencode, urlsplit
from xml.etree import ElementTree as ET

import pytest

from intermine314.constraints import BinaryConstraint, LogicNode
from intermine314.errors import UnimplementedError, WebserviceError
from intermine314.model import Model, ModelError
from intermine314.pathfeatures import Join, PathDescription
from intermine314.query import ConstraintError, Query, QueryError, Template
from intermine314.results import (
    InterMineURLOpener,
    decode_binary,
    encode_dict,
    encode_str,
)
from intermine314.webservice import Service, ensure_str
from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession, fixture_bytes


def test_query_string_and_template_resource_path_are_public_calls():
    session = FixtureSession.service()
    with Service(SERVICE_ROOT, session=session) as service:
        query = service.select("Employee.name").where("Employee.age", ">", 20)
        xml = ET.fromstring(str(query))
        assert xml.attrib["view"] == "Employee.name"
        assert xml.find("constraint").attrib["value"] == "20"
        template = Template(service.model, service=service, root="Employee")
        assert template.get_results_path() == "/template/results"
    assert session.close_calls == 0


def test_clear_view_removes_all_selected_paths_without_losing_constraints():
    query = Query(Model(fixture_bytes("model.xml")), root="Employee").select("name", "age")
    query.add_constraint("age", ">", 20)
    query.clear_view()
    assert query.views == []
    assert query.get_constraint("A").value == 20
    query.add_view("fullTime")
    assert query.views == ["Employee.fullTime"]


@pytest.mark.parametrize("code,prefix", [
    (400, "There was a problem with our request"),
    (401, "No permissions - not logged in"),
    (403, "No permissions - not logged in"),
    (404, "Missing resource"),
    (500, "Internal server error"),
    (418, None),
])
def test_public_http_error_callbacks_preserve_exception_arguments_and_close(code, prefix):
    session = FixtureSession.service()
    with Service(SERVICE_ROOT, session=session) as service:
        body = BytesIO(b'{"error":"missing"}')
        callback = getattr(service.opener, f"http_error_{code}", service.opener.http_error_default)
        with pytest.raises(WebserviceError) as raised:
            callback(SERVICE_ROOT, body, code, "reason", {})
        # IOError/OSError stores only its first two positional arguments in args;
        # the third is filename (the upstream callback passes reason there).
        assert raised.value.args == ((code, "reason") if prefix is None else (prefix, code))
        assert raised.value.filename == (b'{"error":"missing"}' if prefix is None else "reason")
        assert body.closed


def test_logic_base_protocol_and_unimplemented_exception_are_usable():
    a = BinaryConstraint("Employee.name", "=", "Ada", code="A")
    b = BinaryConstraint("Employee.name", "=", "Bob", code="B")
    assert isinstance(a, LogicNode)
    assert str(LogicNode.__and__(a, b)) == "A and B"
    assert str(LogicNode.__or__(a, b)) == "A or B"
    with pytest.raises(UnimplementedError, match="unsupported") as raised:
        raise UnimplementedError("unsupported")
    assert raised.value.args == ("unsupported",)


@pytest.mark.parametrize("owned", [False, True])
def test_service_destructor_closes_owned_session_without_network_list_deletion(monkeypatch, owned):
    session = FixtureSession.service()
    if owned:
        monkeypatch.setattr("intermine314.service.session.build_session", lambda **kwargs: session)
        service = Service(SERVICE_ROOT)
    else:
        service = Service(SERVICE_ROOT, session=session)
    manager = service.list_manager()
    manager._temp_lists.add("temporary-server-list")
    before = len(session.requests)
    service.__del__()
    service.__del__()
    assert len(session.requests) == before
    assert manager._temp_lists == {"temporary-server-list"}
    assert session.close_calls == int(owned)


def test_deep_ancestry_avoids_accidental_expansion_but_retains_diamond_branches():
    # Upstream model.py:945-948 iterates the list it extends: A->B->C->D
    # yields [B, C, D, D]. The repaired traversal visits only direct parents.
    model = Model('''<model name="lineage" package="test">
      <class name="A" extends="B"/><class name="B" extends="C"/>
      <class name="C" extends="D"/><class name="D" extends="java.lang.Object"/>
      <class name="Left" extends="D"/><class name="Right" extends="D"/>
      <class name="Diamond" extends="Left Right"/>
    </model>''')
    assert model.to_ancestry(model.get_class("A")) == [model.get_class(n) for n in ("B", "C", "D")]
    assert model.to_ancestry(model.get_class("D")) == []
    assert model.to_ancestry(model.get_class("Diamond")) == [model.get_class(n) for n in ("Left", "Right", "D", "D")]
    model.get_class("D").parents = ["A"]
    with pytest.raises(ModelError, match="Inheritance cycle"):
        model.to_ancestry(model.get_class("A"))


@pytest.mark.parametrize("value,expected", [("Å", "Å"), (b"utf8", b"utf8"), (12, "12"), (None, "None"), (True, "True")])
def test_public_encode_str_preserves_text_bytes_and_stringifies_scalars(value, expected):
    result = encode_str(value)
    assert result == expected and type(result) is type(expected)


def test_public_encode_dict_normalizes_sequences_for_doseq_without_mutating_input():
    data = {"text": "Å", b"binary": b"raw", "number": 12, "null": None, "list": ["Å", 3, None], "tuple": (b"raw", False), "empty": []}
    encoded = encode_dict(data)
    assert encoded == {"text": "Å", b"binary": b"raw", "number": "12", "null": "None", "list": ["Å", "3", "None"], "tuple": [b"raw", "False"], "empty": []}
    assert data["list"] == ["Å", 3, None] and data["tuple"] == (b"raw", False)
    assert parse_qs(urlencode(encoded, doseq=True)) == {"text": ["Å"], "binary": ["raw"], "number": ["12"], "null": ["None"], "list": ["Å", "3", "None"], "tuple": ["raw", "False"]}


def test_public_decode_and_ensure_string_helpers_preserve_their_distinct_types():
    assert decode_binary("Å".encode()) == "Å"
    assert decode_binary(bytearray("Å".encode())) == "Å"
    assert decode_binary(None) is None and decode_binary(12) == 12
    assert ensure_str("Å".encode()) == "Å"
    assert ensure_str("Å") == "Å" and ensure_str(12) == "12"


def test_public_headers_use_standard_user_agent_and_optional_auth_content_headers():
    session = FixtureSession.service()
    with InterMineURLOpener(session=session) as opener:
        assert opener.headers() == {"User-Agent": opener.USER_AGENT}
    with InterMineURLOpener(session=session, token="offline", user_agent="custom-agent") as opener:
        assert opener.headers("text/plain", "application/json") == {
            "User-Agent": "custom-agent", "Authorization": opener.auth_header,
            "Content-Type": "text/plain", "Accept": "application/json",
        }
        assert "UserAgent" not in opener.headers()
    assert not session.requests and session.close_calls == 0


def test_public_opener_read_returns_decoded_text_and_closes_the_response():
    session = FixtureSession({("GET", "/service/text"): "Å + &".encode()})
    with InterMineURLOpener(session=session) as opener:
        result = opener.read(SERVICE_ROOT + "/text")
        assert result == "Å + &" and isinstance(result, str)
        assert session.responses[0].closed
    assert session.close_calls == 0


@pytest.mark.parametrize("url", [SERVICE_ROOT + "/query/results", SERVICE_ROOT + "/query/results?format=json"])
def test_public_prepare_url_adds_encoded_token_and_retains_existing_query(url):
    session = FixtureSession.service()
    with InterMineURLOpener(session=session, token="Å &+") as opener:
        expected = {"token": ["Å &+"]}
        if "?" in url:
            expected["format"] = ["json"]
        prepared = opener.prepare_url(url)
        assert parse_qs(urlsplit(prepared).query) == expected
        assert urlsplit(prepared).path == "/service/query/results"
    with InterMineURLOpener(session=session) as opener:
        assert opener.prepare_url(url) == url
    assert not session.requests


@pytest.mark.parametrize("method,valid,invalid,error", [
    ("verify_views", ["Employee.name"], ["Employee.department"], ConstraintError),
    ("verify_join_paths", [Join("Employee.department")], [Join("Employee.name")], QueryError),
    ("verify_pd_paths", [PathDescription("Employee.name", "Name")], [PathDescription("Employee.missing", "Missing")], ModelError),
])
def test_public_path_verifiers_accept_valid_and_reject_invalid_model_paths(method, valid, invalid, error):
    query = Query(Model(fixture_bytes("model.xml")), root="Employee")
    assert getattr(query, method)(valid) is None
    with pytest.raises(error):
        getattr(query, method)(invalid)
