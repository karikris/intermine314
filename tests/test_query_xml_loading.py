"""Offline XML import contracts against the pinned original client model."""
from io import BytesIO, StringIO
from urllib.parse import parse_qs
from xml.dom import minidom
from xml.etree import ElementTree as ET

import pytest

from intermine314.model import Model, ModelError
from intermine314.query import ConstraintError, Query, QueryError, QueryParseError
from intermine314.query.constraints import LogicParseError
from tests.fixtures.compatibility import FixtureConnection, fixture_bytes

BASIC = '<query name="saved" longDescription="Müller &amp; Ada" view="Employee.name Employee.age" />'


@pytest.fixture
def model():
    return Model(fixture_bytes("model.xml"))


@pytest.mark.parametrize("kind", ["text", "bytes", "file", "path", "borrowed-text", "borrowed-bytes"])
def test_load_sources_and_borrowed_ownership(model, tmp_path, kind):
    path = tmp_path / "saved-query.txt"
    path.write_text(BASIC, encoding="utf-8")
    sources = {"text": BASIC, "bytes": BASIC.encode(), "file": str(path), "path": path,
               "borrowed-text": StringIO(BASIC), "borrowed-bytes": BytesIO(BASIC.encode())}
    source = sources[kind]
    query = Query.from_xml(source, model)
    assert query.name == "saved" and query.description == "Müller & Ada"
    assert query.views == ["Employee.name", "Employee.age"]
    assert query.root.name == "Employee" and query.compatibility == "legacy"
    assert query.do_verification
    if hasattr(source, "read"):
        assert not source.closed


@pytest.mark.parametrize("payload", [BASIC.encode(), b"<query>"])
def test_owned_url_and_local_resources_close_even_on_parse_error(model, monkeypatch, payload):
    from intermine314.query import builder
    connections = []

    def opened(source):
        assert source in ("https://offline.example/saved.xml", "saved.txt")
        connection = FixtureConnection(payload)
        connections.append(connection)
        return connection

    monkeypatch.setattr(builder, "openAnything", opened, raising=False)
    for source in ("https://offline.example/saved.xml", "saved.txt"):
        if payload == BASIC.encode():
            assert Query.from_xml(source, model).name == "saved"
        else:
            with pytest.raises(QueryParseError) as caught:
                Query.from_xml(source, model)
            assert caught.value.__cause__ is not None
        assert connections[-1].closed and connections[-1].close_calls == 1


@pytest.mark.parametrize("xml,count", [("<queries/>", 0), ("<queries><query/><query/></queries>", 2)])
def test_exact_one_query_required(model, xml, count):
    with pytest.raises(QueryParseError, match=f"Only one <query> element is allowed. Found {count}"):
        Query.from_xml(xml, model)


def test_borrowed_parse_failure_stays_open(model):
    source = StringIO("<query>")
    with pytest.raises(QueryParseError) as caught:
        Query.from_xml(source, model)
    assert caught.value.__cause__ is not None and not source.closed


def test_real_model_and_constraint_errors_survive(model):
    with pytest.raises(ModelError):
        Query.from_xml('<query view="Employee.missing"/>', model)
    with pytest.raises(ConstraintError, match="attribute"):
        Query.from_xml('<query view="Employee.name"><constraint path="Employee.department" op="=" value="x"/></query>', model)
    with pytest.raises(QueryParseError, match="Constraints must have a path"):
        Query.from_xml('<query view="Employee.name"><constraint op="=" value="x"/></query>', model)
    with pytest.raises(ConstraintError) as caught:
        Query.from_xml('<query view="Employee.name"><constraint path="Employee.name" op="BOGUS" value="x"/></query>', model)
    assert isinstance(caught.value.__cause__, TypeError)


def test_subclass_constraints_precede_final_selection_validation(model):
    xml = '''<query view="Employee.title Employee.name">
      <pathDescription pathString="Employee.title" description="Position"/>
      <constraint path="Employee.title" op="=" value="Lead"/>
      <constraint path="Employee" type="Manager"/>
    </query>'''
    query = Query.from_xml(xml, model)
    assert query.get_subclass_dict() == {"Employee": "Manager"}
    assert query.get_constraint("A").value == "Lead"
    assert query.path_descriptions[0].path == "Employee.title"
    query.verify()


def test_explicit_codes_reserve_future_allocations_and_empty_value(model):
    xml = '''<query view="Employee.name Employee.age" constraintLogic="A or (B and C)">
      <constraint path="Employee.name" op="=" value="" editable="true" switchable="on"/>
      <constraint path="Employee.age" op="&gt;" value="5" code="A"/>
      <constraint path="Employee.name" op="IS NOT NULL"/>
    </query>'''
    query = Query.from_xml(xml, model)
    assert query.get_constraint("B").value == ""
    assert query.get_constraint("A").value == "5"
    assert str(query.logic) == "A or (B and C)"
    assert query.add_constraint("Employee.age", "<", 20).code == "D"
    xml = query.to_xml()
    assert ET.fromstring(xml).find("constraint[@code='B']").get("value") == ""
    with pytest.raises(ConstraintError, match="already in use"):
        Query.from_xml('''<query view="Employee.name"><constraint path="Employee.name" op="=" value="x" code="A"/>
        <constraint path="Employee.name" op="=" value="y" code="A"/></query>''', model)


@pytest.mark.parametrize("compatibility", ["native", "legacy"])
def test_all_constraint_variants_round_trip_and_parent_node_path(model, compatibility):
    xml = '''<query view="Employee.name Employee.age" constraintLogic="A and (B or C) and D and E and F and G and H">
      <constraint path="Employee.name" op="=" value="x" code="A"/>
      <constraint path="Employee.age" op="IS NOT NULL" code="B"/>
      <constraint path="Employee.name" op="ONE OF" code="C"><value></value><value>Müller &amp; Ada</value></constraint>
      <constraint path="Employee" op="IN" value="saved employees" code="D"/>
      <constraint path="Employee.department" op="!=" loopPath="Employee.departmentThatRejectedMe" code="E"/>
      <constraint path="Employee" op="LOOKUP" value="Ada" extraValue="Company" code="F"/>
      <constraint path="Employee.age" op="OVERLAPS" code="G"><value>1..10</value></constraint>
      <node path="Employee"><constraint op="ISA" code="H"><value>Manager</value><value>CEO</value></constraint></node>
      <join path="Employee.department" style="OUTER"/>
    </query>'''
    query = Query.from_xml(xml, model, compatibility=compatibility)
    assert query.get_constraint("C").values == ["", "Müller & Ada"]
    assert query.get_constraint("D").list_name == "saved employees"
    assert query.get_constraint("E").op == "IS NOT"
    assert query.get_constraint("F").extra_value == "Company"
    assert query.get_constraint("G").values == ["1..10"]
    assert query.get_constraint("H").values == ["Manager", "CEO"]
    again = Query.from_xml(query.to_xml(), model, compatibility=compatibility)
    assert again.to_xml() == query.to_xml()


def test_xml_sorts_ignore_irrelevant_paths_and_keep_selected_directions(model):
    query = Query.from_xml('''<query view="Employee.name Employee.age"
       sortOrder="Employee.nonexistent desc Employee.age desc Employee.name asc"/>''', model)
    assert str(query.get_sort_order()) == "Employee.age desc Employee.name asc"
    assert str(Query.from_xml('<query view="Employee.name" sortOrder="Employee.name"/>', model).get_sort_order()) == "Employee.name asc"


@pytest.mark.parametrize("logic", ["A and Z", "A or", "(A", "A B"])
def test_invalid_xml_logic_has_public_error_without_internal_attribute_crash(model, logic):
    xml = f'<query view="Employee.name" constraintLogic="{logic}"><constraint path="Employee.name" op="=" value="x"/></query>'
    with pytest.raises((QueryError, LogicParseError)) as caught:
        Query.from_xml(xml, model)
    assert not isinstance(caught.value, AttributeError)


def test_path_descriptions_minidom_canonical_wire_and_clone(model):
    from intermine314.pathfeatures import PathDescription
    query = Query(model, root="Employee").select("name")
    pd = query.add_path_description("department", "Team & work")
    assert isinstance(pd, PathDescription) and pd.child_type == "pathDescription"
    assert pd.to_dict() == {"path": "Employee.department", "description": "Team & work"}
    assert pd in query.children()
    node = query.to_Node()
    assert isinstance(node, minidom.Element) and node.ownerDocument is not None
    assert not node.toxml().startswith("<?xml")
    expected = ET.fromstring(query.to_xml()).find("pathDescription").attrib
    assert expected == {"pathString": "Employee.department", "description": "Team & work"}
    assert ET.fromstring(node.toxml()).find("pathDescription").attrib == expected
    assert query.to_spec().path_descriptions == (pd,)
    cloned = query.clone()
    cloned.path_descriptions[0].description = "Changed"
    cloned.add_path_description("name", "Name")
    assert pd.description == "Team & work" and len(query.path_descriptions) == 1
    assert cloned.path_descriptions[0] is not pd
    with pytest.raises(ModelError):
        query.add_path_description("missing", "Missing")
    pd.path = "Employee.missing"
    with pytest.raises(ModelError):
        query.verify()


@pytest.mark.parametrize("attributes", ['pathString="Employee.name"', 'path="Employee.name"', 'path="Employee.name" pathString="Employee.name"'])
def test_path_description_saved_legacy_path_accepted(model, attributes):
    query = Query.from_xml(f'<query view="Employee.name"><pathDescription {attributes} description="Name"/></query>', model)
    assert query.path_descriptions[0].path == "Employee.name"
    assert ET.fromstring(query.to_xml()).find("pathDescription").get("pathString") == "Employee.name"


def test_path_description_conflicts_rejected(model):
    with pytest.raises(QueryParseError, match="Conflicting"):
        Query.from_xml('<query view="Employee.name"><pathDescription path="Employee.age" pathString="Employee.name"/></query>', model)


@pytest.mark.parametrize("factory", ["native_service_factory", "legacy_service_factory"])
def test_service_load_query_binds_executable_profile_prefetch_and_managed_url(request, factory, offline_session_factory, monkeypatch):
    session = offline_session_factory(
        rows=b'{"results":[\n["Ada",42]\n],"wasSuccessful":true,"statusCode":200,"error":null}',
        routes={("GET", "/saved.xml"): BASIC.encode()},
    )
    service = request.getfixturevalue(factory)(session=session, prefetch_depth=2, prefetch_id_only=True,
                                              token="secret", verify_tls="/custom/ca.pem")
    from intermine314.util import resources
    monkeypatch.setattr(resources.request, "urlopen", lambda *a, **kw: pytest.fail("unmanaged URL transport"))
    query = service.load_query("https://offline.example/saved.xml", root="Employee")
    assert query.service is service and query.compatibility == service.compatibility
    assert query.prefetch_depth == 2 and query.prefetch_id_only is True
    assert (query.root.name if query.compatibility == "legacy" else query.root) == "Employee"
    query.add_path_description("name", "Person")
    assert [row for row in query.results("dict")] == [{"Employee.name": "Ada", "Employee.age": 42}]
    loaded = next(r for r in session.requests if r.path == "/saved.xml")
    assert loaded.options["verify"] == "/custom/ca.pem" and "secret" in str(loaded.headers)
    executed = next(r for r in session.requests if r.path == "/service/query/results")
    wire = ET.fromstring(parse_qs(executed.data.decode())["query"][0])
    assert wire.find("pathDescription").get("pathString") == "Employee.name"
    assert all(response.closed for response in session.responses)
    assert session.close_calls == 0


def test_bound_url_parse_failure_closes_owned_response(legacy_service_factory, offline_session_factory):
    session = offline_session_factory(routes={("GET", "/bad.xml"): b"<query>"})
    service = legacy_service_factory(session=session)
    with pytest.raises(QueryParseError):
        service.load_query("https://offline.example/bad.xml")
    assert session.responses[-1].closed


def test_native_model_fallback_and_legacy_model_error(request, offline_session_factory):
    from intermine314.model import ModelParseError
    for factory in ("native_service_factory", "legacy_service_factory"):
        session = offline_session_factory(routes={("GET", "/service/model"): b'<model name="stub" />'})
        service = request.getfixturevalue(factory)(session=session)
        if factory == "native_service_factory":
            query = service.load_query('<query view="Unknown.name"/>')
            assert query.compatibility == "native" and query.service is service
        else:
            with pytest.raises(ModelParseError):
                service.load_query('<query view="Unknown.name"/>')


def test_csv_source_cannot_silently_discard_path_descriptions(model, tmp_path):
    query = Query(model, root="Employee")
    query.add_path_description("Employee", "People")
    with pytest.raises(ValueError, match="conflicts"):
        query.to_parquet(tmp_path / "out.parquet", csv_input=StringIO("name\nAda\n"))


def test_unary_and_subclass_empty_scalar_placeholders_are_ignored(model):
    query = Query.from_xml('''<query view="Employee.name">
      <constraint path="Employee.name" op="IS NOT NULL" value=""/>
      <constraint path="Employee" type="Manager" value=""/>
    </query>''', model)
    assert query.get_constraint("A").op == "IS NOT NULL"
    assert query.get_subclass_dict() == {"Employee": "Manager"}


def test_explicit_code_reservations_and_cloned_allocator_beyond_z(model):
    constraints = ''.join('<constraint path="Employee.name" op="=" value="x"/>' for _ in range(27))
    xml = f'<query view="Employee.name">{constraints}<constraint path="Employee.age" op="&gt;" value="5" code="AA"/></query>'
    query = Query.from_xml(xml, model)
    assert query.get_constraint("AA").path == "Employee.age"
    assert query.get_constraint("AB").path == "Employee.name"
    clone = query.clone()
    assert clone.add_constraint("name", "=", "clone").code == "AC"
    assert query.add_constraint("name", "=", "original").code == "AC"
    assert clone.get_constraint("AC").value == "clone"


def test_bound_url_source_preserves_tor_transport(legacy_service_factory, offline_session_factory):
    session = offline_session_factory(routes={("GET", "/saved.xml"): BASIC.encode()})
    service = legacy_service_factory(session=session, tor=True, proxy_url="socks5h://127.0.0.1:9050")
    query = Query.from_xml("https://offline.example/saved.xml", service.model, service=service)
    assert query.service is service
    loaded = next(r for r in session.requests if r.path == "/saved.xml")
    assert loaded.options["stream"] is True
    assert service.opener._session is session and service.opener.tor_mode
    assert service.opener.proxy_url == "socks5h://127.0.0.1:9050"
    assert session.responses[-1].closed


def test_owned_response_closes_on_semantic_failure(model, monkeypatch):
    from intermine314.query import builder
    stream = FixtureConnection(b'<query view="Employee.missing"/>')
    monkeypatch.setattr(builder, "openAnything", lambda source: stream)
    with pytest.raises(ModelError):
        Query.from_xml("saved.txt", model)
    assert stream.closed


def test_source_open_failure_retains_cause(model, monkeypatch):
    from intermine314.query import builder

    def fail(source):
        raise OSError("unreadable query")

    monkeypatch.setattr(builder, "openAnything", fail)
    with pytest.raises(QueryParseError) as caught:
        Query.from_xml("saved.txt", model)
    assert isinstance(caught.value.__cause__, OSError)


@pytest.mark.parametrize("attributes", ["", 'op=""', 'type=""', 'op="" type=""'])
@pytest.mark.parametrize("model_backed", [True, False])
def test_constraint_xml_requires_operator_or_subclass(model, attributes, model_backed):
    xml = f'<query view="Employee.name"><constraint path="Employee.name" {attributes}/></query>'
    with pytest.raises(ConstraintError, match="operator or subclass"):
        Query.from_xml(xml, model if model_backed else None, compatibility="native")


@pytest.mark.parametrize("compatibility", ["native", "legacy"])
@pytest.mark.parametrize("path,op", [
    ("Employee.name", "ONE OF"), ("Employee.name", "NONE OF"),
    ("Employee.age", "OVERLAPS"), ("Employee.age", "DOES NOT OVERLAP"),
    ("Employee.age", "WITHIN"), ("Employee.age", "OUTSIDE"),
    ("Employee.age", "CONTAINS"), ("Employee.age", "DOES NOT CONTAIN"),
    ("Employee", "ISA"),
])
def test_zero_value_collection_constraints_round_trip(model, compatibility, path, op):
    query = Query(model, compatibility=compatibility, root="Employee").select("name")
    query.add_constraint(path, op, [])
    saved = query.to_xml()
    assert ET.fromstring(saved).find("constraint/value") is None
    loaded = Query.from_xml(saved, model, compatibility=compatibility)
    assert loaded.get_constraint("A").values == []
    assert loaded.get_constraint("A").__class__ is query.get_constraint("A").__class__
    assert loaded.to_xml() == saved


@pytest.mark.parametrize("compatibility", ["native", "legacy"])
def test_empty_where_in_round_trip_and_missing_binary_value_rejected(model, compatibility):
    query = Query(model, compatibility=compatibility, root="Employee").select("name").where_in("name", [])
    assert Query.from_xml(query.to_xml(), model, compatibility=compatibility).get_constraint("A").values == []
    with pytest.raises(ConstraintError, match="Invalid constraint"):
        Query.from_xml('<query view="Employee.name"><constraint path="Employee.name" op="="/></query>',
                       model, compatibility=compatibility)
    binary = Query.from_xml('<query view="Employee.name"><constraint path="Employee.name" op="CONTAINS" value=""/></query>',
                            model, compatibility=compatibility)
    assert binary.get_constraint("A").value == ""


def test_native_service_model_fallback_rejects_constraint_without_operator(native_service_factory, offline_session_factory):
    session = offline_session_factory(routes={("GET", "/service/model"): b'<model name="stub" />'})
    service = native_service_factory(session=session)
    with pytest.raises(ConstraintError, match="operator or subclass"):
        service.load_query('<query view="Employee.name"><constraint path="Employee.name"/></query>')
