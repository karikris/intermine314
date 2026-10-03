"""Object contracts exercised against the shared offline HTTP transport."""
import json
from urllib.parse import parse_qs
from xml.etree import ElementTree

import pytest

from intermine314 import results
from intermine314.model import Model, ModelError
from intermine314.query import Query
from intermine314.service import Service
from intermine314.webservice import Service as LegacyService
from tests.fixtures.compatibility import FixtureSession, fixture_bytes

ROOT = "https://offline.example/service"
VIEWS = ["Department.name", "Department.company.vatNumber", "Department.employees.age", "Department.employees.name"]


def payload(*objects):
    return ('{"results":[\n' + ',\n'.join(json.dumps(obj) for obj in objects) + '\n],"wasSuccessful":true}\n').encode()


def wire(session):
    return parse_qs(session.requests[-1].data.decode())


@pytest.mark.parametrize("client", [Service, LegacyService])
@pytest.mark.parametrize("mode", ["object", "objects", "objectformat", "jsonobjects"])
def test_explicit_object_aliases_schema_nested_types_cache_and_closure(client, mode):
    session = FixtureSession.service(rows=fixture_bytes("objects-nested.json"))
    with client(ROOT, session=session) as service:
        query = service.select(*VIEWS)
        stream = query.results(mode, start=2, size=3)
        department = next(stream)
        assert isinstance(department, results.ResultObject)
        assert stream.cld is service.model.get_class("Department")
        assert query.to_spec().root_class == "Department"
        assert department.id == 3000008 and department.type == "Department"
        assert department.company.vatNumber == 665261
        assert department.company is department.company
        assert department.employees is department.employees
        assert len(department.employees) == 6
        manager = department.employees[4]
        assert manager.type == "Manager" and manager._cld is service.model.get_class("Manager")
        assert str(department) == "Department(name = 'Sales')"
        assert "company = Company(vatNumber = 665261)" in repr(department)
        with pytest.raises(ModelError, match="no field"):
            department.unknown
        assert wire(session)["format"] == ["jsonobjects"]
        assert wire(session)["start"] == ["2"] and wire(session)["size"] == ["3"]
        stream.close()
        assert all(response.closed for response in session.responses)
    assert session.close_calls == 0


@pytest.mark.parametrize("client,legacy", [(Service, False), (LegacyService, True)])
def test_profile_defaults_and_direct_model_query(client, legacy):
    session = FixtureSession.service(rows=payload({"class": "Employee", "name": "A"}))
    with client(ROOT, session=session) as service:
        query = service.select("Employee.name")
        for stream in (query.results(), iter(query)):
            try:
                value = next(stream)
                assert isinstance(value, results.ResultObject if legacy else dict)
                assert wire(session)["format"] == ["jsonobjects" if legacy else "json"]
            finally:
                stream.close()
        session.routes[("POST", "/service/query/results")] = payload(["A"])
        stream = query.rows()
        assert isinstance(next(stream), results.ResultRow if legacy else dict)
        stream.close()
        direct = Query(service.model, service=service, root="Employee").select("name")
        assert direct.compatibility == "legacy"
        session.routes[("POST", "/service/query/results")] = payload({"name": "A"})
        stream = direct.results()
        assert next(stream).type == "Employee"
        stream.close()


def test_selected_missing_nulls_noncontiguous_nested_groups_and_no_id_do_not_fetch():
    session = FixtureSession.service()
    with LegacyService(ROOT, session=session) as service:
        cld = service.model.get_class("Department")
        data = {"objectId": 1, "name": None, "company": {"objectId": 2, "CEO": {}}, "employees": None}
        view = ["Department.company.name", "Department.name", "Department.company.CEO.name", "Department.employees.name", "Department.company.vatNumber", "Department.manager.name"]
        obj = results.ResultObject(data, cld, view)
        before = len(session.requests)
        assert obj.name is None and obj.employees == [] and obj.manager is None
        assert obj.company.name is None and obj.company.vatNumber is None
        assert obj.company.CEO.name is None
        assert obj._attr_cache["name"] is None
        assert obj.company._attr_cache["vatNumber"] is None
        no_id = results.ResultObject({}, cld)
        assert no_id.name is None and no_id.company is None and no_id.employees == []
        assert no_id.id is None and no_id.type == "Department"
        assert len(session.requests) == before


def test_loaded_nulls_are_cached_without_fetch_even_when_not_selected():
    model = Model(fixture_bytes("model.xml"))
    obj = results.ResultObject({"objectId": 1, "name": None, "company": None, "employees": None}, model.get_class("Department"))
    assert obj.name is None and obj.company is None and obj.employees == []
    assert obj._attr_cache == {"name": None, "company": None, "employees": []}


@pytest.mark.parametrize("field,value", [("name", "Fetched"), ("name", None), ("company", {"name": "Company"}), ("employees", [{"name": "One"}, {"name": "Two"}]), ("company", None)])
def test_lazy_fetch_actual_query_id_filter_complete_collection_and_early_close(field, value):
    session = FixtureSession.service(rows=payload({"objectId": 1, field: value}, {"objectId": 2, field: value}))
    with LegacyService(ROOT, session=session) as service:
        obj = results.ResultObject({"objectId": 1}, service.model.get_class("Department"))
        fetched = getattr(obj, field)
        assert getattr(obj, field) is fetched
        if field == "employees":
            assert [employee.name for employee in fetched] == ["One", "Two"]
        elif field == "company" and value is not None:
            assert fetched.name == "Company"
        else:
            assert fetched == value
        requests = [request for request in session.requests if request.method == "POST"]
        assert len(requests) == 1
        params = wire(session)
        assert "size" not in params and params["format"] == ["jsonobjects"]
        xml = ElementTree.fromstring(params["query"][0])
        assert xml.find("constraint").attrib["value"] == "1"
        assert xml.find("constraint").attrib["path"] == "Department.id"
        assert all(response.closed for response in session.responses)


def test_empty_lazy_result_is_cached():
    session = FixtureSession.service(rows=payload())
    with LegacyService(ROOT, session=session) as service:
        obj = results.ResultObject({"objectId": 1}, service.model.get_class("Department"))
        assert obj.name is None and obj.name is None
        assert len([r for r in session.requests if r.method == "POST"]) == 1
        assert all(response.closed for response in session.responses)


def test_cycle_repr_is_bounded_and_never_fetches():
    model = Model(fixture_bytes("model.xml"))
    data = {"objectId": 1, "name": "D"}
    employee = {"objectId": 2, "name": "E", "department": data}
    data["employees"] = [employee]
    obj = results.ResultObject(data, model.get_class("Department"))
    assert obj.employees[0].department is obj
    assert repr(obj) == "Department(name = 'D', employees = [Employee(name = 'E', department = ...)])"
    assert str(obj) == "Department(name = 'D')"


def test_empty_view_prefetch_and_constraint_augmentation_match_wire_on_clone():
    session = FixtureSession.service(rows=payload({"name": "D"}))
    with LegacyService(ROOT, session=session, prefetch_depth=2, prefetch_id_only=True) as service:
        query = Query(service.model, service=service, root="Department")
        original = query.to_xml()
        stream = query.results()
        next(stream)
        xml = ElementTree.fromstring(wire(session)["query"][0])
        assert "Department.company.id" in xml.attrib["view"].split()
        assert xml.find("join").attrib["style"] == "OUTER"
        assert stream.view == xml.attrib["view"].split()
        stream.close()
        assert query.to_xml() == original and query.views == []
        query = service.select("Department.name").where("company.name", "=", "A").where("employees", "LOOKUP", "B")
        original = query.to_xml()
        stream = query.results()
        next(stream)
        xml = ElementTree.fromstring(wire(session)["query"][0])
        assert stream.view == ["Department.name", "Department.company.name", "Department.employees.id"]
        assert stream.view == xml.attrib["view"].split()
        stream.close()
        assert query.to_xml() == original


@pytest.mark.parametrize("cld", [None, "Department"])
def test_service_historical_get_results_normalizes_actual_class(cld):
    session = FixtureSession.service(rows=fixture_bytes("objects-nested.json"))
    with Service(ROOT, session=session) as service:
        stream = service.get_results("/query/results", {"query": "xml"}, "objects", VIEWS, cld)
        assert stream.cld is service.model.get_class("Department")
        assert next(stream).name == "Sales"
        stream.close()


@pytest.mark.parametrize("data", [["bad"], {"class": "Unknown"}])
def test_object_parser_errors_close_response(data):
    session = FixtureSession.service(rows=payload(data))
    with LegacyService(ROOT, session=session) as service:
        stream = service.select("Department.name").results()
        with pytest.raises((ModelError, TypeError, ValueError)):
            next(stream)
        assert session.responses[-1].closed


@pytest.mark.parametrize("client", [Service, LegacyService])
def test_owned_analytics_keep_dictionary_wire_and_decimal_export(client, tmp_path):
    from decimal import Decimal

    from intermine314.export import query_parquet

    session = FixtureSession.service(rows=b'{"results":[\n["A",1.234567890123456789]\n],"wasSuccessful":true}\n')
    with client(ROOT, session=session) as service:
        query = service.select("Types.name", "Types.bigDecimalObjType")
        assert [row for row in query.iter_rows()][0]["Types.name"] == "A"
        assert wire(session)["format"] == ["json"]
        target = tmp_path / "objects-profile.parquet"
        query.export(target, size=1)
        assert wire(session)["format"] == ["json"]
        frame = query_parquet(target)
        assert frame["Types.bigDecimalObjType"].to_list() == [Decimal("1.234567890123456789")]
        assert all(response.closed for response in session.responses)


@pytest.mark.parametrize("where,selected", [("company", "Department.companyCode"), ("company.name", "Department.companyCode")])
def test_constraint_augmentation_respects_segment_boundaries(where, selected):
    source = b'<model name="test" package="test"><class name="Department"><attribute name="companyCode" type="String"/><reference name="company" referenced-type="Company"/></class><class name="Company"><attribute name="name" type="String"/></class></model>'
    session = FixtureSession.service(rows=payload({"companyCode": "C"}))
    session.routes[("GET", "/service/model")] = source
    with LegacyService(ROOT, session=session) as service:
        query = service.select(selected).where(where, "IS NOT NULL")
        stream = query.results()
        next(stream)
        assert stream.view == [selected, "Department." + where + (".id" if where == "company" else "")]
        stream.close()
        assert query.views == [selected]


def test_missing_schema_requests_raise_public_errors_before_result_http():
    from types import SimpleNamespace

    from intermine314.service.session import ResultIterator

    session = FixtureSession.service()
    with Service(ROOT, session=session) as service:
        query = Query(SimpleNamespace(name="model"), service=service, root="Department", validate=False)
        query.add_view("Department.name")
        with pytest.raises(ModelError, match="valid query model"):
            query.results("object")
        service._model = SimpleNamespace(name="model")
        with pytest.raises(ModelError, match="valid service model"):
            service.get_results("/query/results", {}, "object", VIEWS, "Department")
        with pytest.raises(ModelError, match="Class descriptor"):
            ResultIterator(service, "/query/results", {}, "objects", VIEWS)
        assert not any(request.method == "POST" for request in session.requests)


@pytest.mark.parametrize("version", [7, 8])
def test_object_exhaustion_and_independent_stream_close(version):
    session = FixtureSession.service(version=version, rows=fixture_bytes("objects-nested.json"))
    with LegacyService(ROOT, session=session) as service:
        facade = service.select(*VIEWS).results()
        first, second = iter(facade), iter(facade)
        assert next(first).name == "Sales"
        first.close()
        assert session.responses[-2].closed and not session.responses[-1].closed
        assert len([obj for obj in second]) == 8
        assert session.responses[-1].closed
        assert wire(session)["format"] == ["jsonobjects"]
        early = iter(facade)
        facade.close()
        assert session.responses[-1].closed
        assert next(early, None) is None


@pytest.mark.parametrize("field", ["name", "company", "employees"])
def test_lazy_errors_close_and_are_not_cached(field):
    session = FixtureSession.service(rows=payload(["invalid"]))
    with LegacyService(ROOT, session=session) as service:
        obj = results.ResultObject({"objectId": 1}, service.model.get_class("Department"))
        with pytest.raises(TypeError, match="JSON object"):
            getattr(obj, field)
        assert session.responses[-1].closed and field not in obj._attr_cache
        session.routes[("POST", "/service/query/results")] = payload({field: None})
        assert getattr(obj, field) == ([] if field == "employees" else None)
        assert session.responses[-1].closed


def test_idless_class_does_not_fetch_even_with_object_id():
    session = FixtureSession.service()
    with LegacyService(ROOT, session=session) as service:
        obj = results.ResultObject({"objectId": 1}, service.model.get_class("SimpleObject"))
        assert obj.name is None and obj.employee is None
        assert not any(request.method == "POST" for request in session.requests)


def test_query_count_len_and_result_path_use_bound_service():
    session = FixtureSession.service(rows=b"42\n")
    with LegacyService(ROOT, session=session) as service:
        query = service.select("Department.name")
        assert query.get_results_path() == service.QUERY_PATH
        assert len(query) == 42
        assert session.responses[-1].closed


def test_query_model_descriptor_and_lazy_fetch_survive_different_service_model():
    session = FixtureSession.service(rows=payload({"objectId": 1, "name": "Custom"}))
    with Service(ROOT, session=session) as service:
        assert "Special" not in service.model.classes
        model = Model(b'<model name="custom" package="custom"><class name="Special"><attribute name="name" type="String"/><attribute name="extra" type="String"/></class></model>')
        query = Query(model, service=service, root="Special").select("name")
        stream = query.results("objects")
        obj = next(stream)
        stream.close()
        assert obj._cld is model.get_class("Special") and obj.name == "Custom"
        assert stream.cld is obj._cld
        session.routes[("POST", "/service/query/results")] = payload({"extra": "Fetched"})
        assert obj.extra == "Fetched"
        assert ElementTree.fromstring(wire(session)["query"][0]).attrib["view"] == "Special.extra"
        assert all(response.closed for response in session.responses)


def test_composed_returned_class_keeps_wire_type_and_all_descriptors():
    model = Model(fixture_bytes("model.xml"))
    obj = results.ResultObject({"class": "Employee,ImportantPerson", "age": 42, "seniority": 7}, model.get_class("Employable"))
    assert obj.type == "Employee,ImportantPerson"
    assert obj._cld.isa("Employee") and obj._cld.isa("ImportantPerson")
    assert obj.age == 42 and obj.seniority == 7
    assert str(obj) == "Employee_ImportantPerson(age = 42,  seniority = 7)"
    assert repr(obj) == "Employee_ImportantPerson(age = 42, seniority = 7)"


def test_composed_class_lazy_fetch_uses_declaring_schema_class():
    session = FixtureSession.service(rows=payload({"seniority": 7}))
    with LegacyService(ROOT, session=session) as service:
        obj = results.ResultObject({"class": "Employee,ImportantPerson", "objectId": 1}, service.model.get_class("Employee"))
        assert obj.seniority == 7
        xml = ElementTree.fromstring(wire(session)["query"][0])
        assert xml.attrib["view"] == "ImportantPerson.seniority"
        assert xml.find("constraint").attrib["value"] == "1"
        assert session.responses[-1].closed
