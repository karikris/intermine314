"""Cardinality and eager helpers through actual managed result requests."""
import json
from urllib.parse import parse_qs
from xml.etree import ElementTree

import pytest

from intermine314.model import Model
from intermine314.query import Query, QueryError
from intermine314.results import ResultObject, ResultRow
from intermine314.service import Service
from intermine314.webservice import Service as LegacyService
from tests.fixtures.compatibility import FixtureSession, fixture_bytes

ROOT = "https://offline.example/service"


def payload(*rows):
    return ('{"results":[\n' + ',\n'.join(json.dumps(row) for row in rows) + '\n],"wasSuccessful":true}\n').encode()


class ResultSession(FixtureSession):
    """Separate count/results routes and observe bounded stream consumption."""

    def __init__(self, rows, count=1, version=8, interrupt=False):
        super().__init__(FixtureSession.service(version=version).routes)
        self.rows = rows
        self.count = count
        self.interrupt = interrupt
        self.lines = []

    def request(self, method, url, data=None, **kwargs):
        if method == "POST" and url.endswith("/query/results"):
            params = parse_qs(data.decode())
            self.routes[(method, "/service/query/results")] = str(self.count).encode() if params["format"] == ["count"] else self.rows
        response = super().request(method, url, data=data, **kwargs)
        if method == "POST" and params["format"] != ["count"]:
            original = response.iter_lines

            def lines(decode_unicode=False):
                for index, line in enumerate(original(decode_unicode)):
                    self.lines.append(line)
                    if self.interrupt and index == 1:
                        raise KeyboardInterrupt
                    yield line

            response.iter_lines = lines
        return response


def requests(session):
    return [parse_qs(request.data.decode()) for request in session.requests if request.method == "POST"]


def assert_closed(session):
    assert all(response.closed for response in session.responses)
    assert session.close_calls == 0


@pytest.mark.parametrize("client", [Service, LegacyService])
@pytest.mark.parametrize("mode", ["jsonobjects", "object", "objects", "objectformat"])
def test_first_object_aliases_keep_complete_collections_start_and_query(client, mode):
    session = ResultSession(fixture_bytes("objects-nested.json"))
    with client(ROOT, session=session) as service:
        query = service.select("Department.name", "Department.employees.name")
        before = query.to_xml(), query.compatibility, query.model
        department = query.first(mode, start=4)
        assert department.name == "Sales" and len(department.employees) == 6
        params, = requests(session)
        assert params["format"] == ["jsonobjects"] and params["start"] == ["4"]
        assert "size" not in params
        assert (query.to_xml(), query.compatibility, query.model) == before
        # The fixture contains eight top-level objects: only the first is read.
        assert len(session.lines) == 2
        assert_closed(session)


@pytest.mark.parametrize("client", [Service, LegacyService])
@pytest.mark.parametrize("helper", ["first", "one"])
def test_cardinality_default_is_object_in_both_profiles(client, helper):
    session = ResultSession(payload({"name": "D"}))
    with client(ROOT, session=session) as service:
        assert isinstance(getattr(service.select("Department.name"), helper)(), ResultObject)
        assert requests(session)[-1]["format"] == ["jsonobjects"]
        assert_closed(session)


@pytest.mark.parametrize("mode", ["dict", "list", "rr", "json", "tsv", "csv"])
def test_first_flat_sends_size_one_and_closes_early(mode):
    session = ResultSession(b"A\nB\n" if mode in ("tsv", "csv") else payload(["A"], ["B"]))
    with LegacyService(ROOT, session=session) as service:
        query = service.select("Department.name")
        value = query.first(mode, start=3)
        if mode == "dict":
            assert value == {"Department.name": "A"}
        elif mode == "rr":
            assert value[0] == "A"
        else:
            assert value == ("A" if mode in ("tsv", "csv") else ["A"])
        assert requests(session)[0]["size"] == ["1"]
        assert requests(session)[0]["start"] == ["3"]
        assert len(session.lines) == (1 if mode in ("tsv", "csv") else 2)
        assert_closed(session)


@pytest.mark.parametrize("mode", ["jsonobjects", "dict", "tsv"])
def test_first_empty_returns_none_and_closes(mode):
    session = ResultSession(b"" if mode == "tsv" else payload())
    with LegacyService(ROOT, session=session) as service:
        assert service.select("Department.name").first(mode) is None
        assert_closed(session)


@pytest.mark.parametrize("helper", ["first", "one", "get_results_list"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_helpers_close_parser_failures_and_interrupts(helper, interrupt):
    session = ResultSession(payload(["invalid object"]), count=2, interrupt=interrupt)
    with LegacyService(ROOT, session=session) as service:
        with pytest.raises(KeyboardInterrupt if interrupt else TypeError):
            getattr(service.select("Department.name"), helper)()
        assert_closed(session)


@pytest.mark.parametrize("client", [Service, LegacyService])
@pytest.mark.parametrize("mode", ["jsonobjects", "object", "objects", "objectformat"])
@pytest.mark.parametrize("count", [1, 6])
def test_one_objects_uses_top_level_cardinality_not_joined_row_count(client, mode, count):
    session = ResultSession(payload({"name": "D", "employees": [{"name": "A"}, {"name": "B"}]}), count=count)
    with client(ROOT, session=session) as service:
        query = service.select("Department.name", "Department.employees.name")
        before = query.to_xml()
        obj = query.one(mode)
        assert obj.name == "D" and [employee.name for employee in obj.employees] == ["A", "B"]
        count_params, result_params = requests(session)
        assert count_params["format"] == ["count"]
        assert result_params["format"] == ["jsonobjects"] and "size" not in result_params
        assert query.to_xml() == before
        assert_closed(session)


@pytest.mark.parametrize("rows,count,message,lines", [
    (payload(), 0, "No results received", 2),
    (payload({"name": "A"}, {"name": "B"}, {"name": "C"}), 3, "More than one result received", 3),
])
def test_one_object_errors_scan_only_needed_objects_and_close(rows, count, message, lines):
    session = ResultSession(rows, count=count)
    with LegacyService(ROOT, session=session) as service:
        with pytest.raises(QueryError) as error:
            service.select("Department.name").one()
        assert error.value.message == message
        assert len(requests(session)) == 2 and len(session.lines) == lines
        assert_closed(session)


@pytest.mark.parametrize("count", [0, 2])
def test_one_flat_count_errors_do_not_request_rows(count):
    session = ResultSession(payload(["A"]), count=count)
    with Service(ROOT, session=session) as service:
        with pytest.raises(QueryError) as error:
            service.select("Department.name").one("dict")
        assert error.value.message == f"Result size is not one: got {count} results"
        assert [request["format"] for request in requests(session)] == [["count"]]
        assert_closed(session)


def test_one_flat_count_one_requests_bounded_first():
    session = ResultSession(payload(["A"], ["B"]))
    with Service(ROOT, session=session) as service:
        assert service.select("Department.name").one("dict") == {"Department.name": "A"}
        assert requests(session)[1]["size"] == ["1"]
        assert len(session.lines) == 2
        assert_closed(session)


@pytest.mark.parametrize("client,legacy", [(Service, False), (LegacyService, True)])
@pytest.mark.parametrize("version", [7, 8])
def test_eager_row_lists_profile_pagination_single_request(client, legacy, version):
    rows = payload(["A"], ["B"]) if version == 8 else payload([{"value": "A"}], [{"value": "B"}])
    session = ResultSession(rows, version=version)
    with client(ROOT, session=session) as service:
        query = service.select("Department.name")
        values = query.get_row_list(2, 3)
        assert len(values) == 2
        assert isinstance(values[0], ResultRow if legacy else dict)
        assert [value[0] if legacy else value["Department.name"] for value in values] == ["A", "B"]
        params, = requests(session)
        assert params["format"] == ["json" if version == 8 else "jsonrows"]
        assert params["start"] == ["2"] and params["size"] == ["3"]
        assert_closed(session)


@pytest.mark.parametrize("client,legacy", [(Service, False), (LegacyService, True)])
@pytest.mark.parametrize("helper", ["get_results_list", "all"])
def test_eager_results_default_empty_and_single_request(client, legacy, helper):
    session = ResultSession(payload({"name": "A"}) if legacy else payload(["A"]))
    with client(ROOT, session=session) as service:
        query = service.select("Department.name")
        values = getattr(query, helper)()
        assert isinstance(values, list) and len(values) == 1
        assert isinstance(values[0], ResultObject if legacy else dict)
        assert len(requests(session)) == 1
        session.rows = payload()
        assert getattr(query, helper)() == []
        assert len(requests(session)) == 2
        assert_closed(session)
    assert Query.all is Query.get_results_list


@pytest.mark.parametrize("helper", ["get_results_list", "all"])
def test_helper_forwarding_and_exact_custom_model_binding(helper):
    class CustomQuery(Query):
        def results(self, *args, marker=None, **kwargs):
            self.marker = marker
            return super().results(*args, **kwargs)

    model = Model(b'<model name="custom" package="custom"><class name="Special"><attribute name="name" type="String"/><attribute name="extra" type="String"/></class></model>')
    session = ResultSession(payload({"class": "Special", "objectId": 1, "name": "A"}))
    with Service(ROOT, session=session) as service:
        query = CustomQuery(model, service=service, root="Special").select("name")
        assert "Special" not in service.model.classes
        obj = query.first(start=5, marker="first")
        assert obj._cld is model.get_class("Special") and query.marker == "first"
        assert requests(session)[0]["start"] == ["5"]
        assert getattr(query, helper)("objects", 6, 7, marker="eager")[0].name == "A"
        assert query.marker == "eager"
        assert requests(session)[1]["start"] == ["6"] and requests(session)[1]["size"] == ["7"]
        assert query.one()._cld is model.get_class("Special")
        session.rows = payload({"extra": "Fetched"})
        assert obj.extra == "Fetched"
        assert ElementTree.fromstring(requests(session)[-1]["query"][0]).attrib["view"] == "Special.extra"
        assert_closed(session)


@pytest.mark.parametrize("helper", ["one", "get_results_list"])
def test_scan_and_eager_helpers_close_late_server_errors(helper):
    rows = b'{"results":[\n{"name":"A"}\n],"wasSuccessful":false,"error":"broken"}\n'
    session = ResultSession(rows, count=2)
    with LegacyService(ROOT, session=session) as service:
        from intermine314.service.errors import WebserviceError

        with pytest.raises(WebserviceError, match="broken"):
            getattr(service.select("Department.name"), helper)()
        assert_closed(session)


@pytest.mark.parametrize("helper", ["first", "one", "get_results_list"])
def test_custom_result_protocol_can_omit_close(helper):
    class CustomQuery(Query):
        def results(self, *args, **kwargs):
            return iter([{"name": "A"}])

        def count(self):
            return 2

    query = CustomQuery(Model(fixture_bytes("model.xml")), root="Department")
    value = getattr(query, helper)()
    assert value == ([{"name": "A"}] if helper == "get_results_list" else {"name": "A"})


@pytest.mark.parametrize("helper", ["first", "one", "get_results_list"])
def test_custom_stream_cleanup_preserves_primary_error(helper):
    class Stream:
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            raise ValueError("primary parser failure")

        def close(self):
            self.closed = True
            raise RuntimeError("cleanup failure")

    class CustomQuery(Query):
        def results(self, *args, **kwargs):
            return stream

        def count(self):
            return 2

    stream = Stream()
    query = CustomQuery(Model(fixture_bytes("model.xml")), root="Department")
    with pytest.raises(ValueError, match="primary parser failure"):
        getattr(query, helper)()
    assert stream.closed
