"""Summary and historical dataframe calls through real offline transport."""
from xml.etree import ElementTree

import polars as pl
import pytest

from intermine314.model import Model, ModelError
from intermine314.query import Query
from intermine314.query.parallel_offset import ParallelExecutionError
from intermine314.service import Service
from intermine314.service.errors import WebserviceError
from intermine314.webservice import Service as LegacyService
from tests.fixtures.compatibility import FixtureSession, fixture_bytes
from tests.test_query_eager_results import (
    ResultSession,
    assert_closed,
    payload,
    requests,
)

ROOT = "https://offline.example/service"


@pytest.mark.parametrize("client", [Service, LegacyService])
@pytest.mark.parametrize("version", [7, 8])
@pytest.mark.parametrize("mode", [None, "jsonobjects", "object", "dict", "rr", "csv", "dataframe", "unknown"])
def test_summary_forces_raw_jsonrows_before_object_augmentation(client, version, mode):
    session = FixtureSession.service(version=version, rows=fixture_bytes("summary-categorical.json"))
    with client(ROOT, session=session) as service:
        query = service.select("Department.name").where("company.name", "=", "Acme")
        before = query.to_xml(), query.model, query.to_spec().root_class
        stream = query.results(mode, 2, 0, "name")
        assert [row for row in stream] == [
            {"item": "Sales", "count": 6}, {"item": "Engineering", "count": 3}, {"item": None, "count": 1},
        ]
        params, = requests(session)
        assert params["format"] == ["jsonrows"] and params["summaryPath"] == ["Department.name"]
        assert params["start"] == ["2"] and params["size"] == ["0"]
        assert ElementTree.fromstring(params["query"][0]).attrib["view"] == "Department.name"
        assert (query.to_xml(), query.model, query.to_spec().root_class) == before
        assert_closed(session)


@pytest.mark.parametrize("client", [Service, LegacyService])
@pytest.mark.parametrize("path_form", ["relative", "full", "path"])
def test_summaries_numeric_first_row_float_and_category_counts(client, path_form):
    session = ResultSession(fixture_bytes("summary-numeric.json"))
    with client(ROOT, session=session) as service:
        query = service.select("Employee.name", "Employee.age")
        path = {"relative": "age", "full": "Employee.age", "path": query.model.make_path("Employee.age")}[path_form]
        assert query.summarise(path, row="objects", start=4, size=0) == {
            "average": 12.5, "stdev": 2.25, "max": 20.0, "min": 5.0,
        }
        assert len(session.lines) == 2  # The second stats row is deliberately not consumed.
        params, = requests(session)
        assert params["summaryPath"] == ["Employee.age"] and params["size"] == ["0"]
        assert params["start"] == ["4"]
        session.rows = fixture_bytes("summary-categorical.json")
        assert query.summarize("name", size=3) == {"Sales": 6, "Engineering": 3, None: 1}
        assert requests(session)[1]["size"] == ["3"]
        assert_closed(session)
    assert Query.summarize is Query.summarise


@pytest.mark.parametrize("type_name", sorted(Model.NUMERIC_TYPES))
def test_numeric_selection_uses_model_numeric_types_and_exact_custom_model(type_name):
    model = Model(f'<model name="custom" package="custom"><class name="Special"><attribute name="metric" type="{type_name}"/></class></model>'.encode())
    session = ResultSession(fixture_bytes("summary-numeric.json"))
    with Service(ROOT, session=session) as service:
        query = Query(model, service=service, root="Special").select("metric")
        assert "Special" not in service.model.classes
        assert query.summarise("metric")["average"] == 12.5
        assert requests(session)[0]["summaryPath"] == ["Special.metric"]
        assert query.model is model
        assert_closed(session)


@pytest.mark.parametrize("path,expected", [("age", StopIteration), ("name", {})])
def test_empty_summary_source_behavior_and_closure(path, expected):
    session = ResultSession(payload())
    with LegacyService(ROOT, session=session) as service:
        query = service.select("Employee.name")
        if isinstance(expected, type):
            with pytest.raises(expected):
                query.summarise(path)
        else:
            assert query.summarise(path) == expected
        assert_closed(session)


@pytest.mark.parametrize("path,rows,error", [
    ("age", payload({"average": "invalid"}), ValueError),
    ("age", payload({"average": None}), TypeError),
    ("name", payload({"item": "Sales"}), KeyError),
    ("name", b'{"results":[\n{"item":"Sales","count":1}\n],"wasSuccessful":false,"error":"broken"}\n', WebserviceError),
    ("age", b'{"results":[\ninvalid\n', WebserviceError),
])
def test_summary_errors_close_managed_stream(path, rows, error):
    session = ResultSession(rows)
    with Service(ROOT, session=session) as service:
        with pytest.raises(error):
            service.select("Employee.name").summarise(path)
        assert_closed(session)


@pytest.mark.parametrize("path", ["age", "name"])
def test_summary_interrupt_closes_actual_response(path):
    session = ResultSession(fixture_bytes("summary-categorical.json"), interrupt=True)
    with LegacyService(ROOT, session=session) as service:
        with pytest.raises(KeyboardInterrupt):
            service.select("Employee.name").summarise(path)
        assert_closed(session)


def test_invalid_model_summary_path_fails_before_result_request():
    session = ResultSession(payload())
    with Service(ROOT, session=session) as service:
        with pytest.raises(ModelError):
            service.select("Employee.name").summarise("missing")
        assert requests(session) == []
        assert_closed(session)


@pytest.mark.parametrize("type_name", ["byte", "Byte", "java.lang.Byte", "BigDecimal", "java.math.BigDecimal"])
@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_server_numeric_byte_and_bigdecimal_stats_are_supported(type_name, profile):
    model = Model(f'<model name="custom" package="custom"><class name="Special"><attribute name="metric" type="{type_name}"/></class></model>'.encode())
    session = ResultSession(fixture_bytes("summary-numeric.json"))
    with Service(ROOT, session=session) as service:
        query = Query(model, service=service, root="Special", compatibility=profile).select("metric")
        assert query.summarise("metric") == {"average": 12.5, "stdev": 2.25, "max": 20.0, "min": 5.0}
        assert_closed(session)


def test_big_integer_keeps_categorical_type_classification():
    model = Model(b'<model name="custom" package="custom"><class name="Special"><attribute name="metric" type="java.math.BigInteger"/></class></model>')
    session = ResultSession(payload({"item": "12345678901234567890", "count": 2}))
    with Service(ROOT, session=session) as service:
        query = Query(model, service=service, root="Special").select("metric")
        assert query.model.make_path("Special.metric").end.type_name not in Model.NUMERIC_TYPES
        assert query.summarise("metric") == {"12345678901234567890": 2}
        assert_closed(session)


@pytest.mark.parametrize("client", [Service, LegacyService])
def test_summary_nested_paths_and_subclass_fields(client):
    session = ResultSession(fixture_bytes("summary-numeric.json"))
    with client(ROOT, session=session) as service:
        query = service.select("Department.name")
        assert query.summarise("manager.seniority")["average"] == 12.5
        assert requests(session)[0]["summaryPath"] == ["Department.manager.seniority"]
        session.rows = fixture_bytes("summary-categorical.json")
        query = service.select("Employee.name")
        query.add_constraint(path="Employee", subclass="Manager")
        assert query.summarise("title")["Sales"] == 6
        assert requests(session)[1]["summaryPath"] == ["Employee.title"]
        assert_closed(session)


@pytest.mark.parametrize("consumed", [False, True])
def test_raw_summary_stream_explicit_close_before_or_after_read(consumed):
    session = ResultSession(fixture_bytes("summary-categorical.json"))
    with LegacyService(ROOT, session=session) as service:
        stream = iter(service.select("Department.name").results(summary_path="name"))
        if consumed:
            assert next(stream) == {"item": "Sales", "count": 6}
        stream.close()
        assert_closed(session)


@pytest.mark.parametrize("delegated", [False, True])
def test_no_model_raw_summary_and_fallback_execution(delegated):
    class CustomQuery(Query):
        def _to_execution(self):
            return super()._to_execution() if delegated else None

    session = ResultSession(fixture_bytes("summary-categorical.json"))
    with Service(ROOT, session=session) as service:
        query = CustomQuery(service=service, root="Unknown").select("name")
        assert [row for row in query.results("objects", summary_path="name")][0] == {"item": "Sales", "count": 6}
        assert requests(session)[0]["summaryPath"] == ["Unknown.name"]
        assert_closed(session)


@pytest.mark.parametrize("path", ["age", "name"])
@pytest.mark.parametrize("failure", [None, ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("close_mode", ["absent", "normal", "raises"])
def test_summary_custom_iterator_options_and_cleanup(path, failure, close_mode):
    closed = []

    class Stream:
        def __iter__(self):
            return self

        def __next__(self):
            if failure:
                raise failure("primary")
            if getattr(self, "done", False):
                raise StopIteration
            self.done = True
            return {"average": "1.5"} if path == "age" else {"item": "A", "count": 2}

    if close_mode != "absent":
        def close(self):
            closed.append(True)
            if close_mode == "raises":
                raise OSError("cleanup")
        Stream.close = close

    class CustomQuery(Query):
        def results(self, **options):
            assert options == {"summary_path": path, "start": 2, "size": 0, "marker": "custom"}
            return Stream()

    query = CustomQuery(Model(fixture_bytes("model.xml")), root="Employee")
    if failure:
        with pytest.raises(failure, match="primary"):
            query.summarise(path, start=2, size=0, marker="custom")
    else:
        assert query.summarise(path, start=2, size=0, marker="custom") == ({"average": 1.5} if path == "age" else {"A": 2})
    assert closed == ([] if close_mode == "absent" else [True])


@pytest.mark.parametrize("client", [Service, LegacyService])
@pytest.mark.parametrize("version", [7, 8])
def test_dataframe_row_alias_is_dictionary_iterator(client, version):
    rows = payload(["A", 42]) if version == 8 else payload([{"value": "A"}, {"value": 42}])
    session = ResultSession(rows, version=version)
    with client(ROOT, session=session) as service:
        query = service.select("Employee.name", "Employee.age")
        stream = query.results("dataframe", 3, 0)
        assert not isinstance(stream, pl.DataFrame)
        assert [row for row in stream] == [{"Employee.name": "A", "Employee.age": 42}]
        params, = requests(session)
        assert params["start"] == ["3"] and params["size"] == ["0"]
        assert params["format"] == ["json" if version == 8 else "jsonrows"]
        assert_closed(session)


@pytest.mark.parametrize("client", [Service, LegacyService])
@pytest.mark.parametrize("persistent", [False, True])
def test_dataframe_actual_remote_polars_pagination_persistence(tmp_path, monkeypatch, client, persistent):
    from intermine314.export import query as export_query

    session = ResultSession(payload(["A", 42], ["B", 43]))
    paths = []
    original = export_query.query_parquet

    def read(path):
        paths.append(path)
        return original(path)

    monkeypatch.setattr(export_query, "query_parquet", read)
    target = tmp_path / "saved.parquet" if persistent else None
    with client(ROOT, session=session) as service:
        query = service.select("Employee.name", "Employee.age")
        before = query.to_xml()
        frame = query.dataframe(3, 2, parquet_path=target)
        assert isinstance(frame, pl.DataFrame)
        assert frame.to_dicts() == [{"Employee.name": "A", "Employee.age": 42}, {"Employee.name": "B", "Employee.age": 43}]
        assert frame.schema == {"Employee.name": pl.String, "Employee.age": pl.Int32}
        params, = requests(session)
        assert params["format"] == ["json"] and params["start"] == ["3"] and params["size"] == ["2"]
        assert paths[0].exists() is persistent
        assert query.to_xml() == before
        assert_closed(session)
    assert frame.rows() == [("A", 42), ("B", 43)]
    assert not list(tmp_path.rglob("*.csv"))


@pytest.mark.parametrize("client", [Service, LegacyService])
def test_dataframe_actual_empty_page_preserves_full_path_schema(client):
    session = ResultSession(payload())
    with client(ROOT, session=session) as service:
        frame = service.select("Employee.name", "Employee.age").dataframe(2, 0)
        assert frame.height == 0 and frame.schema == {"Employee.name": pl.String, "Employee.age": pl.Int32}
        assert requests(session) == []  # The native export scheduler skips size=0 HTTP.
        assert_closed(session)


@pytest.mark.parametrize("interrupt", [False, True])
def test_dataframe_actual_remote_failure_closes_response_and_managed_storage(monkeypatch, interrupt):
    rows = b'{"results":[\ninvalid\n],"wasSuccessful":false,"error":"broken"}\n'
    session = ResultSession(rows, interrupt=interrupt)
    paths = []
    with Service(ROOT, session=session) as service:
        query = service.select("Employee.name", "Employee.age")
        original = query.to_parquet

        def write(path, **options):
            paths.append(path)
            return original(path, **options)

        monkeypatch.setattr(query, "to_parquet", write)
        with pytest.raises(KeyboardInterrupt if interrupt else ParallelExecutionError) as error:
            query.dataframe()
        if not interrupt:
            assert isinstance(error.value.__cause__, WebserviceError)
        assert not paths[0].parent.exists()
        assert_closed(session)
