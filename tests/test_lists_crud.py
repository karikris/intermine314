from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections.abc import KeysView, ValuesView
from functools import partial
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from intermine314.service.errors import WebserviceError
from intermine314.service.service import Service
from intermine314.webservice import Service as LegacyService
from tests.fixtures.compatibility import FixtureSession, fixture_bytes


def list_module():
    return importlib.import_module("intermine314.lists.list")


def manager_module():
    return importlib.import_module("intermine314.lists.listmanager")


def client(profile="native", *, size=2, name="identifiers", rows=None):
    session = FixtureSession.service(rows=rows)
    info = {"name": name, "title": "Fixed title", "type": "Employee", "size": str(size),
            "description": "Unicode α", "tags": ["tag α"], "status": "CURRENT"}
    session.routes[("GET", "/service/lists")] = json.dumps({
        "wasSuccessful": True, "lists": [info],
    }).encode()
    session.routes[("POST", "/service/lists")] = fixture_bytes("list-created.json")
    session.routes[("DELETE", "/service/lists")] = b'{"wasSuccessful":true}'
    service = (Service if profile == "native" else LegacyService)(
        "https://offline.example/service", session=session, token="secret",
        request_timeout=37, user_agent="list-client", verify_tls="/custom/ca.pem",
    )
    return service, session


def params(request):
    return parse_qs(urlsplit(request.url).query, keep_blank_values=True)


def assert_closed(session):
    assert all(response.closed and response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 0


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_discovery_metadata_views_cache_and_independent_managers(profile):
    service, session = client(profile)
    assert service._list_manager is None
    first, second = service.list_manager(), service.list_manager()
    assert first is not second and first.lists is None and second.lists is None
    assert first._temp_lists is not second._temp_lists
    item = first.get_list("identifiers")
    assert isinstance(item, list_module().List)
    assert first.l("identifiers") is item
    assert first.get_list("missing") is None
    assert isinstance(first.get_all_lists(), ValuesView)
    assert isinstance(first.get_all_list_names(), KeysView)
    assert first.get_list_count() == 1
    assert item.name == "identifiers" and item.get_name() == item.name
    assert item.title == "Fixed title" and item.list_type == "Employee"
    assert item.size == item.count == len(item) == 2
    assert item.tags == frozenset(["tag α"])
    assert item.is_authorized is True and item.status == "CURRENT"
    assert item.date_created is None and item.description == "Unicode α"
    assert str(item) == "identifiers (2 Employee) Unicode α"
    assert item.unmatched_identifiers == set()
    assert sum(r.path == "/service/lists" for r in session.requests) == 1
    assert service._list_manager is None
    assert_closed(session)


def test_pinned_metadata_and_safe_dict():
    service, session = client()
    session.routes[("GET", "/service/lists")] = fixture_bytes("lists.json")
    manager = service.list_manager()
    item = manager.get_list("test-list-1")
    assert item.date_created == "2011-05-07T19:52:03"
    assert str(item) == "test-list-1 (42 Employee) 2011-05-07T19:52:03 An example test list"
    assert manager.get_list("test-list-2").is_authorized is False
    assert manager.get_list("test-list-3").tags == frozenset()
    nested = {"inner": []}
    value = {"κ": nested}
    cloned = manager.safe_dict(value)
    assert cloned == value and cloned is not value and cloned["κ"] is nested
    assert manager.safe_dict(nested["inner"]) is nested["inner"]
    assert manager_module().safe_key("κ") == "κ"
    with pytest.raises(ValueError, match="Missing argument"):
        list_module().List(service=service, manager=manager)
    with pytest.raises(AttributeError, match="only changed"):
        del item.name
    for field in ("size", "count", "title", "description", "list_type", "tags", "date_created", "status", "is_authorized"):
        with pytest.raises(AttributeError):
            setattr(item, field, "invalid")


@pytest.mark.parametrize("kind", ["text", "path", "path_string", "reader", "binary", "iterable", "generator"])
@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_identifier_upload_actual_wire_and_borrowed_lifetime(tmp_path, kind, profile):
    service, session = client(profile)
    manager = service.list_manager()
    raw = "0007\nκ\nname with spaces"
    path = tmp_path / "identifiers.txt"
    path.write_text(raw, encoding="utf-8")
    source = {"text": "  " + raw + "  ", "path": path, "path_string": str(path),
              "reader": StringIO(raw), "binary": BytesIO(raw.encode()),
              "iterable": ["0007", "κ", "name with spaces"],
              "generator": (value for value in ["0007", "κ", "name with spaces"])}[kind]
    item = manager.create_list(source, "Employee", name="identifiers", tags=["α", "β"], add=["DUPLICATE", "", "WILDCARD"])
    assert item is manager.get_list("identifiers")
    assert item.unmatched_identifiers == {"unmatched α", "0000"}
    upload = next(r for r in session.requests if r.method == "POST" and r.path == "/service/lists")
    expected = '"0007"\n"κ"\n"name with spaces"' if kind in ("iterable", "generator") else raw
    assert upload.data == expected.encode()
    assert params(upload) == {"name": ["identifiers"], "type": ["Employee"], "description": [manager.DEFAULT_DESCRIPTION],
                             "tags": ["α;β"], "add": ["duplicate", "wildcard"], "token": ["secret"]}
    assert upload.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert upload.headers["User-Agent"] == "list-client"
    assert upload.options["verify"] == "/custom/ca.pem"
    assert upload.options["timeout"][1] == 37
    if hasattr(source, "closed"):
        assert not source.closed
    assert_closed(session)


@pytest.mark.parametrize("source", ["", [], (), iter(()), StringIO("")])
def test_empty_upload_prints_none_without_http(source, capsys):
    service, session = client()
    result = service.list_manager().create_list(source, "Employee", name="identifiers")
    assert result is None
    assert capsys.readouterr().out == (
        "Lists must have one or more elements - the current list has 0\n"
        "Please create a valid list with at least one element and create the list again.\n"
    )
    assert [r.path for r in session.requests] == ["/service/version/ws"]


def test_name_allocation_rename_cache_and_temporary_retirement():
    service, session = client(name="my_list_1")
    manager = service.list_manager()
    assert manager.get_unused_list_name() == "my_list_2"
    assert manager.get_unused_list_name() == "my_list_3"
    assert manager._temp_lists == {"my_list_2", "my_list_3"}
    item = manager.get_list("my_list_1")
    manager._temp_lists.add(item.name)
    item.name = item.name
    assert all(r.path != "/service/lists/rename" for r in session.requests)
    session.routes[("GET", "/service/lists/rename")] = fixture_bytes("list-renamed.json")
    session.routes[("GET", "/service/lists")] = fixture_bytes("lists-renamed.json")
    item.name = "retained α"
    rename = next(r for r in session.requests if r.path == "/service/lists/rename")
    assert params(rename)["oldname"] == ["my_list_1"] and params(rename)["newname"] == ["retained α"]
    assert item.name == "retained α"
    assert manager.get_list("retained α") is item
    assert manager.get_list("my_list_1") is None
    assert "my_list_1" not in manager._temp_lists and "retained α" not in manager._temp_lists
    assert_closed(session)


def test_delete_skips_missing_accepts_names_and_lists_and_refreshes():
    service, session = client()
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    item.delete()
    manager.delete_lists(["missing", "identifiers"])
    deletes = [r for r in session.requests if r.method == "DELETE"]
    assert len(deletes) == 2 and all(params(r)["name"] == ["identifiers"] for r in deletes)
    assert sum(r.method == "GET" and r.path == "/service/lists" for r in session.requests) == 5
    assert_closed(session)


@pytest.mark.parametrize("action", ["refresh", "upload", "delete", "rename"])
@pytest.mark.parametrize("response", [b'{"wasSuccessful":false,"error":"denied"}', b"invalid json"])
def test_list_protocol_errors_are_public_and_close_responses(action, response):
    service, session = client()
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    error = manager_module().ListServiceError
    assert issubclass(error, WebserviceError)
    if action == "refresh":
        session.routes[("GET", "/service/lists")] = response
        invoke = manager.refresh_lists
    elif action == "upload":
        session.routes[("POST", "/service/lists")] = response
        invoke = partial(manager.create_list, ["κ"], "Employee", name="identifiers")
    elif action == "delete":
        session.routes[("DELETE", "/service/lists")] = response
        invoke = partial(manager.delete_lists, [item])
    else:
        session.routes[("GET", "/service/lists/rename")] = response
        invoke = partial(item.set_name, "renamed")
    with pytest.raises(error, match="denied|Error parsing response"):
        invoke()
    assert item.name == "identifiers"
    assert_closed(session)


def test_http_failure_is_not_retried_as_another_content_type():
    service, session = client()
    session.routes[("POST", "/service/lists")] = (500, b"failed")
    source = StringIO("κ")
    with pytest.raises(WebserviceError):
        service.list_manager().create_list(source, "Employee", name="identifiers")
    assert not source.closed
    assert sum(r.method == "POST" for r in session.requests) == 1
    assert_closed(session)


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_real_list_iteration_index_display_named_constraint_and_stream_closure(profile, capsys):
    service, session = client(profile, rows=fixture_bytes("list-employees-objects.json") if profile == "legacy" else fixture_bytes("list-employees-rows.json"))
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    values = [value for value in item]
    assert len(values) == 2
    assert (values[0]["Employee.name"] if profile == "native" else values[0].name) == "Comma , here"
    item.display()
    output = capsys.readouterr().out
    assert "Row 1:" in output and "Row 2:" in output
    assert "name = Comma , here" in output and "name = κ (brackets)" in output
    session.routes[("POST", "/service/query/results")] = fixture_bytes("list-employees-objects.json")
    assert item[0].name == "Comma , here"
    assert item[-1].name == "Comma , here"  # fixture server returns its fixed page; check offset below
    queries = [parse_qs(r.data.decode())["query"][0] for r in session.requests if r.path == "/service/query/results"]
    assert all('path="Employee"' in query and 'op="IN"' in query and 'value="identifiers"' in query for query in queries)
    assert parse_qs(session.requests[-1].data.decode())["start"] == ["1"]
    before = len(session.requests)
    for index in (2, -3, "1", slice(None)):
        with pytest.raises(IndexError):
            item[index]
    assert len(session.requests) == before
    assert item.to_query().to_xml() == item._contents_query().to_xml()
    assert_closed(session)


@pytest.mark.parametrize("source_kind", ["path", "text", "binary", "nonseekable"])
def test_csv_strings_wire_null_policy_and_borrowed_lifetime(tmp_path, source_kind):
    service, session = client()
    text = 'identifier,other\n0007,9\nκ,10\n"name, with spaces",11\n"a ""quote""",12\n'
    path = tmp_path / "identifiers.txt"
    path.write_text(text, encoding="utf-8")
    class ReadOnly:
        closed = False
        def read(self):
            return text
    source = {"path": path, "text": StringIO(text), "binary": BytesIO(text.encode()), "nonseekable": ReadOnly()}[source_kind]
    item = service.list_manager().create_list(csv_input=source, csv_column="identifier", list_type="Employee", name="identifiers")
    assert item.name == "identifiers"
    upload = next(r for r in session.requests if r.method == "POST")
    assert upload.data == '"0007"\n"κ"\n"name, with spaces"\n"a ""quote"""'.encode()
    if hasattr(source, "closed"):
        assert not source.closed
    assert_closed(session)


@pytest.mark.parametrize("options", [
    {"csv_input": StringIO("id\n0007\n"), "list_type": "Employee"},
    {"csv_input": StringIO("id\n0007\n"), "csv_column": "id"},
    {"content": ["0007"], "csv_input": StringIO("id\n0007\n"), "csv_column": "id", "list_type": "Employee"},
    {"csv_column": "id"}, {"csv_options": {}},
    {"csv_input": StringIO("id\n0007\n"), "csv_column": "id", "list_type": "Employee", "organism": "human"},
])
def test_csv_invalid_combinations_fail_before_requests(options):
    service, session = client()
    with pytest.raises((TypeError, ValueError)):
        service.list_manager().create_list(name="identifiers", **options)
    assert len(session.requests) == 1
    if "csv_input" in options:
        assert not options["csv_input"].closed


@pytest.mark.parametrize("text,match", [("id\n0007\n\n", "null"), ('id\n"line\nbreak"\n', "line breaks"), ("other\n0007\n", "column")])
def test_csv_invalid_identifiers_leave_borrowed_open_and_no_upload(text, match):
    service, session = client()
    source = StringIO(text)
    with pytest.raises(ValueError, match=match):
        service.list_manager().create_list(csv_input=source, csv_column="id", list_type="Employee", name="identifiers")
    assert not source.closed and len(session.requests) == 1


def test_ordinary_list_facades_and_usage_do_not_import_analytics():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", "\n".join([
        "import json, sys",
        "from intermine314.lists import List, ListManager, ListServiceError",
        "from intermine314.lists.listmanager import safe_key",
        "from intermine314.service.service import Service",
        "from tests.fixtures.compatibility import FixtureSession, fixture_bytes",
        "session=FixtureSession.service()",
        "session.routes['GET','/service/lists']=fixture_bytes('lists.json')",
        "service=Service('https://offline.example/service', session=session)",
        "manager=service.list_manager(); manager.get_all_lists()",
        "print(json.dumps([m for m in sys.modules if m.split('.')[0] in ('pandas','polars','duckdb','pyarrow')]))",
    ])], text=True, capture_output=True, env=dict(os.environ, PYTHONPATH=str(root / "src") + os.pathsep + str(root)))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize("content,expected", [(['a "quote"', " space, κ \t"], '"a ""quote"""\n" space, κ \t"'),
                                               (['line\nbreak'], None), (['line\rbreak'], None)])
def test_iterable_token_escaping_and_line_break_rejection(content, expected):
    service, session = client()
    manager = service.list_manager()
    if expected is None:
        with pytest.raises(ValueError, match="line breaks"):
            manager.create_list(content, "Employee", name="identifiers")
        assert len(session.requests) == 1
    else:
        manager.create_list(content, "Employee", name="identifiers")
        assert next(r.data for r in session.requests if r.method == "POST") == expected.encode()


@pytest.mark.parametrize("schema_option", ["schema", "schema_overrides"])
def test_csv_forces_identifier_string_despite_numeric_schema(schema_option):
    import polars as pl

    service, session = client()
    source = StringIO("identifier,other\n0007,9\n0010,10\n")
    options = {schema_option: {"identifier": pl.Int64, "other": pl.Int64}}
    service.list_manager().create_list(csv_input=source, csv_column="identifier", csv_options=options,
                                       list_type="Employee", name="identifiers")
    assert next(r.data for r in session.requests if r.method == "POST") == b'"0007"\n"0010"'
    assert options[schema_option]["identifier"] == pl.Int64
    assert not source.closed


@pytest.mark.parametrize("failure", [None, "null", "reader", "sql", "parquet", "http", "parser"])
def test_csv_temporary_parquet_duckdb_arrow_cleanup_on_all_paths(tmp_path, monkeypatch, failure):
    import duckdb
    import polars as pl

    csv_module = importlib.import_module("intermine314.lists._csv")
    service, session = client()
    source = StringIO("id\n0007\n" if failure != "null" else "id\n0007\n\n")
    if failure == "http":
        session.routes[("POST", "/service/lists")] = (500, b"failed")
    if failure == "parser":
        session.routes[("POST", "/service/lists")] = b"invalid json"
    owned_directories, connections, readers, parquet_writes = [], [], [], []
    original_temporary = csv_module.TemporaryDirectory
    original_connect = duckdb.connect
    original_sink = pl.LazyFrame.sink_parquet

    def temporary(**kwargs):
        directory = original_temporary(dir=tmp_path, **kwargs)
        owned_directories.append(Path(directory.name))
        return directory

    class Reader:
        def __init__(self, wrapped):
            self.wrapped, self.closed = wrapped, False
        def __iter__(self):
            if failure == "reader":
                raise RuntimeError("reader failed")
            return iter(self.wrapped)
        def close(self):
            self.closed = True
            self.wrapped.close()

    class Connection:
        def __init__(self):
            self.wrapped, self.closed = original_connect(":memory:"), False
        def execute(self, *args):
            if failure == "sql":
                raise RuntimeError("SQL failed")
            self.wrapped.execute(*args)
            return self
        def to_arrow_reader(self, batch_size):
            reader = Reader(self.wrapped.to_arrow_reader(batch_size))
            readers.append(reader)
            return reader
        def close(self):
            self.closed = True
            self.wrapped.close()

    def connect(*args):
        connection = Connection()
        connections.append(connection)
        return connection

    def sink(frame, path, **kwargs):
        parquet_writes.append(Path(path))
        if failure == "parquet":
            raise RuntimeError("Parquet failed")
        return original_sink(frame, path, **kwargs)

    monkeypatch.setattr(csv_module, "TemporaryDirectory", temporary)
    monkeypatch.setattr(duckdb, "connect", connect)
    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", sink)
    manager = service.list_manager()
    invoke = partial(manager.create_list, csv_input=source, csv_column="id", list_type="Employee", name="identifiers")
    if failure is None:
        assert invoke().name == "identifiers"
    else:
        with pytest.raises((ValueError, RuntimeError, WebserviceError)):
            invoke()
    assert not source.closed
    assert owned_directories and all(not directory.exists() for directory in owned_directories)
    assert parquet_writes and all(path.suffix == ".parquet" for path in parquet_writes)
    assert all(connection.closed for connection in connections)
    assert all(reader.closed for reader in readers)
    assert list(tmp_path.iterdir()) == []
    assert_closed(session)


def test_display_closes_real_stream_when_printing_raises(monkeypatch):
    service, session = client(rows=fixture_bytes("list-employees-rows.json"))
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("interrupted output")
    monkeypatch.setattr("builtins.print", interrupted)
    with pytest.raises(KeyboardInterrupt):
        item.display()
    assert_closed(session)


def test_csv_empty_string_identifier_rejected_but_zero_rows_print_none(capsys):
    service, session = client()
    manager = service.list_manager()
    source = StringIO('id\n""\n')
    with pytest.raises(ValueError, match="empty"):
        manager.create_list(csv_input=source, csv_column="id", list_type="Employee", name="identifiers")
    assert not source.closed and len(session.requests) == 1
    zero = StringIO("id\n")
    assert manager.create_list(csv_input=zero, csv_column="id", list_type="Employee", name="identifiers") is None
    assert "Lists must have one or more elements" in capsys.readouterr().out
    assert not zero.closed and len(session.requests) == 1


def test_csv_scratch_path_is_literal_with_glob_characters(tmp_path, monkeypatch):
    csv_module = importlib.import_module("intermine314.lists._csv")
    original = csv_module.TemporaryDirectory
    scratch_root = tmp_path / "scratch[1]"
    scratch_root.mkdir()
    def temporary(**kwargs):
        return original(dir=scratch_root, **kwargs)
    monkeypatch.setattr(csv_module, "TemporaryDirectory", temporary)
    service, session = client()
    source = StringIO("id\n0007\n")
    item = service.list_manager().create_list(csv_input=source, csv_column="id", list_type="Employee", name="identifiers")
    assert item.name == "identifiers"
    assert next(r.data for r in session.requests if r.method == "POST") == b'"0007"'
    assert list(scratch_root.iterdir()) == [] and not source.closed


def test_delete_lists_can_receive_its_own_temporary_name_set():
    service, session = client()
    session.routes[("GET", "/service/lists")] = fixture_bytes("lists.json")
    manager = service.list_manager()
    manager._temp_lists.update(["test-list-1", "test-list-2"])
    manager.delete_lists(manager._temp_lists)
    assert manager._temp_lists == set()
    assert {params(r)["name"][0] for r in session.requests if r.method == "DELETE"} == {"test-list-1", "test-list-2"}
    assert_closed(session)


@pytest.mark.parametrize("name", [None, "identifiers"])
def test_create_and_delete_publish_server_cache_updates(name, monkeypatch):
    service, session = client()
    manager = service.list_manager()
    session.routes[("GET", "/service/lists")] = b'{"wasSuccessful":true,"lists":[]}'
    expected_name = "my_list_1" if name is None else name
    session.routes[("POST", "/service/lists")] = json.dumps({"wasSuccessful": True, "listName": expected_name}).encode()
    metadata = json.loads(fixture_bytes("lists-renamed.json"))["lists"][0]
    metadata["name"] = expected_name
    original_request = session.request
    def request(method, url, **kwargs):
        response = original_request(method, url, **kwargs)
        if urlsplit(url).path == "/service/lists":
            if method == "POST":
                assert params(session.requests[-1])["name"] == [expected_name]
                session.routes[("GET", "/service/lists")] = json.dumps({"wasSuccessful": True, "lists": [metadata]}).encode()
            elif method == "DELETE":
                session.routes[("GET", "/service/lists")] = b'{"wasSuccessful":true,"lists":[]}'
        return response
    monkeypatch.setattr(session, "request", request)
    assert manager.get_list_count() == 0
    item = manager.create_list(["0007", "κ"], "Employee", name=name, description="explicit description")
    assert manager.get_list_count() == 1 and manager.get_list(expected_name) is item
    assert manager._temp_lists == ({expected_name} if name is None else set())
    assert params(next(r for r in session.requests if r.method == "POST"))["description"] == ["explicit description"]
    item.delete()
    assert manager.get_list_count() == 0 and manager.get_list(expected_name) is None
    assert manager._temp_lists == set()
    assert_closed(session)
