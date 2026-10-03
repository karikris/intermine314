from __future__ import annotations

import importlib
import json
from functools import partial
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from intermine314.service.errors import WebserviceError
from intermine314.service.session import _ResponseStreamAdapter
from tests.fixtures.compatibility import fixture_bytes
from tests.test_lists_crud import assert_closed, client, manager_module, params


def append_route(session):
    session.routes[("POST", "/service/lists/append")] = fixture_bytes("list-created.json")


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("kind", ["text", "path", "path_string", "reader", "binary", "iterable", "generator"])
def test_append_identifiers_keyword_wire_and_same_object(tmp_path, profile, kind):
    service, session = client(profile)
    append_route(session)
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    item.unmatched_identifiers.add("previous")
    raw = '0007\nκ\na "quote"'
    path = tmp_path / "ids.txt"
    path.write_text(raw, encoding="utf-8")
    source = {"text": "  " + raw + "  ", "path": path, "path_string": str(path),
              "reader": StringIO(raw), "binary": BytesIO(raw.encode()),
              "iterable": ["0007", "κ", 'a "quote"'],
              "generator": (value for value in ["0007", "κ", 'a "quote"'])}[kind]
    metadata = json.loads(session.routes[("GET", "/service/lists")])
    metadata["lists"][0]["size"] = "5"
    session.routes[("GET", "/service/lists")] = json.dumps(metadata).encode()
    assert item.append(appendix=source) is item
    assert item.size == 5 and item.unmatched_identifiers == {"previous", "unmatched α", "0000"}
    upload = next(r for r in session.requests if r.method == "POST")
    assert upload.path == "/service/lists/append"
    assert params(upload) == {"name": ["identifiers"], "token": ["secret"]}
    expected = ('"0007"\n"κ"\n"a ""quote"""' if kind in ("iterable", "generator")
                else "  " + raw + "  " if kind == "text" else raw)
    assert upload.data == expected.encode()
    assert upload.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert upload.headers["User-Agent"] == "list-client"
    assert upload.options["verify"] == "/custom/ca.pem" and upload.options["timeout"][1] == 37
    if hasattr(source, "closed"):
        assert not source.closed
    assert_closed(session)


@pytest.mark.parametrize("failure", ["http", "parser", "unsuccessful", "interrupt"])
def test_append_failure_never_retries_or_updates_metadata(monkeypatch, failure):
    service, session = client()
    append_route(session)
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    if failure == "http":
        session.routes[("POST", "/service/lists/append")] = (500, b"failed")
    elif failure == "parser":
        session.routes[("POST", "/service/lists/append")] = b"bad json"
    elif failure == "unsuccessful":
        session.routes[("POST", "/service/lists/append")] = b'{"wasSuccessful":false,"error":"denied"}'
    else:
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt("read interrupted")
        monkeypatch.setattr(_ResponseStreamAdapter, "read", interrupted)
    source = StringIO("0007")
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else WebserviceError):
        item.append(source)
    assert sum(r.method == "POST" for r in session.requests) == 1
    assert not source.closed and item.size == 2 and item.unmatched_identifiers == set()
    assert_closed(session)


@pytest.mark.parametrize("source", ["", [], (), iter(()), StringIO("")])
def test_append_empty_input_preserves_original_post_and_return(source, capsys):
    service, session = client()
    append_route(session)
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    assert item.append(source) is item
    assert next(r for r in session.requests if r.method == "POST").data == b""
    assert capsys.readouterr().out == ""
    assert_closed(session)


@pytest.mark.parametrize("source", [["line\nbreak"], ["line\rbreak"]])
def test_append_iterable_line_breaks_reject_before_upload(source):
    service, session = client()
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    before = len(session.requests)
    with pytest.raises(ValueError, match="line breaks"):
        item.append(source)
    assert len(session.requests) == before


@pytest.mark.parametrize("source_kind", ["path", "text", "binary", "nonseekable"])
def test_append_csv_strings_quotes_and_borrowed_streams(tmp_path, source_kind):
    service, session = client()
    append_route(session)
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    text = 'id,other\n0007,9\nκ,10\n"name, with spaces",11\n"a ""quote""",12\n'
    path = tmp_path / "ids.txt"
    path.write_text(text, encoding="utf-8")
    class ReadOnly:
        closed = False
        def read(self):
            return text
    source = {"path": path, "text": StringIO(text), "binary": BytesIO(text.encode()), "nonseekable": ReadOnly()}[source_kind]
    assert item.append(csv_input=source, csv_column="id") is item
    upload = next(r for r in session.requests if r.method == "POST")
    assert upload.path == "/service/lists/append"
    assert upload.data == '"0007"\n"κ"\n"name, with spaces"\n"a ""quote"""'.encode()
    assert params(upload)["name"] == [item.name] and "type" not in params(upload)
    if hasattr(source, "closed"):
        assert not source.closed
    assert_closed(session)


def test_append_csv_header_only_preserves_empty_append_semantics(capsys):
    service, session = client()
    append_route(session)
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    source = StringIO("id\n")
    assert item.append(csv_input=source, csv_column="id") is item
    assert next(r for r in session.requests if r.method == "POST").data == b""
    assert capsys.readouterr().out == "" and not source.closed
    assert_closed(session)


@pytest.mark.parametrize("kwargs", [{"csv_column": "id"}, {"csv_options": {}},
    {"csv_input": StringIO("id\n0007\n")},
    {"appendix": ["0007"], "csv_input": StringIO("id\n0007\n"), "csv_column": "id"}])
def test_append_csv_conflicts_fail_before_http(kwargs):
    service, session = client()
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    before = len(session.requests)
    with pytest.raises(ValueError):
        item.append(**kwargs)
    assert len(session.requests) == before
    if "csv_input" in kwargs:
        assert not kwargs["csv_input"].closed


@pytest.mark.parametrize("failure", [None, "null", "empty", "line", "column", "reader", "sql", "parquet", "http", "parser", "interrupt"])
def test_append_csv_managed_cleanup_and_preupload_validation(tmp_path, monkeypatch, failure):
    import duckdb
    import polars as pl

    csv_module = importlib.import_module("intermine314.lists._csv")
    service, session = client()
    append_route(session)
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    text = {"null": "id\n0007\n\n", "empty": 'id\n""\n', "line": 'id\n"line\nbreak"\n',
            "column": "other\n0007\n"}.get(failure, "id\n0007\n")
    source = StringIO(text)
    if failure == "http":
        session.routes[("POST", "/service/lists/append")] = (500, b"failed")
    if failure == "parser":
        session.routes[("POST", "/service/lists/append")] = b"invalid json"
    directories, connections, readers, paths = [], [], [], []
    original_temporary, original_connect, original_sink = csv_module.TemporaryDirectory, duckdb.connect, pl.LazyFrame.sink_parquet
    scratch_root = tmp_path / "scratch[1]"
    scratch_root.mkdir()
    def temporary(**kwargs):
        directory = original_temporary(dir=scratch_root, **kwargs)
        directories.append(Path(directory.name))
        return directory
    class Reader:
        def __init__(self, wrapped):
            self.wrapped, self.closed = wrapped, False
        def __iter__(self):
            if failure == "reader":
                raise RuntimeError("reader failed")
            if failure == "interrupt":
                raise KeyboardInterrupt("reader interrupted")
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
        def to_arrow_reader(self, size):
            reader = Reader(self.wrapped.to_arrow_reader(size))
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
        paths.append(Path(path))
        if failure == "parquet":
            raise RuntimeError("Parquet failed")
        return original_sink(frame, path, **kwargs)
    monkeypatch.setattr(csv_module, "TemporaryDirectory", temporary)
    monkeypatch.setattr(duckdb, "connect", connect)
    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", sink)
    invoke = partial(item.append, csv_input=source, csv_column="id")
    if failure is None:
        assert invoke() is item
    else:
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else (ValueError, RuntimeError, WebserviceError)):
            invoke()
    assert not source.closed
    assert all(not directory.exists() for directory in directories)
    assert all(path.suffix == ".parquet" for path in paths)
    assert all(connection.closed for connection in connections) and all(reader.closed for reader in readers)
    assert list(scratch_root.iterdir()) == []
    if failure in ("null", "empty", "line", "column", "reader", "sql", "parquet", "interrupt"):
        assert not any(r.method == "POST" for r in session.requests)
    assert_closed(session)


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_tags_exact_methods_semicolon_refresh_and_returns(profile):
    service, session = client(profile)
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    for method, tags in [("POST", ["tag α", "new &+ 人"]), ("DELETE", ["new &+ 人"]), ("GET", ["server κ"])]:
        session.routes[(method, "/service/list/tags")] = json.dumps({"wasSuccessful": True, "tags": tags}).encode()
    assert item.add_tags("tag α", "new &+ 人") is None
    assert item.tags == frozenset(["tag α", "new &+ 人"])
    assert item.remove_tags("tag α") is None and item.tags == frozenset(["new &+ 人"])
    assert item.update_tags("ignored upstream argument") is None and item.tags == frozenset(["server κ"])
    assert manager.add_tags(item, ["tag α"]) == ["tag α", "new &+ 人"]
    assert manager.remove_tags(item, ["tag α"]) == ["new &+ 人"]
    assert manager.get_tags(item) == ["server κ"]
    calls = [r for r in session.requests if r.path == "/service/list/tags"]
    assert [r.method for r in calls] == ["POST", "DELETE", "GET"] * 2
    assert parse_qs(calls[0].data.decode()) == {"name": [item.name], "tags": ["tag α;new &+ 人"]}
    assert params(calls[1])["tags"] == ["tag α"]
    assert params(calls[2]) == {"name": [item.name], "token": ["secret"]}
    assert_closed(session)


@pytest.mark.parametrize("method", ["add_tags", "remove_tags", "update_tags"])
@pytest.mark.parametrize("body", [b"invalid json", b'{"wasSuccessful":false,"error":"denied"}'])
def test_tags_errors_preserve_cached_tags_and_close(method, body):
    service, session = client()
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    verb = {"add_tags": "POST", "remove_tags": "DELETE", "update_tags": "GET"}[method]
    session.routes[(verb, "/service/list/tags")] = body
    with pytest.raises(manager_module().ListServiceError):
        getattr(item, method)("new")
    assert item.tags == frozenset(["tag α"])
    assert_closed(session)


@pytest.mark.parametrize("method", ["add_tags", "remove_tags", "update_tags"])
@pytest.mark.parametrize("failure", ["http", "interrupt"])
def test_tags_http_errors_and_interruptions_do_not_retry(monkeypatch, method, failure):
    service, session = client()
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    verb = {"add_tags": "POST", "remove_tags": "DELETE", "update_tags": "GET"}[method]
    session.routes[(verb, "/service/list/tags")] = ((500, b"failed") if failure == "http"
                                                   else b'{"wasSuccessful":true,"tags":[]}')
    if failure == "interrupt":
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt("read interrupted")
        monkeypatch.setattr(_ResponseStreamAdapter, "read", interrupted)
    with pytest.raises(WebserviceError if failure == "http" else KeyboardInterrupt):
        getattr(item, method)("new")
    assert item.tags == frozenset(["tag α"])
    assert sum(r.path == "/service/list/tags" for r in session.requests) == 1
    assert_closed(session)


def test_context_returns_manager_deletes_unnamed_retains_named_and_renamed(monkeypatch):
    service, session = client()
    original_request = session.request
    metadata = json.loads(session.routes[("GET", "/service/lists")])["lists"][0]
    inventory = {metadata["name"]: metadata}
    def publish():
        session.routes[("GET", "/service/lists")] = json.dumps({"wasSuccessful": True, "lists": list(inventory.values())}).encode()
    def request(method, url, **kwargs):
        query = parse_qs(urlsplit(url).query)
        if method == "POST":
            name = query["name"][0]
            inventory[name] = dict(metadata, name=name)
            session.routes[(method, "/service/lists")] = json.dumps({"wasSuccessful": True, "listName": name}).encode()
        elif method == "GET" and urlsplit(url).path == "/service/lists/rename":
            old, new = query["oldname"][0], query["newname"][0]
            inventory[new] = dict(inventory.pop(old), name=new)
            session.routes[(method, "/service/lists/rename")] = json.dumps({"wasSuccessful": True, "listName": new}).encode()
        response = original_request(method, url, **kwargs)
        if method == "DELETE":
            inventory.pop(query["name"][0])
        publish()
        return response
    monkeypatch.setattr(session, "request", request)
    manager = service.list_manager()
    with manager as entered:
        assert entered is manager
        temporary = manager.create_list(["0007"], "Employee")
        named = manager.create_list(["0007"], "Employee", name="named")
        renamed = manager.create_list(["0007"], "Employee")
        renamed.name = "retained α"
    assert manager._temp_lists == set()
    assert set(inventory) == {"identifiers", named.name, renamed.name}
    assert [params(r)["name"][0] for r in session.requests if r.method == "DELETE"] == [temporary.name]
    before = len(session.requests)
    manager.delete_temporary_lists()
    assert len(session.requests) == before
    assert_closed(session)


def test_temporary_cleanup_partial_failure_retries_remaining_names(monkeypatch):
    service, session = client()
    session.routes[("GET", "/service/lists")] = fixture_bytes("lists.json")
    manager = service.list_manager()
    manager._temp_lists.update(["test-list-1", "test-list-2"])
    original = session.request
    attempts = []
    def request(method, url, **kwargs):
        if method == "DELETE":
            attempts.append(parse_qs(urlsplit(url).query)["name"][0])
            if len(attempts) == 2:
                session.routes[(method, "/service/lists")] = b'{"wasSuccessful":false,"error":"denied"}'
        return original(method, url, **kwargs)
    monkeypatch.setattr(session, "request", request)
    with pytest.raises(manager_module().ListServiceError, match="denied"):
        manager.delete_temporary_lists()
    assert manager._temp_lists == {attempts[1]}
    session.routes[("DELETE", "/service/lists")] = b'{"wasSuccessful":true}'
    manager.delete_temporary_lists()
    assert attempts == [attempts[0], attempts[1], attempts[1]] and manager._temp_lists == set()
    assert_closed(session)


@pytest.mark.parametrize("body_error", [ValueError("body failed"), KeyboardInterrupt("body interrupted")])
def test_context_preserves_body_error_with_observable_cleanup_failure(body_error):
    service, session = client()
    manager = service.list_manager()
    manager._temp_lists.add("identifiers")
    session.routes[("DELETE", "/service/lists")] = b'{"wasSuccessful":false,"error":"cleanup denied"}'
    with pytest.raises(type(body_error)) as captured:
        with manager:
            raise body_error
    assert captured.value is body_error
    assert any("cleanup denied" in note for note in captured.value.__notes__)
    assert manager._temp_lists == {"identifiers"}
    assert_closed(session)


def test_context_cleanup_failure_without_body_propagates_and_keeps_tracking():
    service, session = client()
    manager = service.list_manager()
    manager._temp_lists.add("identifiers")
    session.routes[("DELETE", "/service/lists")] = b'{"wasSuccessful":false,"error":"denied"}'
    with pytest.raises(manager_module().ListServiceError, match="denied"):
        with manager:
            pass
    assert manager._temp_lists == {"identifiers"}
    assert_closed(session)


@pytest.mark.parametrize("cleanup_error", [KeyboardInterrupt("cleanup interrupted"), SystemExit("cleanup interrupted")])
def test_context_does_not_suppress_cleanup_interrupt(monkeypatch, cleanup_error):
    service, _ = client()
    manager = service.list_manager()
    def interrupted():
        raise cleanup_error
    monkeypatch.setattr(manager, "delete_temporary_lists", interrupted)
    body_error = ValueError("body failed")
    with pytest.raises(type(cleanup_error), match="cleanup interrupted") as captured:
        with manager:
            raise body_error
    assert captured.value.__context__ is body_error


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_flush_cleans_internal_only_invalidates_all_caches_and_keeps_transport(profile):
    service, session = client(profile)
    internal = service._get_list_manager()
    internal._temp_lists.add("identifiers")
    external = service.list_manager()
    external._temp_lists.add("caller-owned")
    old_model = service.model
    service._resolve_query_model()
    service.release
    service.widgets
    for name in ("_templates", "_templates_raw", "_all_templates", "_all_templates_raw", "_all_templates_names"):
        setattr(service, name, {"stale": object()})
    opener = service.opener
    assert service.flush() is None
    assert internal._temp_lists == set() and external._temp_lists == {"caller-owned"}
    assert service._list_manager is None
    fresh = service._get_list_manager()
    assert fresh is not internal and fresh.lists is None and fresh._temp_lists == set()
    for name in ("_model", "_model_xml", "_model_name", "_query_model", "_version", "_release", "_widgets", "_templates", "_templates_raw", "_all_templates", "_all_templates_raw", "_all_templates_names"):
        assert getattr(service, name) is None
    assert service.opener is opener and opener.token == "secret"
    assert [params(r)["name"][0] for r in session.requests if r.method == "DELETE"] == ["identifiers"]
    assert service.model is not old_model
    assert_closed(session)


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_flush_template_caches_survive_failed_cleanup_until_success(profile):
    service, session = client(profile)
    session.routes.update({
        ("GET", "/service/templates"): fixture_bytes("templates.xml"),
        ("GET", "/service/alltemplates"): fixture_bytes("all-templates.xml"),
    })
    manager = service._get_list_manager()
    manager._temp_lists.add("identifiers")
    global_template = service.get_template("employeeByName")
    user_template = service.get_template_by_user("shared", "alice")
    service.all_templates_names
    fields = ("_templates", "_templates_raw", "_all_templates", "_all_templates_raw", "_all_templates_names")
    caches = {field: getattr(service, field) for field in fields}
    session.routes[("DELETE", "/service/lists")] = b'{"wasSuccessful":false,"error":"denied"}'
    with pytest.raises(manager_module().ListServiceError):
        service.flush()
    assert all(getattr(service, field) is cache for field, cache in caches.items())
    assert service.get_template("employeeByName") is global_template
    assert service.get_template_by_user("shared", "alice") is user_template
    assert service._list_manager is manager
    session.routes[("DELETE", "/service/lists")] = b'{"wasSuccessful":true}'
    assert service.flush() is None
    assert manager._temp_lists == set() and service._list_manager is None
    assert all(getattr(service, field) is None for field in fields)
    assert_closed(session)


def test_flush_unallocated_manager_does_not_make_list_request():
    service, session = client()
    before = len(session.requests)
    assert service.flush() is None
    assert service._list_manager is None and len(session.requests) == before
    assert_closed(session)


@pytest.mark.parametrize("failure", ["protocol", "interrupt"])
def test_flush_failure_retains_manager_caches_and_transport(monkeypatch, failure):
    service, session = client()
    manager = service._get_list_manager()
    manager._temp_lists.add("identifiers")
    model, widgets = service.model, service.widgets
    if failure == "protocol":
        session.routes[("DELETE", "/service/lists")] = b'{"wasSuccessful":false,"error":"denied"}'
        error = manager_module().ListServiceError
    else:
        def interrupted():
            raise KeyboardInterrupt("cleanup interrupted")
        monkeypatch.setattr(manager, "delete_temporary_lists", interrupted)
        error = KeyboardInterrupt
    with pytest.raises(error):
        service.flush()
    assert service._list_manager is manager and manager._temp_lists == {"identifiers"}
    assert service._model is model and service._widgets is widgets
    assert service.opener.token == "secret"
    assert_closed(session)
