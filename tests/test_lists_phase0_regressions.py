from __future__ import annotations

import inspect
import json
import logging

import pytest

from intermine314.lists.list import List
from intermine314.lists import listmanager as listmanager_module
from intermine314.lists.listmanager import ListManager


class _TrackingResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.closed = False

    def read(self):
        return self._payload

    def close(self):
        self.closed = True


class _UploadResult:
    def __init__(self, size=0):
        self.unmatched_identifiers = set()
        self.size = size


class _FakeListQuery:
    def __init__(self):
        self.constraints = []
        self.views = []
        self.root = type("_Root", (), {"name": "Gene"})()

    def add_constraint(self, *args):
        self.constraints.append(args)

    def to_query(self):
        return self

    def add_view(self, value):
        self.views.append(value)

    def select(self, *_args, **_kwargs):
        return None


class _FakeServiceForOrganism:
    root = "https://example.org/service"

    def __init__(self):
        self.created_queries = []

    def new_query(self, _list_type):
        query = _FakeListQuery()
        self.created_queries.append(query)
        return query


class _LenBombContent:
    def __len__(self):
        raise AssertionError("len(content) must not be called")

    def strip(self):
        return "idA\nidB"


class _FakeCreationOpener:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post_plain_text(self, url, ids):
        self.calls.append((url, ids))
        return self.payload


class _FakeCreationService:
    root = "https://example.org/service"
    LIST_CREATION_PATH = "/lists"

    def __init__(self, opener):
        self.opener = opener


def _make_list_for_manager(manager, service):
    return List(
        service=service,
        manager=manager,
        name="L",
        title="L",
        type="Gene",
        size=0,
        tags=[],
    )


class _FakeBatchQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self.added_views = []
        self.added_constraints = []

    def add_view(self, value):
        self.added_views.append(value)

    def add_constraint(self, *args):
        self.added_constraints.append(args)

    def count(self):
        return len(self._rows)

    def results(self, *, row, start, size):
        assert row == "list"
        for item in self._rows[start : start + size]:
            yield [item]


def test_create_list_with_organism_does_not_construct_new_service(monkeypatch):
    class _UnexpectedService:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("create_list(organism=...) must not construct a new Service")

    monkeypatch.setattr("intermine314.service.service.Service", _UnexpectedService)

    service = _FakeServiceForOrganism()
    manager = ListManager(service)
    sentinel = object()
    manager._create_list_from_queryable = lambda *_args, **_kwargs: sentinel

    result = manager.create_list(["AT1G01010"], list_type="Gene", name="tmp", organism="Arabidopsis")

    assert result is sentinel
    assert service.created_queries


def test_parse_list_upload_response_refreshes_once_not_n_times():
    class _Service:
        pass

    manager = ListManager(_Service())
    manager.lists = None
    manager._lists_cache_valid = False

    class _CachedList:
        def __init__(self):
            self.failed = []

        def _add_failed_matches(self, ids):
            self.failed.append(list(ids or []))

    cached = _CachedList()
    refresh_calls = {"count": 0}

    def _refresh_lists():
        refresh_calls["count"] += 1
        manager.lists = {"L1": cached}

    manager.refresh_lists = _refresh_lists
    response = json.dumps({"wasSuccessful": True, "listName": "L1", "unmatchedIdentifiers": ["x"]}).encode("utf-8")

    manager.parse_list_upload_response(response)
    manager.parse_list_upload_response(response)
    manager.parse_list_upload_response(response)

    assert refresh_calls["count"] == 1


def test_append_queryable_response_is_closed():
    response = _TrackingResponse(b'{"wasSuccessful": true, "listName": "L"}')

    class _FakeOpener:
        def open(self, _url, _form):
            return response

        def post_plain_text(self, *_args, **_kwargs):
            raise AssertionError("queryable append path should not use post_plain_text")

    class _FakeService:
        root = "https://example.org/service"
        LIST_APPENDING_PATH = "/lists/append"
        opener = _FakeOpener()

    class _Queryable:
        views = []
        root = type("_Root", (), {"name": "Gene"})()

        def to_query(self):
            return self

        def add_view(self, *_args, **_kwargs):
            return None

        def select(self, *_args, **_kwargs):
            return None

        def get_list_append_uri(self):
            return "https://example.org/service/query/append/tolist"

        def to_query_params(self):
            return {"query": "<xml/>"}

    service = _FakeService()
    manager = ListManager(service)
    manager.parse_list_upload_response = lambda _data, **_kwargs: _UploadResult(size=7)
    manager.append_to_list_content(
        list_name="L",
        content=_Queryable(),
        list_type="Gene",
        description="desc",
        tags=[],
    )

    assert response.closed is True


def test_unmatched_identifier_sample_is_bounded():
    class _FakeService:
        pass

    class _FakeManager:
        pass

    im_list = _make_list_for_manager(_FakeManager(), _FakeService())
    im_list._add_failed_matches([f"id_{i}" for i in range(6000)])

    assert len(im_list.unmatched_identifiers) <= 5000


def test_create_list_does_not_call_len_on_query_like_content():
    payload = b'{"wasSuccessful": true, "listName": "tmp"}'
    opener = _FakeCreationOpener(payload=payload)
    service = _FakeCreationService(opener=opener)
    manager = ListManager(service)
    manager.parse_list_upload_response = lambda _response, **_kwargs: "parsed-ok"

    result = manager.create_list(
        _LenBombContent(),
        list_type="Gene",
        name="tmp",
        description="desc",
        tags=[],
        add=[],
    )

    assert result == "parsed-ok"
    assert opener.calls == [("https://example.org/service/lists?name=tmp&type=Gene&description=desc&tags=", "idA\nidB")]


def test_create_list_iterable_is_chunked_and_appended(monkeypatch):
    class _ChunkingOpener:
        def __init__(self):
            self.calls = []

        def post_plain_text(self, url, payload):
            self.calls.append((url, payload))
            return b'{"wasSuccessful": true, "listName": "chunked"}'

    class _Service:
        root = "https://example.org/service"
        LIST_CREATION_PATH = "/lists"
        LIST_APPENDING_PATH = "/lists/append"

        def __init__(self, opener):
            self.opener = opener

    opener = _ChunkingOpener()
    service = _Service(opener)
    manager = ListManager(service)
    manager.parse_list_upload_response = lambda _response, **_kwargs: "ok"
    monkeypatch.setattr(manager, "DEFAULT_UPLOAD_CHUNK_SIZE", 5)

    ids = [f"id{i}" for i in range(12)]
    result = manager.create_list(ids, list_type="Gene", name="chunked", description="d", tags=[], add=[])

    assert result == "ok"
    assert len(opener.calls) == 3
    assert opener.calls[0][0].startswith("https://example.org/service/lists?")
    assert opener.calls[1][0] == "https://example.org/service/lists/append?name=chunked"
    assert opener.calls[2][0] == "https://example.org/service/lists/append?name=chunked"
    chunk_sizes = [len(payload.decode("utf-8").splitlines()) for _, payload in opener.calls]
    assert chunk_sizes == [5, 5, 2]


def test_delete_lists_updates_cache_without_forced_refresh():
    class _DeleteOpener:
        def __init__(self):
            self.calls = []

        def delete(self, url):
            self.calls.append(url)
            return b'{"wasSuccessful": true}'

    class _Service:
        root = "https://example.org/service"
        LIST_PATH = "/lists"

        def __init__(self, opener):
            self.opener = opener

    opener = _DeleteOpener()
    service = _Service(opener)
    manager = ListManager(service)
    manager.lists = {"L": _make_list_for_manager(manager=manager, service=service)}
    manager._lists_cache_valid = True
    manager.refresh_lists = lambda: (_ for _ in ()).throw(AssertionError("refresh_lists should not be called"))

    manager.delete_lists(["L"])

    assert opener.calls == ["https://example.org/service/lists?name=L"]
    assert "L" not in manager.lists


def test_bulk_mutation_defers_single_refresh_until_exit():
    class _Service:
        pass

    manager = ListManager(_Service())
    refresh_calls = {"count": 0}
    manager.refresh_lists = lambda: refresh_calls.__setitem__("count", refresh_calls["count"] + 1)

    with manager.bulk_mutation():
        manager._mark_cache_invalid()
        assert refresh_calls["count"] == 0

    assert refresh_calls["count"] == 1


def test_normalize_list_input_type_error_is_explicit():
    class _Service:
        pass

    manager = ListManager(_Service())

    with pytest.raises(TypeError, match="Unsupported list input type"):
        manager._normalize_list_input(object())


def test_normalize_list_input_query_does_not_trigger_query_len(monkeypatch):
    class _QueryLike:
        views = []
        root = type("_Root", (), {"name": "Gene"})()

        def __len__(self):
            raise AssertionError("Query len should not be called during normalization")

        def to_query(self):
            return self

        def add_view(self, *_args, **_kwargs):
            return None

        def select(self, *_args, **_kwargs):
            return None

    class _Service:
        pass

    manager = ListManager(_Service())
    normalized = manager._normalize_list_input(_QueryLike())
    assert normalized["content_type"] == "query"
    assert normalized["query"] is not None


def test_iter_entry_batches_yields_batched_identifiers():
    class _Service:
        def new_query(self, _list_type):
            return _FakeBatchQuery(["id1", "id2", "id3", "id4", "id5"])

    manager = type("_ManagerStub", (), {})()
    im_list = _make_list_for_manager(manager=manager, service=_Service())

    batches = list(im_list.iter_entry_batches(batch_size=2))

    assert batches == [["id1", "id2"], ["id3", "id4"], ["id5"]]


def test_get_entries_flattens_batches():
    class _Service:
        def new_query(self, _list_type):
            return _FakeBatchQuery(["id1", "id2", "id3"])

    manager = type("_ManagerStub", (), {})()
    im_list = _make_list_for_manager(manager=manager, service=_Service())

    assert list(im_list.get_entries(batch_size=2)) == ["id1", "id2", "id3"]


def test_to_polars_ids_returns_series_and_dataframe(monkeypatch):
    class _FakeSeries(list):
        def __init__(self, name, values):
            super().__init__(values)
            self.name = name

    class _FakePolars:
        @staticmethod
        def Series(name, values):
            return _FakeSeries(name, values)

        @staticmethod
        def concat(items, rechunk=True):
            _ = rechunk
            merged = []
            for item in items:
                merged.extend(item)
            return _FakeSeries(items[0].name, merged)

        @staticmethod
        def DataFrame(mapping):
            return {"type": "df", "mapping": mapping}

    monkeypatch.setattr("intermine314.lists.list.require_polars", lambda _ctx: _FakePolars)

    class _Service:
        def new_query(self, _list_type):
            return _FakeBatchQuery(["id1", "id2", "id3"])

    manager = type("_ManagerStub", (), {})()
    im_list = _make_list_for_manager(manager=manager, service=_Service())

    series = im_list.to_polars_ids(batch_size=2, column_name="gene_id")
    frame = im_list.to_polars_ids(batch_size=2, as_dataframe=True, column_name="gene_id")

    assert list(series) == ["id1", "id2", "id3"]
    assert series.name == "gene_id"
    assert frame["type"] == "df"
    assert list(frame["mapping"]["gene_id"]) == ["id1", "id2", "id3"]


def test_update_tags_replaces_existing_tags():
    class _Manager:
        def __init__(self):
            self.removed = []
            self.added = []

        def remove_tags(self, _list_obj, tags):
            self.removed.append(tuple(tags))
            return []

        def add_tags(self, _list_obj, tags):
            self.added.append(tuple(tags))
            return list(tags)

    manager = _Manager()
    service = type("_Service", (), {})()
    im_list = List(
        service=service,
        manager=manager,
        name="L",
        title="L",
        type="Gene",
        size=0,
        tags=["old_a", "old_b"],
    )

    im_list.update_tags("new_a", "new_b")

    assert len(manager.removed) == 1
    assert set(manager.removed[0]) == {"old_a", "old_b"}
    assert manager.added == [("new_a", "new_b")]
    assert im_list.tags == frozenset({"new_a", "new_b"})


def test_listmanager_module_has_no_import_time_basicconfig_call():
    source = inspect.getsource(listmanager_module)
    assert "logging.basicConfig(" not in source


def test_refresh_lists_debug_logs_do_not_include_full_payload(caplog):
    sensitive_name = "SENSITIVE_IDENTIFIER_123"
    payload = json.dumps(
        {
            "wasSuccessful": True,
            "lists": [
                {
                    "name": "list1",
                    "title": "List 1",
                    "type": "Gene",
                    "size": 1,
                    "tags": [sensitive_name],
                }
            ],
        }
    )

    class _Opener:
        def read(self, _url):
            return payload

    class _Service:
        root = "https://example.org/service"
        LIST_PATH = "/lists"
        opener = _Opener()

    service = _Service()
    manager = ListManager(service)
    with caplog.at_level(logging.DEBUG, logger="intermine314.lists.listmanager"):
        manager.refresh_lists()
    assert "payload_bytes=" in caplog.text
    assert "list_count=" in caplog.text
    assert sensitive_name not in caplog.text


def test_parse_upload_debug_logs_do_not_include_unmatched_identifiers(caplog):
    manager = ListManager.__new__(ListManager)
    manager.service = object()
    manager._temp_lists = set()
    manager._lists_cache_valid = True
    manager._bulk_mutation_depth = 0
    manager._bulk_refresh_pending = False
    manager_stub = type("_ManagerStub", (), {})()
    service_stub = type("_ServiceStub", (), {})()
    manager.lists = {"L1": _make_list_for_manager(manager=manager_stub, service=service_stub)}
    manager.refresh_lists = lambda: None
    manager.get_list = lambda _name: manager.lists["L1"]

    response = json.dumps(
        {
            "wasSuccessful": True,
            "listName": "L1",
            "unmatchedIdentifiers": ["PII_GENE_A", "PII_GENE_B"],
        }
    ).encode("utf-8")

    with caplog.at_level(logging.DEBUG, logger="intermine314.lists.listmanager"):
        manager.parse_list_upload_response(response)
    assert "response_keys=" in caplog.text
    assert "unmatched_count=2" in caplog.text
    assert "PII_GENE_A" not in caplog.text
    assert "PII_GENE_B" not in caplog.text
