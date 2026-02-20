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


@pytest.mark.xfail(reason="Phase 2 pending: incremental cache refresh", strict=True)
def test_parse_list_upload_response_refreshes_once_not_n_times():
    manager = ListManager.__new__(ListManager)
    manager.service = object()
    manager._temp_lists = set()
    manager.lists = None

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
        def get_list_append_uri(self):
            return "https://example.org/service/query/append/tolist"

        def to_query_params(self):
            return {"query": "<xml/>"}

    class _FakeManager:
        def _get_listable_query(self, value):
            return value

        def parse_list_upload_response(self, _data):
            return _UploadResult(size=7)

    manager = _FakeManager()
    im_list = _make_list_for_manager(manager, _FakeService())

    im_list._do_append(_Queryable())

    assert response.closed is True


@pytest.mark.xfail(reason="Phase 2 pending: bounded unmatched retention", strict=True)
def test_unmatched_identifier_sample_is_bounded():
    class _FakeService:
        pass

    class _FakeManager:
        pass

    im_list = _make_list_for_manager(_FakeManager(), _FakeService())
    im_list._add_failed_matches([f"id_{i}" for i in range(6000)])

    assert len(im_list.unmatched_identifiers) <= 5000


@pytest.mark.xfail(reason="Phase 2 pending: remove len(content) probe", strict=True)
def test_create_list_does_not_call_len_on_query_like_content():
    payload = b'{"wasSuccessful": true, "listName": "tmp"}'
    opener = _FakeCreationOpener(payload=payload)
    service = _FakeCreationService(opener=opener)
    manager = ListManager(service)
    manager.parse_list_upload_response = lambda _response: "parsed-ok"

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
