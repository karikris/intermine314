from __future__ import annotations

import json

import pytest

from intermine314.service.errors import ServiceError
from intermine314.service.service import Service


class _FakeList:
    def __init__(self, name):
        self.name = name


class _ObservedIdentifiers:
    def __init__(self, values):
        self._values = list(values)
        self.seen = []

    def __iter__(self):
        for value in self._values:
            self.seen.append(value)
            yield value


class _ResolveOpener:
    def __init__(self, fail_on_prefix="bad"):
        self.calls = []
        self._fail_on_prefix = fail_on_prefix

    def post_content(self, url, body, mimetype):
        _ = mimetype
        payload = json.loads(body)
        self.calls.append((url, payload))
        identifiers = payload["identifiers"]
        if identifiers and identifiers[0].startswith(self._fail_on_prefix):
            return json.dumps({"uid": None, "error": "batch failed"})
        return json.dumps({"uid": f"uid_{len(self.calls)}", "error": None})


def _make_service():
    service = Service.__new__(Service)
    service.root = "https://example.org/service"
    service._version = 10
    service._list_manager = None
    return service


def test_iter_created_lists_streams_batches_from_generator():
    service = _make_service()
    created_calls = []

    def _fake_create_list(content, **kwargs):
        created_calls.append((list(content), dict(kwargs)))
        return _FakeList(kwargs["name"])

    service.create_list = _fake_create_list
    identifiers = _ObservedIdentifiers(["id1", "id2", "id3"])
    created_iter = service.iter_created_lists(identifiers, list_type="Gene", chunk_size=2, name_prefix="tmp")

    first = next(created_iter)
    assert first.name.startswith("tmp_")
    assert identifiers.seen == ["id1", "id2"]

    second = next(created_iter)
    assert second.name.startswith("tmp_")
    assert identifiers.seen == ["id1", "id2", "id3"]

    with pytest.raises(StopIteration):
        next(created_iter)

    assert [payload for payload, _kwargs in created_calls] == [["id1", "id2"], ["id3"]]


def test_create_batched_lists_keeps_legacy_list_return_type():
    service = _make_service()
    service.create_list = lambda content, **kwargs: _FakeList(kwargs["name"])

    created = service.create_batched_lists(
        (item for item in ["id1", "id2", "id3"]),
        list_type="Gene",
        chunk_size=2,
        name_prefix="legacy",
    )

    assert isinstance(created, list)
    assert len(created) == 2
    assert all(isinstance(item, _FakeList) for item in created)


def test_iter_resolve_ids_supports_generator_and_reports_partial_failures():
    service = _make_service()
    service.opener = _ResolveOpener()

    results = list(
        service.iter_resolve_ids(
            "Gene",
            (item for item in ["id1", "id2", "bad3", "bad4", "id5"]),
            chunk_size=2,
        )
    )

    assert [entry["identifier_count"] for entry in results] == [2, 2, 1]
    assert results[0]["job"] is not None
    assert results[0]["uid"] == "uid_1"
    assert results[1]["job"] is None
    assert results[1]["error"] == "batch failed"
    assert results[2]["job"] is not None
    assert results[2]["uid"] == "uid_3"


def test_iter_resolve_ids_streams_without_pre_materializing():
    service = _make_service()
    service.opener = _ResolveOpener(fail_on_prefix="never")
    identifiers = _ObservedIdentifiers(["id1", "id2", "id3"])
    result_iter = service.iter_resolve_ids("Gene", identifiers, chunk_size=2)

    first = next(result_iter)
    assert first["uid"] == "uid_1"
    assert identifiers.seen == ["id1", "id2"]

    second = next(result_iter)
    assert second["uid"] == "uid_2"
    assert identifiers.seen == ["id1", "id2", "id3"]

    with pytest.raises(StopIteration):
        next(result_iter)


def test_iter_resolve_ids_rejects_empty_identifier_stream():
    service = _make_service()
    service.opener = _ResolveOpener(fail_on_prefix="never")

    with pytest.raises(ServiceError, match="No identifiers supplied"):
        list(service.iter_resolve_ids("Gene", iter(()), chunk_size=2))


def test_resolve_ids_chunked_alias_streams_batches():
    service = _make_service()
    service.opener = _ResolveOpener(fail_on_prefix="never")

    rows = list(service.resolve_ids_chunked("Gene", ["id1", "id2", "id3"], chunk_size=2))

    assert [row["identifier_count"] for row in rows] == [2, 1]
    assert [row["uid"] for row in rows] == ["uid_1", "uid_2"]
