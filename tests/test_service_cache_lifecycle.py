from __future__ import annotations

import pytest

from intermine314.service.service import Registry, Service


class _FakeListManager:
    def delete_temporary_lists(self):
        return None


class _FakeTemplate:
    calls = []

    @classmethod
    def from_xml(cls, xml, model, service):
        inst = cls()
        inst.xml = xml
        inst.model = model
        inst.service = service
        cls.calls.append((xml, model, service, inst))
        return inst


class _CloseTrackingResponse:
    def __init__(self, payload):
        self._payload = payload
        self.closed = False

    def read(self):
        return self._payload

    def close(self):
        self.closed = True


class _CloseTrackingRegistryOpener:
    last_instance = None

    def __init__(self, *args, **kwargs):
        _ = (args, kwargs)
        self._session = object()
        self.responses = []
        _CloseTrackingRegistryOpener.last_instance = self

    def open(self, _url, *_args, **_kwargs):
        resp = _CloseTrackingResponse(b'{"instances": []}')
        self.responses.append(resp)
        return resp


class _CloseTrackingOpener:
    def __init__(self, payload_by_url):
        self._payload_by_url = {
            key: list(value) if isinstance(value, (list, tuple)) else [value]
            for key, value in payload_by_url.items()
        }
        self.responses = []

    def open(self, url, *_args, **_kwargs):
        queue = self._payload_by_url[url]
        payload = queue.pop(0)
        resp = _CloseTrackingResponse(payload)
        self.responses.append((url, resp))
        return resp


def _make_template_service():
    service = Service.__new__(Service)
    service._all_templates = {
        "alice": {"A": '<template name="A"></template>'},
        "bob": {"B": '<template name="B"></template>'},
    }
    service._model = object()
    service._list_manager = _FakeListManager()
    return service


def _make_transport_service(opener):
    service = Service.__new__(Service)
    service.root = "https://example.org/service"
    service._version = None
    service._list_manager = _FakeListManager()
    service.opener = opener
    return service


def test_get_template_by_user_keeps_nested_cache_shape(monkeypatch):
    _FakeTemplate.calls = []
    monkeypatch.setattr("intermine314.service.service._query_classes", lambda: (object, _FakeTemplate))
    service = _make_template_service()

    _ = service.get_template_by_user("A", "alice")
    _ = service.get_template_by_user("B", "bob")

    assert set(service.all_templates.keys()) == {"alice", "bob"}
    assert isinstance(service.all_templates["alice"], dict)
    assert isinstance(service.all_templates["bob"], dict)
    assert "A" in service.all_templates["alice"]
    assert "B" in service.all_templates["bob"]
    assert "A" not in service.all_templates
    assert "B" not in service.all_templates


def test_get_template_by_user_repeated_fetch_does_not_add_top_level_keys(monkeypatch):
    _FakeTemplate.calls = []
    monkeypatch.setattr("intermine314.service.service._query_classes", lambda: (object, _FakeTemplate))
    service = _make_template_service()
    before = set(service.all_templates.keys())

    first = service.get_template_by_user("A", "alice")
    second = service.get_template_by_user("A", "alice")

    assert first is second
    assert set(service.all_templates.keys()) == before
    assert len(_FakeTemplate.calls) == 1


def test_registry_fetch_closes_response(monkeypatch):
    monkeypatch.setattr("intermine314.service.service.InterMineURLOpener", _CloseTrackingRegistryOpener)
    _CloseTrackingRegistryOpener.last_instance = None

    Registry("https://registry.example.org/service/instances")

    inst = _CloseTrackingRegistryOpener.last_instance
    assert inst is not None
    assert inst.responses
    assert inst.responses[0].closed is True


def test_service_version_read_closes_response():
    root = "https://example.org/service"
    opener = _CloseTrackingOpener({f"{root}/version/ws": b"35"})
    service = _make_transport_service(opener)

    assert service.version == 35
    assert len(opener.responses) == 1
    assert opener.responses[0][1].closed is True
    assert service.version == 35
    assert len(opener.responses) == 1


def test_service_resolve_service_path_closes_response():
    root = "https://example.org/service"
    opener = _CloseTrackingOpener({f"{root}/check/widgets": b"ok"})
    service = _make_transport_service(opener)

    assert service.resolve_service_path("widgets") == b"ok"
    assert len(opener.responses) == 1
    assert opener.responses[0][1].closed is True

