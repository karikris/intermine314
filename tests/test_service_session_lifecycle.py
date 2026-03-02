from __future__ import annotations

from collections import OrderedDict

from intermine314.service.service import Registry, Service


class _TrackingListManager:
    def __init__(self):
        self.delete_calls = 0

    def delete_temporary_lists(self):
        self.delete_calls += 1


class _TrackingOpener:
    def __init__(self):
        self.close_calls = 0
        self._owns_session = True
        self._session = object()

    def close(self):
        self.close_calls += 1


class _TrackingSession:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _make_service_with_tracking_opener():
    service = Service.__new__(Service)
    service._closed = False
    service._list_manager = _TrackingListManager()
    service.opener = _TrackingOpener()
    service._owns_session = True
    return service


def test_service_close_is_idempotent_and_closes_opener():
    service = _make_service_with_tracking_opener()

    service.close()
    service.close()

    assert service._list_manager.delete_calls == 1
    assert service.opener.close_calls == 1
    assert service._closed is True


def test_service_context_manager_calls_close():
    service = _make_service_with_tracking_opener()

    with service as managed:
        assert managed is service

    assert service._closed is True
    assert service._list_manager.delete_calls == 1
    assert service.opener.close_calls == 1


def _make_registry_with_tracking_session(*, owns_session):
    registry = Registry.__new__(Registry)
    registry._closed = False
    registry._owns_session = bool(owns_session)
    registry._session = _TrackingSession()
    registry._opener = None
    registry._Registry__mine_cache = OrderedDict([("mine-a", object())])
    return registry


def test_registry_close_closes_owned_session_and_clears_cache():
    registry = _make_registry_with_tracking_session(owns_session=True)

    registry.close()
    registry.close()

    assert registry._closed is True
    assert registry._session is None
    assert registry._owns_session is False
    assert len(registry._Registry__mine_cache) == 0


def test_registry_close_does_not_close_caller_session():
    registry = _make_registry_with_tracking_session(owns_session=False)
    tracked_session = registry._session

    registry.close()

    assert tracked_session.close_calls == 0
    assert registry._closed is True


def test_registry_context_manager_calls_close():
    registry = _make_registry_with_tracking_session(owns_session=True)

    with registry as managed:
        assert managed is registry

    assert registry._closed is True
    assert registry._session is None


def test_service_get_mine_info_closes_registry(monkeypatch):
    events = []

    class _FakeRegistry:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def info(self, name):
            events.append(("info", name))
            return {"name": name}

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr("intermine314.service.service.Registry", _FakeRegistry)

    service = Service.__new__(Service)
    service.request_timeout = 12
    service.proxy_url = "socks5h://127.0.0.1:9050"
    service.verify_tls = True
    service.tor = True
    service.strict_tor_proxy_scheme = True
    service.allow_insecure_tor_proxy_scheme = False
    service.allow_http_over_tor = False
    service.user_agent = "agent/1.0"

    result = service.get_mine_info("thalemine")

    assert result == {"name": "thalemine"}
    assert ("info", "thalemine") in events
    assert ("close", None) in events


def test_service_get_all_mines_closes_registry(monkeypatch):
    events = []

    class _FakeRegistry:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def all_mines(self, organism=None):
            events.append(("all_mines", organism))
            return [{"name": "MineA"}]

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr("intermine314.service.service.Registry", _FakeRegistry)

    result = Service.get_all_mines(organism="Zea mays", request_timeout=9)

    assert result == [{"name": "MineA"}]
    assert ("all_mines", "Zea mays") in events
    assert ("close", None) in events
