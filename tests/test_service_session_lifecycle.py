from __future__ import annotations

from collections import OrderedDict

import pytest

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


class _TrackingCachedService:
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


def test_service_context_manager_closes_opener_once():
    service = _make_service_with_tracking_opener()

    with service as managed:
        assert managed is service

    assert service.opener.close_calls == 1
    assert service._closed is True


def _make_registry_with_tracking_session(*, owns_session):
    registry = Registry.__new__(Registry)
    registry._closed = False
    registry._owns_session = bool(owns_session)
    registry._session = _TrackingSession()
    registry._opener = None
    registry._cache_hits = 0
    registry._cache_misses = 0
    registry._cache_evictions = 0
    registry._cache_clears = 0
    registry._cache_closed_services = 0
    registry._max_cached_services = 4
    registry._Registry__mine_dict = {}
    registry._Registry__synonyms = {}
    registry._Registry__mine_cache = OrderedDict([("mine-a", object())])
    return registry


@pytest.mark.parametrize(
    ("owns_session", "expected_session_close_calls"),
    (
        (True, 1),
        (False, 0),
    ),
)
def test_registry_close_respects_session_ownership(owns_session, expected_session_close_calls):
    registry = _make_registry_with_tracking_session(owns_session=owns_session)
    tracked_session = registry._session

    registry.close()
    registry.close()

    assert tracked_session.close_calls == expected_session_close_calls
    assert registry._closed is True
    assert registry._session is None
    assert registry._owns_session is False


def test_registry_clear_cache_closes_cached_services_and_updates_metrics():
    registry = _make_registry_with_tracking_session(owns_session=False)
    svc_a = _TrackingCachedService()
    svc_b = _TrackingCachedService()
    registry._Registry__mine_cache = OrderedDict([("mine-a", svc_a), ("mine-b", svc_b)])

    cleared = registry.clear_cache()

    assert cleared == 2
    assert len(registry._Registry__mine_cache) == 0
    assert svc_a.close_calls == 1
    assert svc_b.close_calls == 1
    metrics = registry.service_cache_metrics()
    assert metrics["cache_clears"] == 1
    assert metrics["cache_closed_services"] == 2


def test_registry_context_manager_closes_owned_session():
    registry = _make_registry_with_tracking_session(owns_session=True)
    tracked_session = registry._session

    with registry as managed:
        assert managed is registry

    assert tracked_session.close_calls == 1
    assert registry._closed is True
