import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from intermine314.service.service import Registry, Service
from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession, fixture_bytes


def test_cold_model_and_template_metadata_initialize_once():
    session = FixtureSession.service()
    session.routes['GET', '/service/templates'] = fixture_bytes('templates.xml')
    service = Service(SERVICE_ROOT, session=session)
    with ThreadPoolExecutor(8) as pool:
        models = list(pool.map(lambda _: service.model, range(24)))
        templates = list(pool.map(lambda _: service.templates, range(24)))
    assert all(model is models[0] for model in models)
    assert all(value is templates[0] for value in templates)
    paths = [request.path for request in session.requests]
    assert paths.count('/service/model') == paths.count('/service/templates') == 1
    service.close()


def test_independent_metadata_keys_overlap_and_failed_initialization_retries(monkeypatch):
    session = FixtureSession.service()
    session.routes['GET', '/service/templates'] = fixture_bytes('templates.xml')
    service = Service(SERVICE_ROOT, session=session)
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = service._read_template_snapshot
    def snapshot(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
            raise ValueError('temporary discovery failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(service, '_read_template_snapshot', snapshot)
    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(lambda: service.templates)
        try:
            assert entered.wait(2)
            assert pool.submit(lambda: service.release).result(2)
        finally:
            release.set()
        with pytest.raises(ValueError, match='temporary'):
            future.result(2)
    assert service.templates and len(calls) == 2
    service.close()


@pytest.mark.parametrize('operation', ['flush', 'close'])
def test_metadata_lifecycle_waits_for_initializer(monkeypatch, operation):
    session = FixtureSession.service()
    session.routes['GET', '/service/templates'] = fixture_bytes('templates.xml')
    service = Service(SERVICE_ROOT, session=session)
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    original = service._read_model_xml
    def read():
        entered.set()
        assert release.wait(3)
        return original()
    monkeypatch.setattr(service, '_read_model_xml', read)
    def finish():
        getattr(service, operation)()
        completed.set()
    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(lambda: service.model)
        assert entered.wait(2)
        closer = pool.submit(finish)
        try:
            assert not completed.wait(.05)
        finally:
            release.set()
        assert future.result(2).name == 'testmodel'
        closer.result(2)
    assert completed.is_set()
    if operation == 'flush':
        assert service._model is None and service._model_xml is None
    service.close()


def registry_client(size=1):
    session = FixtureSession.service()
    mines = json.loads(fixture_bytes('registry.json'))
    key = 'instances' if 'instances' in mines else 'mines'
    first = mines[key][0]
    mines[key].append(dict(first, name='AnotherMine'))
    session.routes['GET', '/service/instances'] = json.dumps(mines).encode()
    return Registry('https://registry.example', session=session, max_cached_services=size), session


def test_registry_leases_defer_eviction_clear_and_transport_close():
    registry, session = registry_client()
    closed = []
    original = registry._opener.close
    def close():
        closed.append(True)
        original()
    registry._opener.close = close
    with registry.lease('OfflineMine') as first:
        with registry.lease('offlinemine') as same:
            assert same is first
            with registry.lease('AnotherMine') as second:
                assert not first._closed
                assert registry.service_cache_metrics()['cache_size'] == 1
                registry.close()
                assert not first._closed and not second._closed and not closed
            assert second._closed and not first._closed and not closed
        assert not first._closed and not closed
    assert first._closed and closed == [True]
    assert session.close_calls == 0


def test_registry_singleflight_one_client_per_key_and_retry(monkeypatch):
    registry, _ = registry_client(2)
    calls = []
    original = registry._new_service
    def build(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise ValueError('initialization failed')
        return original(*args, **kwargs)
    monkeypatch.setattr(registry, '_new_service', build)
    with pytest.raises(ValueError, match='initialization'):
        registry['offlinemine']
    with ThreadPoolExecutor(8) as pool:
        clients = list(pool.map(lambda _: registry['offlinemine'], range(24)))
    assert all(client is clients[0] for client in clients) and len(calls) == 2
    registry.close()


def test_registry_independent_mines_initialize_concurrently(monkeypatch):
    registry, _ = registry_client(2)
    barrier = threading.Barrier(2)
    original = registry._new_service
    def build(*args, **kwargs):
        barrier.wait(2)
        return original(*args, **kwargs)
    monkeypatch.setattr(registry, '_new_service', build)
    with ThreadPoolExecutor(2) as pool:
        clients = list(pool.map(registry.__getitem__, ['offlinemine', 'anothermine']))
    assert clients[0] is not clients[1]
    registry.close()


@pytest.mark.parametrize('action', ['clear_cache', 'close'])
def test_registry_inflight_construction_does_not_repopulate_after_clear(monkeypatch, action):
    registry, _ = registry_client()
    entered, release = threading.Event(), threading.Event()
    original = registry._new_service
    candidates = []
    def build(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        result = original(*args, **kwargs)
        candidates.append(result)
        return result
    monkeypatch.setattr(registry, '_new_service', build)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(registry.__getitem__, 'offlinemine')
        try:
            assert entered.wait(2)
            getattr(registry, action)()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match='during client initialization'):
            future.result(2)
    assert candidates[0]._closed
    assert registry.service_cache_metrics()['cache_size'] == 0
    registry.close()


def test_metadata_state_allocation_can_run_service_finalizers(tmp_path):
    import subprocess
    import sys

    code = '''
from intermine314.service import metadata
from intermine314.service.service import Service
from tests.fixtures.compatibility import FixtureSession, SERVICE_ROOT
old = Service(SERVICE_ROOT, session=FixtureSession.service())
original = metadata.MetadataState
class Allocation(original):
    def __init__(self):
        # Models GC invoking a finalizer while constructing a new state.
        old.__del__()
        super().__init__()
metadata.MetadataState = Allocation
new = Service(SERVICE_ROOT, session=FixtureSession.service())
assert old._closed
new.close()
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
