import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from intermine314.service.admission import HostAdmission, origin, retry_after_seconds
from intermine314.service.session import _ResponseStreamAdapter


def test_origin_normalizes_without_credentials_or_paths():
    assert origin('HTTPS://user:password@EXAMPLE.org/service?token=secret') == origin('https://example.org:443/another')
    assert origin('https://example.org') != origin('http://example.org')


def test_host_permit_is_held_until_response_close():
    gate = HostAdmission()
    release, _ = gate.acquire('https://mine.test/a', host_limit=1)
    response = SimpleNamespace(iter_lines=lambda **kwargs: iter([b'row']), close=lambda: None)
    stream = _ResponseStreamAdapter(response, release)
    entered = threading.Event()

    def another_export():
        done, _ = gate.acquire('https://mine.test/b', host_limit=1, timeout=5)
        try:
            entered.set()
        finally:
            done()

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(another_export)
        try:
            assert next(stream) == b'row'
            assert not entered.wait(0.05)
        finally:
            stream.close()
        future.result(5)
    stream.close()
    assert entered.is_set() and not gate.active


def test_global_limit_applies_across_hosts_and_timeout_does_not_leak():
    gate = HostAdmission()
    done, _ = gate.acquire('https://first.test', host_limit=4, global_limit=1)
    try:
        with pytest.raises(TimeoutError):
            gate.acquire('https://second.test', host_limit=4, global_limit=1, timeout=0.01)
    finally:
        done()
    release, _ = gate.acquire('https://second.test', host_limit=4, global_limit=1, timeout=1)
    release()
    assert not gate.active and all(not state['waiting'] for state in gate.hosts.values())


def test_throttling_reduces_admission_and_recovers_to_the_configured_ceiling():
    gate = HostAdmission()
    done, observe = gate.acquire('https://mine.test', host_limit=4)
    observe(SimpleNamespace(status_code=429, headers={'Retry-After': '0'}, raw=None))
    done()
    state = gate.hosts[origin('https://mine.test')]
    assert state['adaptive'] == 2
    for _ in range(24):
        done, observe = gate.acquire('https://mine.test', host_limit=4)
        observe(SimpleNamespace(status_code=200, headers={}, raw=None))
        done()
    assert state['adaptive'] == 4 and not gate.active


@pytest.mark.parametrize('value', ['nan', 'inf', '-2', 'invalid', None])
def test_invalid_retry_after_does_not_poison_host_state(value):
    assert retry_after_seconds(value) == 0


def test_close_failure_still_releases_host_permit():
    gate = HostAdmission()
    release, _ = gate.acquire('https://mine.test', host_limit=1)

    def failed_close():
        raise OSError('close failed')

    stream = _ResponseStreamAdapter(SimpleNamespace(iter_lines=lambda **kwargs: iter(()), close=failed_close), release)
    with pytest.raises(OSError):
        stream.close()
    assert not gate.active


def test_discarded_direct_result_facade_releases_without_cyclic_gc():
    from tests.test_result_formats import make_iterator

    iterator, session, _ = make_iterator()
    next(iterator)
    response = session.responses[-1]
    del iterator
    assert response.closed
