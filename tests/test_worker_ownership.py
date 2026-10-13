import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

import pytest
import requests

from intermine314.export.runtime import ResourceRuntime, get_runtime, in_http_worker
from intermine314.service.transport import borrowed_session_lock, build_session


def test_executor_reuse_context_and_close():
    marker = ContextVar('test_marker', default=None)
    with ResourceRuntime('cpu2') as runtime:
        marker.set('first')
        with runtime.executor(1) as job:
            first = job.submit(lambda: (threading.current_thread(), marker.get(), get_runtime(), in_http_worker())).result(5)
        marker.set('second')
        with runtime.executor(1) as job:
            second = job.submit(lambda: (threading.current_thread(), marker.get())).result(5)
        assert first[0] is second[0]
        assert first[1:] == ('first', runtime, True)
        assert second[1] == 'second'
        assert not job.futures
    assert not first[0].is_alive()


def test_job_cancellation_does_not_cancel_another_job():
    started, release = threading.Event(), threading.Event()
    with ResourceRuntime('cpu2') as runtime:
        with runtime.executor(1) as other:
            def running():
                started.set()
                assert release.wait(5)
                return 7

            future = other.submit(running)
            try:
                assert started.wait(5)
                with runtime.executor(1) as cancelled:
                    queued = cancelled.submit(lambda: pytest.fail('Cancelled job executed'))
                assert queued.cancelled()
                assert not future.cancelled()
            finally:
                release.set()
            assert future.result(5) == 7


def test_workers_have_private_reused_sessions_and_current_configuration(monkeypatch):
    gate = threading.Barrier(2)
    seen = []

    def request(self, *args, **kwargs):
        seen.append((self, dict(self.headers)))
        if len(seen) <= 2:
            gate.wait(timeout=5)
        return requests.Response()

    monkeypatch.setattr(requests.Session, 'request', request)
    session = build_session(proxy_url=None)
    try:
        with ResourceRuntime('cpu2') as runtime:
            with runtime.executor(2) as job:
                futures = [job.submit(session.request, 'GET', 'https://mine.test') for _ in range(2)]
                for future in futures:
                    future.result(5)
            assert len({id(item[0]) for item in seen}) == 2
            assert all(item[0] is not session for item in seen)
            assert all(item[0]._read_adapter is not session._read_adapter for item in seen)
            session.headers['X-Updated'] = 'after-first-job'
            with runtime.executor(2) as job:
                job.submit(session.request, 'GET', 'https://mine.test').result(5)
            assert seen[-1][1]['X-Updated'] == 'after-first-job'
    finally:
        session.close()
    assert all(item[0]._transport_closed for item in seen)
    assert not session._worker_sessions


@pytest.mark.parametrize('configuration', ['cookies', 'auth', 'hooks', 'adapter'])
def test_sensitive_sessions_keep_the_original_owner(monkeypatch, configuration):
    seen = []

    def request(self, *args, **kwargs):
        seen.append(self)
        return requests.Response()

    monkeypatch.setattr(requests.Session, 'request', request)
    with build_session(proxy_url=None) as session:
        if configuration == 'cookies':
            session.cookies.set('session', 'opaque')
        elif configuration == 'auth':
            session.auth = ('name', 'password')
        elif configuration == 'hooks':
            session.hooks['response'].append(lambda response: response)
        else:
            session.mount('https://special.test', requests.adapters.HTTPAdapter())
        with ResourceRuntime('cpu2') as runtime, runtime.executor(2) as job:
            futures = [job.submit(session.request, 'GET', 'https://mine.test') for _ in range(4)]
            for future in futures:
                future.result(5)
        assert seen == [session] * 4
        assert not session._worker_sessions


def test_new_cookie_moves_subsequent_requests_to_single_owner(monkeypatch):
    seen = []

    def request(self, *args, **kwargs):
        seen.append(self)
        self.cookies.set('session', 'opaque')
        return requests.Response()

    monkeypatch.setattr(requests.Session, 'request', request)
    with build_session(proxy_url=None) as session:
        with ResourceRuntime('cpu2') as runtime, runtime.executor(1) as job:
            job.submit(session.request, 'GET', 'https://mine.test').result(5)
            job.submit(session.request, 'GET', 'https://mine.test').result(5)
        assert seen[0] is not session and seen[1] is session
        assert session.cookies.get('session') == 'opaque'


def test_borrowed_session_lock_is_shared_without_mutation():
    with requests.Session() as session:
        before = dict(vars(session))
        with ThreadPoolExecutor(4) as pool:
            states = list(pool.map(lambda _: borrowed_session_lock(session), range(20)))
        assert all(state is states[0] for state in states)
        assert vars(session) == before
