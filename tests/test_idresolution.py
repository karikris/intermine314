"""Identifier-resolution wire and lifecycle contracts, using a fake clock."""

import gc
import importlib
import json
import os
import subprocess
import sys
import weakref

import pytest

from intermine314.service.errors import ServiceError, WebserviceError
from intermine314.service.service import Service
from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession, fixture_bytes


def job_session(version=10):
    session = FixtureSession.service(version=version)
    session.routes.update({
        ('POST', '/service/ids'): fixture_bytes('job-created.json'),
        ('GET', '/service/ids/offline-job-1/status'): fixture_bytes('job-pending.json'),
        ('GET', '/service/ids/offline-job-1/result'): fixture_bytes('job-results.json'),
        ('DELETE', '/service/ids/offline-job-1'): b'{"error":null}',
    })
    return session


@pytest.fixture
def fake_clock(monkeypatch):
    module = importlib.import_module('intermine314.idresolution')
    sleeps = []
    monkeypatch.setattr(module.time, 'sleep', sleeps.append)
    return sleeps


@pytest.mark.parametrize('factory', ['native_service_factory', 'legacy_service_factory'])
@pytest.mark.parametrize('options', [{}, {'extra': 'Müller 人', 'case_sensitive': True, 'wildcards': True}])
def test_submission_wire_configuration_and_job(request, factory, options):
    from intermine314.idresolution import Job

    session = job_session()
    service = request.getfixturevalue(factory)(
        session=session, token='resolution-token', request_timeout=13,
        proxy_url='socks5h://127.0.0.1:9050', tor=True,
        verify_tls='/custom/ca.pem', user_agent='resolution-client',
    )
    job = service.resolve_ids('Employee', iter(['00123', 'Müller &+*']), **options)
    assert isinstance(job, Job) and job.uid == 'offline-job-1'
    assert job.status is None and job.backoff == .05 and job.decay == 1.25 and job.max_backoff == 60
    assert job.service.opener is service.opener
    call = session.requests[-1]
    assert call.method == 'POST' and call.path == '/service/ids'
    assert json.loads(call.data) == {
        'type': 'Employee', 'identifiers': ['00123', 'Müller &+*'],
        'extra': options.get('extra', ''), 'caseSensitive': options.get('case_sensitive', False),
        'wildCards': options.get('wildcards', False),
    }
    assert call.headers['Content-Type'] == 'application/json; charset=utf-8'
    assert call.headers['Authorization'] == service.opener.auth_header
    assert call.headers['User-Agent'] == 'resolution-client'
    assert call.options == {'stream': True, 'timeout': (10.0, 13.0), 'verify': '/custom/ca.pem'}
    assert all(response.close_calls == 1 for response in session.responses)
    service.close()
    assert session.close_calls == 0


@pytest.mark.parametrize('version,kind,ids,message', [
    (9, 'Employee', ['x'], 'This feature requires API version 10+'),
    (9, '', [], 'This feature requires API version 10+'),
    (10, '', ['x'], 'No data-type supplied'),
    (10, None, ['x'], 'No data-type supplied'),
    (10, 'Employee', [], 'No identifiers supplied'),
    (10, 'Employee', None, 'No identifiers supplied'),
])
def test_submission_gates_before_post(native_service_factory, version, kind, ids, message):
    session = job_session(version)
    service = native_service_factory(session=session)
    before = len(session.requests)
    with pytest.raises(ServiceError) as failure:
        service.resolve_ids(kind, ids)
    assert failure.value.args == (message,) and len(session.requests) == before


@pytest.mark.parametrize('terminal', ['SUCCESS', 'ERROR'])
def test_poll_waits_before_fetch_and_terminal_polls_are_inert(native_service_factory, fake_clock, terminal):
    session = job_session()
    service = native_service_factory(session=session)
    job = service.resolve_ids('Employee', ['00123'])
    original_request = session.request

    def capture(method, url, **kwargs):
        if url.endswith('/status'):
            assert len(fake_clock) == sum(call.path.endswith('/status') for call in session.requests) + 1
        return original_request(method, url, **kwargs)

    session.request = capture
    assert job.poll() is False and job.status == 'PENDING'
    session.routes[('GET', '/service/ids/offline-job-1/status')] = b'{"error":null,"status":"RUNNING"}'
    assert job.poll() is False and job.status == 'RUNNING'
    session.routes[('GET', '/service/ids/offline-job-1/status')] = (
        fixture_bytes('job-success.json') if terminal == 'SUCCESS'
        else b'{"error":null,"status":"ERROR"}'
    )
    assert job.poll() is True and job.status == terminal
    assert fake_clock == [.05, .0625, .078125]
    before = len(session.requests)
    assert job.poll() is True and job.poll() is True
    assert len(session.requests) == before and len(fake_clock) == 3
    assert all(response.close_calls == 1 for response in session.responses)


def test_poll_caps_delay_without_limiting_attempts(native_service_factory, fake_clock):
    session = job_session()
    job = native_service_factory(session=session).resolve_ids('Employee', ['00123'])
    job.backoff = 50
    for _ in range(100):
        assert job.poll() is False
    assert fake_clock == [50] + [60] * 99 and job.backoff == 60
    assert sum(call.path.endswith('/status') for call in session.requests) == 100


def test_fetches_do_not_poll_or_change_cached_status_and_delete_returns_none(native_service_factory, fake_clock):
    session = job_session()
    service = native_service_factory(session=session)
    job = service.resolve_ids('Employee', ['00123'])
    assert job.fetch_status() == 'PENDING' and job.status is None
    results = job.fetch_results()
    assert results == json.loads(fixture_bytes('job-results.json'))['results']
    assert results['matches'][0]['matches'][0]['id'] == 9007199254740993
    assert job.delete() is None and job.uid == 'offline-job-1' and job.status is None
    assert [(call.method, call.path, call.data) for call in session.requests[2:]] == [
        ('GET', '/service/ids/offline-job-1/status', None),
        ('GET', '/service/ids/offline-job-1/result', None),
        ('DELETE', '/service/ids/offline-job-1', None),
    ]
    assert fake_clock == [] and all(response.close_calls == 1 for response in session.responses)


@pytest.mark.parametrize('operation,path,key', [
    ('submit', '/service/ids', 'uid'), ('status', '/service/ids/offline-job-1/status', 'status'),
    ('result', '/service/ids/offline-job-1/result', 'results'), ('delete', '/service/ids/offline-job-1', None),
])
@pytest.mark.parametrize('payload,exception,args', [
    (b'{"error":"server failure"}', Exception, ('server failure',)),
    (b'{"error":""}', Exception, ('',)),
    (b'{"error":false}', Exception, (False,)),
    (b'{}', KeyError, ('error',)),
    (b'broken json', json.JSONDecodeError, None),
    (b'\xff', UnicodeDecodeError, None),
])
def test_json_errors_and_owned_response_cleanup(native_service_factory, operation, path, key, payload, exception, args):
    session = job_session()
    service = native_service_factory(session=session)
    job = service.resolve_ids('Employee', ['00123'])
    method = {'submit': 'POST', 'delete': 'DELETE'}.get(operation, 'GET')
    session.routes[(method, path)] = payload
    invoke = {'submit': lambda: service.resolve_ids('Employee', ['x']), 'status': job.fetch_status,
              'result': job.fetch_results, 'delete': job.delete}[operation]
    with pytest.raises(exception) as failure:
        invoke()
    if args is not None:
        assert failure.value.args == args
    if operation == 'submit' and exception is Exception:
        assert type(failure.value) is ServiceError
    assert session.responses[-1].close_calls == 1 and session.close_calls == 0


@pytest.mark.parametrize('key', ['status', 'results'])
def test_get_json_missing_key_exact_error(native_service_factory, key):
    from intermine314.idresolution import get_json

    session = job_session()
    path = '/ids/offline-job-1/status'
    session.routes[('GET', '/service' + path)] = b'{"error":null}'
    service = native_service_factory(session=session)
    with pytest.raises(Exception) as failure:
        get_json(service, path, key)
    assert type(failure.value) is Exception
    assert str(failure.value) == key + ' not returned from ' + path
    assert session.responses[-1].close_calls == 1


def test_uid_validation_and_source_edge_exceptions(native_service_factory):
    from intermine314.idresolution import Job

    session = job_session()
    service = native_service_factory(session=session)
    with pytest.raises(Exception, match='^No uid found$'):
        Job(service, None)
    session.routes[('POST', '/service/ids')] = b'{"error":null,"uid":null}'
    with pytest.raises(Exception, match='^No uid found$'):
        service.resolve_ids('Employee', ['x'])
    session.routes[('POST', '/service/ids')] = b'{"error":null}'
    with pytest.raises(KeyError, match='uid'):
        service.resolve_ids('Employee', ['x'])
    assert Job(service, '').uid == ''
    with pytest.raises(TypeError):
        Job(service, 123).delete()
    assert all(response.close_calls == 1 for response in session.responses)


def test_job_does_not_keep_service_alive():
    from intermine314.idresolution import Job

    session = job_session()
    service = Service(SERVICE_ROOT, session=session)
    job = Job(service, 'offline-job-1')
    service_ref = weakref.ref(service)
    del service
    gc.collect()
    assert service_ref() is None and session.close_calls == 0
    with pytest.raises(ReferenceError):
        job.fetch_status()


@pytest.mark.parametrize('operation', ['submit', 'poll', 'result', 'delete'])
@pytest.mark.parametrize('failure', [OSError('read failure'), KeyboardInterrupt('interrupt')])
def test_read_failures_and_interrupts_close_owned_response(monkeypatch, native_service_factory, fake_clock, operation, failure):
    from intermine314.service.session import _ResponseStreamAdapter

    session = job_session()
    service = native_service_factory(session=session)
    job = service.resolve_ids('Employee', ['x'])

    def fail_read(self, size=-1):
        raise failure

    monkeypatch.setattr(_ResponseStreamAdapter, 'read', fail_read)
    invoke = {'submit': lambda: service.resolve_ids('Employee', ['x']), 'poll': job.poll,
              'result': job.fetch_results, 'delete': job.delete}[operation]
    with pytest.raises(type(failure)):
        invoke()
    assert session.responses[-1].close_calls == 1 and session.close_calls == 0
    assert job.status is None
    assert fake_clock == ([.05] if operation == 'poll' else [])
    assert job.backoff == (.0625 if operation == 'poll' else .05)


@pytest.mark.parametrize('operation,method,path', [
    ('submit', 'POST', '/service/ids'), ('status', 'GET', '/service/ids/offline-job-1/status'),
    ('result', 'GET', '/service/ids/offline-job-1/result'), ('delete', 'DELETE', '/service/ids/offline-job-1'),
])
def test_http_errors_use_shared_validation(native_service_factory, operation, method, path):
    session = job_session()
    service = native_service_factory(session=session)
    job = service.resolve_ids('Employee', ['x'])
    session.routes[(method, path)] = (400, b'{"error":"bad resolution request"}')
    invoke = {'submit': lambda: service.resolve_ids('Employee', ['x']), 'status': job.fetch_status,
              'result': job.fetch_results, 'delete': job.delete}[operation]
    with pytest.raises(WebserviceError) as failure:
        invoke()
    assert failure.value.args == ('There was a problem with our request', 400)
    assert failure.value.filename == 'Offline HTTP error'
    assert session.responses[-1].close_calls == 1 and session.close_calls == 0


@pytest.mark.parametrize('failure', [OSError('error body read failure'), KeyboardInterrupt('error body interrupt')])
@pytest.mark.parametrize('close_fails', [False, True])
def test_http_error_body_failures_close_owned_response(monkeypatch, native_service_factory, failure, close_fails):
    from intermine314.service.session import _ResponseBodyAdapter
    from tests.fixtures.compatibility import FixtureResponse

    session = job_session()
    service = native_service_factory(session=session)
    job = service.resolve_ids('Employee', ['x'])
    session.routes[('GET', '/service/ids/offline-job-1/status')] = (400, b'{"error":"bad request"}')

    def fail_read(self):
        raise failure

    original_close = FixtureResponse.close

    def fail_close(self):
        original_close(self)
        raise OSError('secondary close failure')

    if close_fails:
        monkeypatch.setattr(FixtureResponse, 'close', fail_close)
    monkeypatch.setattr(_ResponseBodyAdapter, 'read', fail_read)
    with pytest.raises(type(failure)) as caught:
        job.fetch_status()
    assert caught.value is failure
    assert session.responses[-1].close_calls == 1 and session.close_calls == 0


def test_import_and_execution_require_no_query_analytics_or_original_client():
    script = '''
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'matplotlib', 'plotly', 'intermine'} or fullname.startswith('intermine314.query'):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
import intermine314.webservice
assert 'intermine314.idresolution' not in sys.modules
from intermine314.idresolution import Job, get_json
from tests.fixtures.compatibility import SERVICE_ROOT
from tests.test_idresolution import job_session
service = intermine314.webservice.Service(SERVICE_ROOT, session=job_session())
job = service.resolve_ids('Employee', ['00123'])
assert isinstance(job, Job)
assert job.fetch_status() == 'PENDING'
assert job.fetch_results()['unresolved'] == ['unknown']
assert job.delete() is None
service.close()
'''
    completed = subprocess.run([sys.executable, '-c', script], env=os.environ.copy(), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
