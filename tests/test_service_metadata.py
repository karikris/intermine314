"""Offline metadata requests through the shared managed HTTP opener."""

import json
import os
import subprocess
import sys
from urllib.parse import parse_qs
from xml.parsers.expat import ExpatError

import pytest

from intermine314.service.errors import ServiceError, WebserviceError
from intermine314.service.service import Service
from intermine314.service.session import InterMineURLOpener, _ResponseStreamAdapter
from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession, fixture_bytes


@pytest.mark.parametrize('factory', ['native_service_factory', 'legacy_service_factory'])
def test_metadata_uses_configured_opener_and_preserves_search_facets(request, factory, offline_session_factory):
    payload = {'error': None, 'results': [{'name': 'Müller 🧬'}], 'facets': {'Organism': {'人': 2}}}
    session = offline_session_factory(routes={
        ('POST', '/service/search'): json.dumps(payload).encode(),
        ('GET', '/service/check/enrichment'): b'/list/enrichment\n',
        ('GET', '/service/metadata-xml'): '<metadata name="Müller"/>'.encode(),
    })
    service = request.getfixturevalue(factory)(
        session=session, token='private-token', request_timeout=13,
        proxy_url='socks5h://127.0.0.1:9050', tor=True,
        verify_tls='/custom/ca.pem', user_agent='metadata-client',
    )
    opener = service.opener
    assert isinstance(opener, InterMineURLOpener)
    assert service.search('Müller 🧬 & +', Organism=['人', 'H. sapiens'], Type='Gène') == (
        payload['results'], payload['facets'])
    assert service.widgets == {'age_groups': json.loads(fixture_bytes('widgets.json'))['widgets'][0]}
    assert service.release == fixture_bytes('version-release.txt').decode().strip()
    assert service.resolve_service_path('enrichment') == b'/list/enrichment\n'
    assert service._get_xml('/metadata-xml').documentElement.getAttribute('name') == 'Müller'
    assert service.opener is opener and opener._session is session
    assert opener.proxy_url == 'socks5h://127.0.0.1:9050' and opener.tor_mode is True
    assert [call.path for call in session.requests] == [
        '/service/version/ws', '/service/search', '/service/widgets',
        '/service/version/release', '/service/check/enrichment', '/service/metadata-xml',
    ]
    search = session.requests[1]
    assert search.method == 'POST'
    assert parse_qs(search.data.decode()) == {
        'q': ['Müller 🧬 & +'], 'facet_Organism': ['人', 'H. sapiens'], 'facet_Type': ['Gène'],
    }
    assert search.headers['Content-Type'] == 'application/x-www-form-urlencoded; charset=utf-8'
    assert [call.headers.get('Accept') for call in session.requests] == [
        None, 'application/json', 'application/json', None, None, 'application/xml',
    ]
    for call in session.requests:
        assert call.headers['Authorization'] == opener.auth_header
        assert call.headers['User-Agent'] == 'metadata-client'
        assert call.options == {'stream': True, 'timeout': (10.0, 13.0), 'verify': '/custom/ca.pem'}
    assert all(response.closed and response.close_calls == 1 for response in session.responses)
    service.close()
    assert session.close_calls == 0


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('release', [b'  release-\xc3\xa9\n', b' \n'])
@pytest.mark.parametrize('widgets', [b'{"error":null,"widgets":[]}', fixture_bytes('widgets.json')])
def test_metadata_caches_even_empty_values(native_service_factory, offline_session_factory, profile, release, widgets):
    session = offline_session_factory(routes={
        ('GET', '/service/version/release'): release,
        ('GET', '/service/widgets'): widgets,
    })
    service = native_service_factory(session=session, compatibility=profile)
    assert service.version == service.version == 8
    assert service.release == service.release == release.decode().strip()
    first = service.widgets
    assert service.widgets is first
    assert first == {widget['name']: widget for widget in json.loads(widgets)['widgets']}
    assert [call.path for call in session.requests] == [
        '/service/version/ws', '/service/version/release', '/service/widgets',
    ]
    assert all(response.close_calls == 1 for response in session.responses)


@pytest.mark.parametrize('payload,error', [
    (b'{"error":"search unavailable"}', ServiceError),
    (b'{"error":""}', ServiceError),
    (b'{"error":false}', ServiceError),
    (b'{broken', json.JSONDecodeError),
    (b'\xff', UnicodeDecodeError),
    (b'{"results":[],"facets":{}}', KeyError),
    (b'{"error":null,"results":[]}', KeyError),
])
def test_search_errors_are_honest_and_close_response(native_service_factory, offline_session_factory, payload, error):
    session = offline_session_factory(routes={('POST', '/service/search'): payload})
    service = native_service_factory(session=session)
    with pytest.raises(error) as raised:
        service.search('test')
    if error is ServiceError:
        assert raised.value.message == json.loads(payload)['error']
    assert session.responses[-1].close_calls == 1


@pytest.mark.parametrize('operation,path,method', [
    ('search', '/service/search', 'POST'),
    ('widgets', '/service/widgets', 'GET'),
    ('release', '/service/version/release', 'GET'),
    ('resolve', '/service/check/enrichment', 'GET'),
    ('xml', '/service/metadata-xml', 'GET'),
    ('version', '/service/version/ws', 'GET'),
])
@pytest.mark.parametrize('failure', ['http', 'read', 'interrupt'])
def test_metadata_closes_owned_response_on_failures(
    native_service_factory, offline_session_factory, monkeypatch, operation, path, method, failure,
):
    session = offline_session_factory()
    service = native_service_factory(session=session)
    session.routes[(method, path)] = (400, b'{"error":"failed"}') if failure == 'http' else b'payload'
    if operation == 'version':
        service._version = None
    if failure != 'http':
        def fail_read(*args, **kwargs):
            raise KeyboardInterrupt() if failure == 'interrupt' else OSError('read failed')
        monkeypatch.setattr(_ResponseStreamAdapter, 'read', fail_read)
    expected = {'http': WebserviceError, 'read': OSError, 'interrupt': KeyboardInterrupt}[failure]
    with pytest.raises(expected):
        if operation == 'search':
            service.search('test')
        elif operation == 'resolve':
            service.resolve_service_path('enrichment')
        elif operation == 'xml':
            service._get_xml('/metadata-xml')
        else:
            getattr(service, operation)
    assert session.responses[-1].closed and session.responses[-1].close_calls == 1
    assert session.close_calls == 0


@pytest.mark.parametrize('attribute,payload,error,valid,expected', [
    ('widgets', b'{"error":null,"widgets":[{}]}', KeyError, b'{"error":null,"widgets":[]}', {}),
    ('widgets', b'{broken', json.JSONDecodeError, b'{"error":null,"widgets":[]}', {}),
    ('release', b'\xff', UnicodeDecodeError, b'fixed\n', 'fixed'),
])
def test_failed_metadata_is_not_cached(native_service_factory, offline_session_factory, attribute, payload, error, valid, expected):
    path = '/service/widgets' if attribute == 'widgets' else '/service/version/release'
    session = offline_session_factory(routes={('GET', path): payload})
    service = native_service_factory(session=session)
    with pytest.raises(error):
        getattr(service, attribute)
    session.routes[('GET', path)] = valid
    assert getattr(service, attribute) == expected
    assert getattr(service, attribute) == expected
    assert [call.path for call in session.requests].count(path) == 2
    assert all(response.close_calls == 1 for response in session.responses)


def test_xml_parse_failure_closes_response(native_service_factory, offline_session_factory):
    session = offline_session_factory(routes={('GET', '/service/metadata-xml'): b'<broken>'})
    service = native_service_factory(session=session)
    with pytest.raises(ExpatError):
        service._get_xml('/metadata-xml')
    assert session.responses[-1].close_calls == 1


def test_version_parse_error_contract_and_response_closure():
    session = FixtureSession.service()
    session.routes[('GET', '/service/version/ws')] = b'not a version'
    with pytest.raises(ServiceError, match='Could not parse a valid webservice version'):
        Service(SERVICE_ROOT, session=session)
    assert session.responses[-1].close_calls == 1
    assert session.close_calls == 0


def test_metadata_facade_shares_methods_and_has_no_analytics_imports():
    script = '''
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'intermine'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
from intermine314.service.service import Service as Native
from intermine314.webservice import Service as Legacy
from tests.fixtures.compatibility import FixtureSession, SERVICE_ROOT
for name in ('search', 'widgets', 'release', 'resolve_service_path', 'version'):
    assert getattr(Native, name) is getattr(Legacy, name)
session = FixtureSession.service()
session.routes[('POST', '/service/search')] = b'{"error":null,"results":[],"facets":{}}'
session.routes[('GET', '/service/check/probe')] = b'/probe'
with Legacy(SERVICE_ROOT, session=session) as service:
    assert service.search('unicode 人') == ([], {})
    assert 'age_groups' in service.widgets
    assert isinstance(service.release, str)
    assert service.resolve_service_path('probe') == b'/probe'
assert 'intermine314.query.builder' not in sys.modules
assert all(response.close_calls == 1 for response in session.responses)
'''
    result = subprocess.run([sys.executable, '-c', script], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
