"""Template discovery through actual XML responses and the managed opener."""

import os
import subprocess
import sys
from xml.parsers.expat import ExpatError

import pytest

from intermine314.query import QueryParseError, Template
from intermine314.service.errors import ServiceError, WebserviceError
from intermine314.service.session import _ResponseStreamAdapter
from tests.fixtures.compatibility import fixture_bytes


@pytest.fixture(params=['native_service_factory', 'legacy_service_factory'])
def template_client(request, offline_session_factory):
    session = offline_session_factory(routes={
        ('GET', '/service/templates'): fixture_bytes('templates.xml'),
        ('GET', '/service/alltemplates'): fixture_bytes('all-templates.xml'),
    })
    service = request.getfixturevalue(request.param)(
        session=session, token='private-token', request_timeout=13,
        proxy_url='socks5h://127.0.0.1:9050', tor=True,
        verify_tls='/custom/ca.pem', user_agent='template-client',
    )
    return service, session


def paths(session):
    return [call.path for call in session.requests]


def assert_closed(session):
    assert all(response.closed and response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 0


def test_global_discovery_is_lazy_and_getter_caches_bound_profile(template_client):
    service, session = template_client
    assert paths(session) == ['/service/version/ws']
    templates = service.templates
    assert isinstance(templates['employeeByName'], str)
    assert service.templates is templates
    assert paths(session) == ['/service/version/ws', '/service/templates']
    template = service.get_template('employeeByName')
    assert isinstance(template, Template) and template.service is service
    assert template.model is service.model and template.compatibility == service.compatibility
    assert template.name == 'employeeByName'
    assert template.get_constraint('A').editable is True
    assert template.get_constraint('A').value == ''
    assert templates['employeeByName'] is template
    assert service.get_template('employeeByName') is template
    assert isinstance(templates['ManagerLookup'], str)
    assert paths(session).count('/service/templates') == paths(session).count('/service/model') == 1
    assert_closed(session)


@pytest.mark.parametrize('names_first', [True, False])
def test_user_names_and_objects_share_snapshot_and_nested_cache(template_client, names_first):
    service, session = template_client
    first = service.all_templates_names if names_first else service.all_templates
    assert '/service/model' not in paths(session)
    names, templates = service.all_templates_names, service.all_templates
    assert first is (names if names_first else templates)
    assert names == {'alice': ['shared', 'gene-é'], 'bob': ['shared']}
    assert all(isinstance(xml, str) for user in templates.values() for xml in user.values())
    alice = service.get_template_by_user('shared', 'alice')
    bob = service.get_template_by_user('shared', 'bob')
    assert alice is service.get_template_by_user('shared', 'alice')
    assert bob is service.get_template_by_user('shared', 'bob') and alice is not bob
    assert set(templates) == {'alice', 'bob'}
    assert templates['alice']['shared'] is alice and templates['bob']['shared'] is bob
    assert (alice.user_name, bob.user_name) == ('alice', 'bob')
    assert (alice.get_constraint('A').value, bob.get_constraint('A').value) == ('Alice', 'Bob')
    for template in (alice, bob):
        assert template.service is service and template.model is service.model
        assert template.compatibility == service.compatibility
    assert service.all_templates_names is names and names['bob'] == ['shared']
    assert paths(session).count('/service/alltemplates') == paths(session).count('/service/model') == 1
    assert_closed(session)


def test_unknown_names_and_users_have_exact_errors_without_model_get(template_client):
    service, session = template_client
    calls = [
        (lambda: service.get_template('missing'), "There is no template called 'missing' at this service"),
        (lambda: service.get_template_by_user('shared', 'missing'), "There is no user called 'missing'"),
        (lambda: service.get_template_by_user('missing', 'alice'),
         "There is no template called 'missing' at this service belonging to 'alice'"),
    ]
    for call, message in calls:
        with pytest.raises(ServiceError) as error:
            call()
        assert error.value.message == message
    assert '/service/model' not in paths(session)
    assert_closed(session)


def test_name_snapshot_does_not_depend_on_mutable_parsed_cache(template_client):
    service, session = template_client
    service.get_template_by_user('shared', 'alice')
    del service.all_templates['bob']
    service.all_templates['alice']['local'] = 'caller-added XML'
    assert service.all_templates_names == {'alice': ['shared', 'gene-é'], 'bob': ['shared']}
    assert paths(session).count('/service/alltemplates') == 1
    assert_closed(session)


@pytest.mark.parametrize('attribute,path', [
    ('templates', '/service/templates'), ('all_templates', '/service/alltemplates'),
    ('all_templates_names', '/service/alltemplates'),
])
def test_duplicate_name_rejected_without_partial_cache(template_client, attribute, path):
    service, session = template_client
    session.routes[('GET', path)] = b'<templates><template name="same" userName="alice"/><template name="same" userName="alice"/></templates>'
    with pytest.raises(ServiceError) as error:
        getattr(service, attribute)
    assert error.value.message == 'Two templates with same name: same'
    session.routes[('GET', path)] = b'<templates/>'
    assert getattr(service, attribute) == {}
    assert paths(session).count(path) == 2
    assert_closed(session)


@pytest.mark.parametrize('attribute,path', [
    ('templates', '/service/templates'), ('all_templates_names', '/service/alltemplates'),
])
@pytest.mark.parametrize('failure', ['xml', 'http', 'read', 'interrupt'])
def test_discovery_failure_closes_owned_response_and_can_retry(template_client, monkeypatch, attribute, path, failure):
    service, session = template_client
    session.routes[('GET', path)] = (500, b'failed') if failure == 'http' else b'<broken>'
    error = {'xml': ExpatError, 'http': WebserviceError, 'read': OSError, 'interrupt': KeyboardInterrupt}[failure]
    with monkeypatch.context() as patch:
        if failure in {'read', 'interrupt'}:
            def fail_read(*args, **kwargs):
                raise error('read interrupted')
            patch.setattr(_ResponseStreamAdapter, 'read', fail_read)
        with pytest.raises(error):
            getattr(service, attribute)
    assert_closed(session)
    session.routes[('GET', path)] = b'<templates/>'
    assert getattr(service, attribute) == {}
    assert paths(session).count(path) == 2
    assert_closed(session)


@pytest.mark.parametrize('by_user', [False, True])
def test_invalid_query_is_only_parsed_on_access_and_raw_xml_is_retained(template_client, by_user):
    service, session = template_client
    path = '/service/alltemplates' if by_user else '/service/templates'
    session.routes[('GET', path)] = b'<templates><template name="bad" userName="alice"/></templates>'
    cache = service.all_templates['alice'] if by_user else service.templates
    raw = cache['bad']
    assert isinstance(raw, str) and '/service/model' not in paths(session)
    for _ in range(2):
        with pytest.raises(QueryParseError):
            if by_user:
                service.get_template_by_user('bad', 'alice')
            else:
                service.get_template('bad')
        assert cache['bad'] == raw
    assert paths(session).count(path) == paths(session).count('/service/model') == 1
    assert_closed(session)


def test_configured_transport_and_borrowed_session_are_preserved(template_client):
    service, session = template_client
    opener = service.opener
    service.get_template('employeeByName')
    service.get_template_by_user('shared', 'bob')
    for call in session.requests:
        assert call.method == 'GET' and call.data is None
        assert call.headers['Authorization'] == opener.auth_header
        assert call.headers['User-Agent'] == 'template-client'
        assert call.options == {'stream': True, 'timeout': (10.0, 13.0), 'verify': '/custom/ca.pem'}
        if call.path in {'/service/templates', '/service/alltemplates'}:
            assert call.headers['Accept'] == 'application/xml'
    assert service.opener is opener and opener._session is session
    assert opener.tor_mode is True and opener.proxy_url == 'socks5h://127.0.0.1:9050'
    service.close()
    assert_closed(session)


def test_flush_invalidates_real_raw_names_and_parsed_caches(template_client):
    service, session = template_client
    old_global = service.get_template('employeeByName')
    old_user = service.get_template_by_user('shared', 'alice')
    old_names = service.all_templates_names
    opener = service.opener
    assert service.flush() is None
    assert service._list_manager is None and service.opener is opener
    assert service._templates_raw is None and service._all_templates_raw is None
    assert service.get_template('employeeByName') is not old_global
    assert service.get_template_by_user('shared', 'alice') is not old_user
    assert service.all_templates_names == old_names and service.all_templates_names is not old_names
    assert paths(session).count('/service/templates') == paths(session).count('/service/alltemplates') == 2
    assert paths(session).count('/service/model') == 2
    assert_closed(session)


def test_discovery_has_no_invented_version_gate(native_service_factory, offline_session_factory):
    session = offline_session_factory(version=1, routes={
        ('GET', '/service/templates'): b'<templates/>',
        ('GET', '/service/alltemplates'): b'<templates/>',
    })
    service = native_service_factory(session=session)
    assert service.templates == service.all_templates == service.all_templates_names == {}
    assert paths(session).count('/service/alltemplates') == 1
    assert_closed(session)


def test_discovery_imports_stay_lazy_and_getters_need_no_analytics():
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
from tests.fixtures.compatibility import FixtureSession, SERVICE_ROOT, fixture_bytes
for name in ('templates', 'all_templates', 'all_templates_names', 'get_template', 'get_template_by_user'):
    assert getattr(Native, name) is getattr(Legacy, name)
for cls in (Native, Legacy):
    session = FixtureSession.service()
    session.routes[('GET', '/service/templates')] = fixture_bytes('templates.xml')
    session.routes[('GET', '/service/alltemplates')] = fixture_bytes('all-templates.xml')
    with cls(SERVICE_ROOT, session=session) as service:
        assert 'employeeByName' in service.templates
        assert service.all_templates_names['bob'] == ['shared']
        if cls is Native:
            assert 'intermine314.query.builder' not in sys.modules
        assert service.get_template('employeeByName').compatibility == service.compatibility
        assert service.get_template_by_user('shared', 'bob').user_name == 'bob'
    assert session.close_calls == 0
    assert all(response.close_calls == 1 for response in session.responses)
'''
    result = subprocess.run([sys.executable, '-c', script], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
