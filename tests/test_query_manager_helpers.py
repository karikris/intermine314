"""Historical saved-query behavior on strict offline account routes."""

import importlib
import json
import os
import subprocess
import sys
from urllib.parse import parse_qs, quote, urlsplit
from xml.etree import ElementTree

import pytest

from intermine314.service.errors import WebserviceError
from intermine314.service.session import _ResponseStreamAdapter
from intermine314.webservice import Registry, Service
from tests.fixtures.compatibility import FixtureOpener, FixtureSession, fixture_bytes

TOKEN = 'secret+&人'
NAME = 'a/+&人'
XML = '<query name="a/+&amp;人" model="genomic" view="Gene.symbol"><constraint value="A+B&amp;人"/></query>'
NOTE = 'Note: name should contain no special symbol            and should be defined first\n'
OPTIONS = dict(request_timeout=13, proxy_url='socks5h://127.0.0.1:9050', tor=True,
               verify_tls='/custom/ca.pem', user_agent='saved-query-client')


def helper_session(queries=None, version=27):
    return FixtureSession({
        ('GET', '/service/instances'): fixture_bytes('registry-helpers.json'),
        ('GET', '/service/instances/OfflineMine'): fixture_bytes('registry-helpers.json'),
        ('GET', '/custom/service/user/queries'): fixture_bytes('saved-queries.json') if queries is None else queries,
        ('GET', '/custom/service/version'): str(version).encode(),
        ('GET', '/custom/service/version/ws'): b'30',
        ('GET', '/service/version/ws'): b'30',
        ('PUT', '/custom/service/user/queries'): b'',
        ('DELETE', '/custom/service/user/queries/' + quote(NAME, safe='')): b'',
    })


@pytest.fixture
def qm():
    module = importlib.import_module('intermine314.query_manager')
    yield module
    if module._state is not None:
        module._state.close()
    module._state = None
    for name in ('mine', 'token'):
        module.__dict__.pop(name, None)


def configure(qm, session, **options):
    assert qm.save_mine_and_token('OfflineMine', TOKEN, session=session, **options) is None


def assert_closed(session):
    assert all(response.close_calls == 1 for response in session.responses)


def test_original_positional_configuration_names_order_and_managed_transport(qm, capsys, caplog):
    session = helper_session()
    configure(qm, session, **OPTIONS)
    assert qm.mine == 'OfflineMine' and qm.token == TOKEN
    assert qm.get_all_query_names() == 'Zulu, a/+&人'
    assert [call.path for call in session.requests] == [
        '/service/instances', '/service/instances/OfflineMine',
        '/custom/service/version/ws', '/custom/service/user/queries', '/custom/service/user/queries']
    for call in session.requests:
        assert call.url.startswith('https://')
        assert call.options == {'stream': True, 'timeout': (10.0, 13.0), 'verify': '/custom/ca.pem'}
        assert call.headers['User-Agent'] == 'saved-query-client'
        parameters = parse_qs(urlsplit(call.url).query)
        assert parameters.get('token') == ([TOKEN] if '/user/queries' in call.path else None)
        assert 'Authorization' not in call.headers
    assert TOKEN not in capsys.readouterr().out + caplog.text
    assert_closed(session)
    assert session.close_calls == 0


@pytest.mark.parametrize('queries,result', [(b'{"queries":{}}', 'No saved queries'),
                                           (b'{"queries":{"Only":{}}}', 'Only')])
def test_empty_and_single_name_source_returns(qm, queries, result):
    session = helper_session(queries)
    configure(qm, session)
    assert qm.get_all_query_names() == result
    assert_closed(session)


@pytest.mark.parametrize('payload,result', [
    (b'<saved-queries></saved-queries>', 'No such query available'),
    (b'<?xml version="1.0"?><saved-queries />\n', 'No such query available'),
    (b'<saved-queries> \n </saved-queries>', 'No such query available'),
    ('<saved-queries><query name="人"/></saved-queries>'.encode(), '<saved-queries><query name="人"/></saved-queries>'),
    (b'raw server text', 'raw server text'),
    (b'', ''),
])
def test_get_query_raw_text_or_empty_xml_sentinel_and_encoded_filter(qm, payload, result):
    session = helper_session()
    configure(qm, session)
    session.routes[('GET', '/custom/service/user/queries')] = payload
    assert qm.get_query(NAME) == result
    assert parse_qs(urlsplit(session.requests[-1].url).query) == {
        'filter': [NAME], 'format': ['xml'], 'token': [TOKEN]}
    assert_closed(session)


def test_delete_existing_encodes_entire_name_once_and_closes_discarded_response(qm):
    session = helper_session()
    configure(qm, session)
    assert qm.delete_query(NAME) == NAME + ' is deleted'
    call = session.requests[-1]
    assert call.method == 'DELETE'
    assert call.path == '/custom/service/user/queries/a%2F%2B%26%E4%BA%BA'
    assert parse_qs(urlsplit(call.url).query) == {'token': [TOKEN]}
    assert_closed(session)


def test_delete_missing_does_not_mutate_account(qm):
    session = helper_session()
    configure(qm, session)
    assert qm.delete_query('absent') == 'No such query available'
    assert not any(call.method == 'DELETE' for call in session.requests)
    assert_closed(session)


@pytest.mark.parametrize('version,parameter', [(26, 'xml'), (27, 'query'), (30, 'query')])
def test_post_version_threshold_encoded_xml_and_token_rereads_names(qm, version, parameter):
    session = helper_session(version=version)
    configure(qm, session)
    assert qm.post_query(XML, overwrite=True) == NAME + ' is posted'
    calls = session.requests[4:]
    assert [(call.method, call.path) for call in calls] == [
        ('GET', '/custom/service/version'), ('GET', '/custom/service/user/queries'),
        ('PUT', '/custom/service/user/queries'), ('GET', '/custom/service/user/queries')]
    assert parse_qs(urlsplit(calls[0].url).query) == {'token': [TOKEN]}
    for call in calls[2:]:
        assert parse_qs(urlsplit(call.url).query) == {parameter: [XML], 'token': [TOKEN]}
        assert call.data is None
    assert_closed(session)


@pytest.mark.parametrize('answer,result,output,mutated', [
    ('y', NAME + ' is posted', 'The query name exists\n', True),
    ('n', None, 'The query name exists\nUse a query name other than ' + NAME + '\n', False),
    ('Y', None, 'The query name exists\nUse a query name other than ' + NAME + '\n', False),
])
def test_default_duplicate_prompt_exact_strings_and_source_answer_policy(qm, monkeypatch, capsys, answer, result, output, mutated):
    session = helper_session()
    configure(qm, session)
    prompts = []

    def input_response(prompt):
        prompts.append(prompt)
        return answer

    monkeypatch.setattr('builtins.input', input_response)
    assert qm.post_query(XML) == result
    assert prompts == ['Do you want to replace the old query? [y/n]']
    assert capsys.readouterr().out == output
    assert any(call.method == 'PUT' for call in session.requests) is mutated
    assert_closed(session)


@pytest.mark.parametrize('overwrite,result,output', [
    (True, NAME + ' is posted', ''),
    (False, None, 'Use a query name other than ' + NAME + '\n'),
])
def test_explicit_overwrite_bypasses_input(qm, monkeypatch, capsys, overwrite, result, output):
    session = helper_session()
    configure(qm, session)

    def forbidden_input(prompt):
        pytest.fail('Explicit overwrite must bypass input')

    monkeypatch.setattr('builtins.input', forbidden_input)
    assert qm.post_query(XML, overwrite=overwrite) == result
    assert capsys.readouterr().out == output
    assert_closed(session)


def test_post_new_query_no_prompt_and_original_incorrect_format_spacing(qm, monkeypatch, capsys):
    session = helper_session(b'{"queries":{}}')
    configure(qm, session)

    def forbidden_input(prompt):
        pytest.fail('New query must not prompt')

    monkeypatch.setattr('builtins.input', forbidden_input)
    assert qm.post_query(XML) == 'Incorrect format'
    assert capsys.readouterr().out == NOTE
    assert_closed(session)


@pytest.mark.parametrize('xml,error', [('broken', ElementTree.ParseError), ('<query/>', KeyError)])
def test_invalid_xml_or_missing_name_propagates_before_network(qm, xml, error):
    session = helper_session()
    configure(qm, session)
    before = len(session.requests)
    with pytest.raises(error):
        qm.post_query(xml)
    assert len(session.requests) == before


@pytest.mark.parametrize('phase,payload,error', [
    ('mine', b'{}', 'KeyError'), ('mine', b'{broken', 'JSONDecodeError'),
    ('token', b'{}', 'KeyError'), ('token', b'{broken', 'JSONDecodeError'),
    ('token', (403, b'secret+&\xe4\xba\xba'), 'WebserviceError'),
])
def test_configuration_errors_class_only_messages_and_invalid_state(qm, phase, payload, error, capsys, caplog):
    session = helper_session()
    path = '/service/instances/OfflineMine' if phase == 'mine' else '/custom/service/user/queries'
    session.routes[('GET', path)] = payload
    assert qm.save_mine_and_token('OfflineMine', TOKEN, session=session) == (
        'An exception of type ' + error + ' occurred. Check ' + phase)
    assert qm.mine == 'OfflineMine' and qm.token == TOKEN
    assert qm._state is None
    before = len(session.requests)
    with pytest.raises(RuntimeError, match='save_mine_and_token'):
        qm.get_all_query_names()
    assert len(session.requests) == before
    assert TOKEN not in capsys.readouterr().out + caplog.text
    assert_closed(session)
    assert session.close_calls == 0


def test_reconfiguration_closes_old_owned_state_and_failed_new_state_without_old_credentials(qm, monkeypatch):
    from intermine314.service import session as transport

    old, new = helper_session(), helper_session()
    sessions = iter([old, new])
    monkeypatch.setattr(transport, 'build_session', lambda **kwargs: next(sessions))
    assert qm.save_mine_and_token('OfflineMine', 'old secret') is None
    old_state = qm._state
    assert old.close_calls == 0
    new.routes[('GET', '/custom/service/user/queries')] = b'{}'
    assert qm.save_mine_and_token('OfflineMine', 'new secret') == 'An exception of type KeyError occurred. Check token'
    assert old.close_calls == new.close_calls == 1
    assert old_state.service._closed and old_state.registry._closed
    assert qm._state is None and qm.token == 'new secret'
    with pytest.raises(RuntimeError):
        qm.get_query('a')
    assert_closed(old)
    assert_closed(new)


def test_successful_reconfiguration_borrowed_session_lifetime_and_state_snapshot(qm):
    old, new = helper_session(), helper_session(b'{"queries":{"New":{}}}')
    configure(qm, old)
    old_state = qm._state
    configure(qm, new)
    assert old_state.service._closed and old_state.registry._closed
    qm.mine = 'changed'
    qm.token = 'changed'
    assert qm.get_all_query_names() == 'New'
    assert parse_qs(urlsplit(new.requests[-1].url).query)['token'] == [TOKEN]
    assert old.close_calls == new.close_calls == 0


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_borrowed_registry_cached_service_configuration_untouched(qm, profile):
    session = helper_session()
    with Registry(session=session, compatibility=profile, **OPTIONS) as registry:
        cached = registry['OfflineMine']
        assert qm.save_mine_and_token('OfflineMine', TOKEN, registry=registry) is None
        assert qm.get_all_query_names() == 'Zulu, a/+&人'
        qm._state.close()
        assert not registry._closed and not cached._closed
        assert registry['OfflineMine'] is cached
        assert cached.opener.token is None and registry._opener.token is None
        assert session.close_calls == 0
    assert_closed(session)


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_borrowed_service_authentication_and_session_untouched(qm, profile):
    session = helper_session()
    with Registry(session=session, compatibility=profile) as registry, Service(
        'https://data.example/custom', token='other secret', session=session, compatibility=profile,
    ) as service:
        original = service.opener.token, service.opener.auth_header
        assert qm.save_mine_and_token('OfflineMine', TOKEN, registry=registry, service=service) is None
        assert qm.get_all_query_names() == 'Zulu, a/+&人'
        assert (service.opener.token, service.opener.auth_header) == original
        assert parse_qs(urlsplit(session.requests[-1].url).query) == {'token': [TOKEN]}
        assert 'Authorization' not in session.requests[-1].headers
        qm._state.close()
        assert not service._closed and not registry._closed and session.close_calls == 0


def test_injected_account_opener_is_borrowed_and_responses_close(qm):
    session = helper_session()
    opener = FixtureOpener({('GET', '/custom/service/user/queries'): fixture_bytes('saved-queries.json')})
    with Registry(session=session) as registry:
        assert qm.save_mine_and_token('OfflineMine', TOKEN, registry=registry, opener=opener) is None
        assert qm.get_all_query_names() == 'Zulu, a/+&人'
        qm._state.close()
        assert opener.close_calls == 0 and not registry._closed
    assert all(connection.close_calls == 1 for connection in opener.connections)


@pytest.mark.parametrize('operation', ['names', 'get', 'delete', 'post'])
@pytest.mark.parametrize('payload,error', [(b'{}', KeyError), (b'{broken', json.JSONDecodeError),
                                         ((500, b'failed'), WebserviceError)])
def test_operations_propagate_account_errors_close_responses_and_keep_configuration(qm, operation, payload, error):
    session = helper_session()
    configure(qm, session)
    session.routes[('GET', '/custom/service/user/queries')] = payload
    calls = {'names': lambda: qm.get_all_query_names(), 'get': lambda: qm.get_query(NAME),
             'delete': lambda: qm.delete_query(NAME), 'post': lambda: qm.post_query(XML, overwrite=True)}
    if operation == 'get' and isinstance(payload, bytes):
        assert calls[operation]() == payload.decode()
    else:
        with pytest.raises(error):
            calls[operation]()
    assert qm._state is not None and session.close_calls == 0
    assert_closed(session)


@pytest.mark.parametrize('failure', [OSError('secret read failure'), KeyboardInterrupt('secret interrupt')])
def test_configuration_read_failure_or_interrupt_closes_owned_state(qm, failure, monkeypatch):
    from intermine314.service import session as transport

    session = helper_session()
    monkeypatch.setattr(transport, 'build_session', lambda **kwargs: session)
    original = _ResponseStreamAdapter.read

    def read(stream, *args, **kwargs):
        if stream._response is session.responses[-1] and len(session.responses) >= 4:
            raise failure
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(_ResponseStreamAdapter, 'read', read)
    if isinstance(failure, Exception):
        assert qm.save_mine_and_token('OfflineMine', TOKEN) == 'An exception of type OSError occurred. Check token'
    else:
        with pytest.raises(KeyboardInterrupt):
            qm.save_mine_and_token('OfflineMine', TOKEN)
    assert qm._state is None and session.close_calls == 1
    assert_closed(session)


def test_helpers_import_and_execute_without_heavy_or_original_imports():
    script = '''
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'matplotlib', 'lxml', 'intermine'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
from intermine314 import query_manager as qm
assert 'intermine314.service.service' not in sys.modules
from tests.test_query_manager_helpers import helper_session, TOKEN, XML
session = helper_session()
assert qm.save_mine_and_token('OfflineMine', TOKEN, session=session) is None
assert qm.get_all_query_names() == 'Zulu, a/+&人'
assert qm.post_query(XML, overwrite=True) == 'a/+&人 is posted'
qm._state.close()
assert session.close_calls == 0
'''
    result = subprocess.run([sys.executable, '-c', script], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('identity', ['token', 'basic'])
def test_borrowed_authenticated_opener_clone_clears_identity_without_mutation(qm, identity):
    from intermine314.results import InterMineURLOpener

    session = helper_session()
    options = {'token': 'other secret'} if identity == 'token' else {'credentials': ('other user', 'other password')}
    with Registry(session=session) as registry, InterMineURLOpener(session=session, **options) as opener:
        original = opener.token, opener.auth_header, opener.using_authentication
        assert qm.save_mine_and_token('OfflineMine', TOKEN, registry=registry, opener=opener) is None
        assert qm.get_all_query_names() == 'Zulu, a/+&人'
        assert (opener.token, opener.auth_header, opener.using_authentication) == original
        call = session.requests[-1]
        assert parse_qs(urlsplit(call.url).query) == {'token': [TOKEN]}
        assert 'Authorization' not in call.headers
        qm._state.close()
        assert session.close_calls == 0 and not registry._closed
    assert_closed(session)


@pytest.mark.parametrize('conflict', ['registry_options', 'service_and_opener', 'authenticated_opener'])
def test_invalid_injection_combinations_reject_before_io_preserve_existing_configuration(qm, conflict):
    session = helper_session()
    configure(qm, session)
    previous = qm._state
    with Registry(session=session) as registry:
        opener = FixtureOpener({})
        options = {'registry': registry}
        if conflict == 'registry_options':
            options['session'] = session
        elif conflict == 'service_and_opener':
            options.update(service=object(), opener=opener)
        else:
            opener.token = 'other secret'
            options['opener'] = opener
        before = len(session.requests)
        with pytest.raises(TypeError):
            qm.save_mine_and_token('OtherMine', 'new secret', **options)
        assert len(session.requests) == before and opener.requests == []
        assert qm._state is previous and qm.mine == 'OfflineMine' and qm.token == TOKEN
        assert not registry._closed and session.close_calls == 0


def test_mismatched_injected_service_never_receives_helper_credentials(qm):
    session = helper_session()
    with Registry(session=session) as registry, Service('https://offline.example', token='other secret', session=session) as service:
        before = len(session.requests)
        assert qm.save_mine_and_token('OfflineMine', TOKEN, registry=registry, service=service) == (
            'An exception of type ValueError occurred. Check mine')
        assert [call.path for call in session.requests[before:]] == ['/service/instances/OfflineMine']
        assert not service._closed and not registry._closed and qm._state is None
        assert service.opener.token == 'other secret'
    assert_closed(session)


@pytest.mark.parametrize('method', ['PUT', 'DELETE'])
def test_account_mutation_http_error_closes_response_and_leaves_borrowed_clients(qm, method):
    session = helper_session()
    configure(qm, session)
    path = '/custom/service/user/queries' + ('/' + quote(NAME, safe='') if method == 'DELETE' else '')
    session.routes[(method, path)] = (500, b'failed')
    with pytest.raises(WebserviceError):
        if method == 'PUT':
            qm.post_query(XML, overwrite=True)
        else:
            qm.delete_query(NAME)
    assert qm._state is not None and session.close_calls == 0
    assert_closed(session)


@pytest.mark.parametrize('operation', ['get', 'names', 'post_readback'])
@pytest.mark.parametrize('failure', [OSError('read failed'), KeyboardInterrupt('interrupted')])
def test_operation_read_error_or_interrupt_closes_active_response(qm, operation, failure, monkeypatch):
    session = helper_session()
    configure(qm, session)
    original = _ResponseStreamAdapter.read

    def read(stream, *args, **kwargs):
        if operation != 'post_readback' or any(call.method == 'PUT' for call in session.requests):
            raise failure
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(_ResponseStreamAdapter, 'read', read)
    with pytest.raises(type(failure)):
        if operation == 'get':
            qm.get_query(NAME)
        elif operation == 'names':
            qm.get_all_query_names()
        else:
            qm.post_query(XML, overwrite=True)
    assert qm._state is not None and session.close_calls == 0
    assert_closed(session)


@pytest.mark.parametrize('phase', ['initial_registry', 'detail', 'service', 'token'])
def test_failed_owned_configuration_closes_immediately_in_all_setup_phases(qm, phase, monkeypatch):
    from intermine314.service import session as transport

    session = helper_session()
    path = {'initial_registry': '/service/instances', 'detail': '/service/instances/OfflineMine',
            'service': '/custom/service/version/ws', 'token': '/custom/service/user/queries'}[phase]
    session.routes[('GET', path)] = (500, b'failed')
    monkeypatch.setattr(transport, 'build_session', lambda **kwargs: session)
    kind = 'ServiceError' if phase == 'service' else 'WebserviceError'
    hint = 'token' if phase == 'token' else 'mine'
    assert qm.save_mine_and_token('OfflineMine', TOKEN) == 'An exception of type ' + kind + ' occurred. Check ' + hint
    assert qm._state is None and session.close_calls == 1
    assert_closed(session)


def test_successful_owned_reconfiguration_releases_previous_clients_once(qm, monkeypatch):
    from intermine314.service import session as transport

    old, new = helper_session(), helper_session()
    sessions = iter([old, new])
    monkeypatch.setattr(transport, 'build_session', lambda **kwargs: next(sessions))
    assert qm.save_mine_and_token('OfflineMine', 'old') is None
    previous = qm._state
    assert qm.save_mine_and_token('OfflineMine', 'new') is None
    assert previous.service._closed and previous.registry._closed
    assert old.close_calls == 1 and new.close_calls == 0
    previous.close()
    assert old.close_calls == 1
    assert qm.get_all_query_names() == 'Zulu, a/+&人'
    qm._state.close()
    assert new.close_calls == 1
    assert_closed(old)
    assert_closed(new)


@pytest.mark.parametrize('allow_http', [False, True])
@pytest.mark.parametrize('tor_source', ['registry', 'opener'])
def test_injected_opener_preserves_resolved_root_tor_https_policy_before_account_io(qm, allow_http, tor_source):
    session = helper_session()
    detail = json.loads(fixture_bytes('registry-helpers.json'))
    detail['instance']['url'] = 'http://data.example/custom/service/'
    session.routes[('GET', '/service/instances/OfflineMine')] = json.dumps(detail).encode()
    opener = FixtureOpener({('GET', '/custom/service/user/queries'): fixture_bytes('saved-queries.json')})
    opener.tor_mode = tor_source == 'opener'
    options = {'allow_http_over_tor': allow_http}
    if tor_source == 'registry':
        options.update(tor=True, proxy_url='socks5h://127.0.0.1:9050')
    with Registry(session=session, **options) as registry:
        result = qm.save_mine_and_token('OfflineMine', TOKEN, registry=registry, opener=opener)
        if allow_http:
            assert result is None and opener.requests[0].url.startswith('http://')
            assert parse_qs(urlsplit(opener.requests[0].url).query) == {'token': [TOKEN]}
        else:
            assert result == 'An exception of type ValueError occurred. Check mine'
            assert opener.requests == [] and qm._state is None
        assert not registry._closed and session.close_calls == 0 and opener.close_calls == 0
    assert all(connection.close_calls == 1 for connection in opener.connections)


@pytest.mark.parametrize('overwrite', ['y', 1, 0, [], object()])
def test_unknown_overwrite_value_rejects_before_http(qm, overwrite):
    session = helper_session()
    configure(qm, session)
    before = len(session.requests)
    with pytest.raises(ValueError, match='overwrite must be None, True or False'):
        qm.post_query(XML, overwrite=overwrite)
    assert len(session.requests) == before
