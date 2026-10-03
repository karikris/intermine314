"""Source-backed registry outputs and transport ownership on strict offline routes."""

import importlib
import json
import os
import subprocess
import sys
from urllib.parse import parse_qs
from xml.etree import ElementTree

import pytest

from intermine314.service.errors import WebserviceError
from intermine314.service.session import _ResponseStreamAdapter
from intermine314.webservice import Registry, Service
from tests.fixtures.compatibility import FixtureSession, fixture_bytes


def helpers():
    return importlib.import_module('intermine314.registry')


def helper_session(detail=None, rows=None):
    session = FixtureSession({
        ('GET', '/service/instances'): fixture_bytes('registry-helpers.json'),
        ('GET', '/service/instances/OfflineMine'): fixture_bytes('registry-helpers.json') if detail is None else detail,
        ('GET', '/custom/service/version/ws'): b'30',
        ('GET', '/custom/service/model'): fixture_bytes('datasets-model.xml'),
        ('POST', '/custom/service/query/results'): fixture_bytes('datasets-rows.json') if rows is None else rows,
    })
    return session


OPTIONS = dict(request_timeout=13, proxy_url='socks5h://127.0.0.1:9050', tor=True,
               verify_tls='/custom/ca.pem', user_agent='registry-helper-client')


def test_versions_keep_literal_keys_detail_case_and_configured_https_transport():
    session = helper_session()
    assert helpers().getVersion('OfflineMine', session=session, **OPTIONS) == {
        'API Version:': '30', 'Release Version:': '48 2019 October', 'InterMine Version:': '4.1.0',
    }
    assert [call.path for call in session.requests] == ['/service/instances', '/service/instances/OfflineMine']
    assert all(call.url.startswith('https://') for call in session.requests)
    for call in session.requests:
        assert call.options == {'stream': True, 'timeout': (10.0, 13.0), 'verify': '/custom/ca.pem'}
        assert call.headers['User-Agent'] == 'registry-helper-client'
    assert all(response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 0


def test_info_exact_labels_order_unicode_and_none_return(capsys):
    session = helper_session()
    assert helpers().getInfo('OfflineMine', session=session) is None
    assert capsys.readouterr().out == (
        'Description: Offline genomics 人\nURL: https://data.example/custom/service/\n'
        'API Version: 30\nRelease Version: 48 2019 October\nInterMine Version: 4.1.0\n'
        'Organisms: \nD. melanogaster\nH. sapiens\nNeighbours: \nMODs\nHumanMine\n')
    assert all(response.close_calls == 1 for response in session.responses)


@pytest.mark.parametrize('helper', ['getVersion', 'getInfo', 'getData'])
@pytest.mark.parametrize('detail', [b'{}', b'{"instance":{}}'])
def test_missing_instance_or_required_fields_return_historical_message(helper, detail, capsys):
    session = helper_session(detail=detail)
    assert getattr(helpers(), helper)('OfflineMine', session=session) == 'No such mine available'
    assert capsys.readouterr().out == ''
    assert all(response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 0


def test_info_preserves_partial_print_before_missing_key(capsys):
    detail = json.dumps({'instance': {'description': 'partial'}}).encode()
    assert helpers().getInfo('OfflineMine', session=helper_session(detail)) == 'No such mine available'
    assert capsys.readouterr().out == 'Description: partial\n'


@pytest.mark.parametrize('organism,output,result', [
    (None, 'OfflineMine\nHumanMine\nSpaceMine\n', None),
    ('D. melanogaster', 'OfflineMine\nOfflineMine\n', None),
    (' D. melanogaster', 'OfflineMine\nOfflineMine\nSpaceMine\n', None),
    ('H. sapiens', 'OfflineMine\nHumanMine\n', None),
    ('d. melanogaster', '', 'No such mine available'),
    ('absent', '', 'No such mine available'),
])
def test_mines_preserve_source_matching_order_duplicates_and_no_match(organism, output, result, capsys):
    session = helper_session()
    assert helpers().getMines(organism, session=session) == result
    assert capsys.readouterr().out == output
    assert [call.path for call in session.requests] == ['/service/instances']
    assert session.responses[0].close_calls == 1


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_data_sort_missing_name_exact_query_detail_url_and_borrowed_registry(profile, capsys, monkeypatch):
    session = helper_session()
    created = []
    with Registry(session=session, compatibility=profile, **OPTIONS) as registry:
        original = registry._new_service

        def capture(root, **kwargs):
            service = original(root, **kwargs)
            created.append(service)
            return service

        monkeypatch.setattr(registry, '_new_service', capture)
        assert helpers().getData('OfflineMine', registry=registry) is None
        assert capsys.readouterr().out == 'No info available\nName: Alpha\nName: Alpha\nName: Zulu\n'
        service, = created
        assert service._closed and not service._owns_session
        assert service.proxy_url == registry.proxy_url and service.tor is registry.tor is True
        assert service.verify_tls == registry.verify_tls == '/custom/ca.pem'
        assert not registry._closed and registry.service_cache_metrics()['cache_size'] == 0
        assert helpers().getVersion('OfflineMine', registry=registry)['API Version:'] == '30'
    query_call = next(call for call in session.requests if call.method == 'POST')
    assert query_call.url == 'https://data.example/custom/service/query/results'
    parameters = parse_qs(query_call.data.decode())
    query = ElementTree.fromstring(parameters['query'][0])
    assert query.attrib['view'] == 'DataSet.name DataSet.url'
    assert query.attrib['model'] == 'genomic' and parameters['format'] == ['json']
    assert all(response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 0


@pytest.mark.parametrize('helper', ['getVersion', 'getInfo', 'getData'])
@pytest.mark.parametrize('payload,error', [
    (b'{broken', json.JSONDecodeError), (b'\xff', UnicodeDecodeError),
    ((400, b'{"error":"missing"}'), WebserviceError),
])
def test_detail_parse_and_http_errors_propagate_and_owned_registry_closes(helper, payload, error, monkeypatch):
    session = helper_session(detail=payload)
    from intermine314.service import session as transport
    monkeypatch.setattr(transport, 'build_session', lambda **kwargs: session)
    with pytest.raises(error):
        getattr(helpers(), helper)('OfflineMine')
    assert all(response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 1


@pytest.mark.parametrize('helper', ['getVersion', 'getInfo', 'getData', 'getMines'])
def test_owned_helper_registry_session_closes_on_success(helper, capsys, monkeypatch):
    session = helper_session()
    from intermine314.service import session as transport
    monkeypatch.setattr(transport, 'build_session', lambda **kwargs: session)
    getattr(helpers(), helper)(*(() if helper == 'getMines' else ('OfflineMine',)))
    capsys.readouterr()
    assert all(response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 1


@pytest.mark.parametrize('failure', [OSError('read failed'), KeyboardInterrupt()])
def test_detail_read_failure_or_interrupt_closes_response_and_owned_registry(failure, monkeypatch):
    session = helper_session()
    from intermine314.service import session as transport
    monkeypatch.setattr(transport, 'build_session', lambda **kwargs: session)
    original = _ResponseStreamAdapter.read

    def read(stream, *args, **kwargs):
        if stream._response is session.responses[0]:
            return original(stream, *args, **kwargs)
        raise failure

    monkeypatch.setattr(_ResponseStreamAdapter, 'read', read)
    with pytest.raises(type(failure)):
        helpers().getVersion('OfflineMine')
    assert all(response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 1


@pytest.mark.parametrize('rows,error', [(b'{broken', WebserviceError),
                                       (b'{"results":[\n[null,"url"]\n],"wasSuccessful":true,"error":null}', TypeError)])
def test_data_query_or_print_failure_closes_local_service_leaves_registry_open(rows, error, monkeypatch, capsys):
    session = helper_session(rows=rows)
    with Registry(session=session) as registry:
        created = []
        original = registry._new_service

        def capture(root, **kwargs):
            service = original(root, **kwargs)
            created.append(service)
            return service

        monkeypatch.setattr(registry, '_new_service', capture)
        with pytest.raises(error):
            helpers().getData('OfflineMine', registry=registry)
        assert len(created) == 1 and created[0]._closed
        assert not registry._closed and session.close_calls == 0
        assert all(response.close_calls == 1 for response in session.responses)
    capsys.readouterr()


def test_injected_registry_conflicting_options_rejected_before_request():
    session = helper_session()
    with Registry(session=session) as registry:
        before = len(session.requests)
        with pytest.raises(TypeError, match='registry_options'):
            helpers().getVersion('OfflineMine', registry=registry, session=session)
        assert len(session.requests) == before and not registry._closed


def test_legacy_registry_mapping_mines_json_factory_errors_and_cache():
    session = FixtureSession.service()
    session.routes[('GET', '/registry/mines.json')] = b'{"mines":[{"name":"OfflineMine","webServiceRoot":"https://offline.example/service"}]}'
    with Registry('https://registry.example/registry', session=session) as registry:
        assert registry.compatibility == 'legacy'
        assert len(registry) == 1 and list(registry) == registry.keys() == ['OfflineMine']
        assert 'OFFLINEMINE' in registry and 'missing' not in registry
        service = registry['offlinemine']
        assert type(service) is Service and service.root == 'https://offline.example/service'
        assert registry['OFFLINEMINE'] is service
        with pytest.raises(KeyError) as failure:
            registry['Missing']
        assert failure.value.args == ('Unknown mine: Missing',)
        with pytest.raises(NotImplementedError, match='You cannot add items to a registry'):
            registry['new'] = service
        with pytest.raises(NotImplementedError, match='You cannot remove items from a registry'):
            del registry['offlinemine']
    assert service._closed and session.close_calls == 0
    assert all(response.close_calls == 1 for response in session.responses)


def test_helpers_import_and_execution_keep_analytics_and_original_package_lazy():
    script = '''
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'matplotlib', 'intermine'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
from intermine314 import registry
assert 'intermine314.service.service' not in sys.modules
from tests.test_registry_helpers import helper_session
session = helper_session()
assert registry.getVersion('OfflineMine', session=session)['API Version:'] == '30'
assert registry.getInfo('OfflineMine', session=session) is None
assert registry.getMines(session=session) is None
assert registry.getData('OfflineMine', session=session) is None
assert all(response.close_calls == 1 for response in session.responses)
assert session.close_calls == 0
'''
    result = subprocess.run([sys.executable, '-c', script], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('registry_url', ['https://custom.example', 'https://custom.example/service/instances/'])
def test_explicit_registry_root_or_instances_endpoint_is_used_for_detail(registry_url):
    session = helper_session()
    assert helpers().getVersion('OfflineMine', registry_url=registry_url, session=session)['API Version:'] == '30'
    assert session.requests[-1].url == 'https://custom.example/service/instances/OfflineMine'


def test_detail_preserves_lowercase_lookup_even_when_registry_name_has_other_case():
    session = helper_session()
    session.routes[('GET', '/service/instances/offlinemine')] = b'{}'
    assert helpers().getVersion('offlinemine', session=session) == 'No such mine available'
    assert session.requests[-1].path == '/service/instances/offlinemine'


@pytest.mark.parametrize('helper', ['getVersion', 'getInfo', 'getData', 'getMines'])
def test_helpers_leave_borrowed_registry_cached_services_open(helper, capsys):
    session = helper_session()
    session.routes.update(FixtureSession.service().routes)
    with Registry(session=session) as registry:
        cached = registry['OFFLINEMINE']
        getattr(helpers(), helper)(*(() if helper == 'getMines' else ('OfflineMine',)), registry=registry)
        assert registry['offlinemine'] is cached and not cached._closed
        assert not registry._closed and session.close_calls == 0
    assert cached._closed
    capsys.readouterr()


def test_data_print_interrupt_during_active_stream_closes_owned_stream_and_service(monkeypatch):
    session = helper_session()
    with Registry(session=session) as registry:
        created = []
        original = registry._new_service

        def capture(root, **kwargs):
            service = original(root, **kwargs)
            created.append(service)
            return service

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(registry, '_new_service', capture)
        monkeypatch.setattr('builtins.print', interrupt)
        with pytest.raises(KeyboardInterrupt):
            helpers().getData('OfflineMine', registry=registry)
        assert created[0]._closed and not registry._closed
        assert all(response.close_calls == 1 for response in session.responses)
        assert session.close_calls == 0


def test_empty_registry_reports_no_mines(capsys):
    session = helper_session()
    session.routes[('GET', '/service/instances')] = b'{"instances":[]}'
    assert helpers().getMines(session=session) == 'No such mine available'
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('target', ['helper', 'native', 'legacy'])
@pytest.mark.parametrize('borrowed', [False, True])
@pytest.mark.parametrize('failure', ['json', 'http', 'read', 'interrupt', 'schema', 'cache'])
def test_initial_registry_failure_closes_transport_with_retained_traceback(target, borrowed, failure, monkeypatch):
    from intermine314.service import session as transport
    from intermine314.service.errors import ServiceError
    from intermine314.service.service import Registry as NativeRegistry

    session = helper_session()
    options = {'session': session} if borrowed else {}
    if not borrowed:
        monkeypatch.setattr(transport, 'build_session', lambda **kwargs: session)
    expected = {
        'json': json.JSONDecodeError, 'http': WebserviceError, 'read': OSError,
        'interrupt': KeyboardInterrupt, 'schema': ServiceError, 'cache': ValueError,
    }[failure]
    primary = None
    if failure == 'json':
        session.routes[('GET', '/service/instances')] = b'{broken'
    elif failure == 'http':
        session.routes[('GET', '/service/instances')] = (400, b'{"error":"initial GET failed"}')
    elif failure == 'schema':
        session.routes[('GET', '/service/instances')] = b'{}'
    elif failure == 'cache':
        options['max_cached_services'] = 0
    else:
        primary = KeyboardInterrupt('initial GET interrupted') if failure == 'interrupt' else OSError('initial read failed')

        def fail_read(*args, **kwargs):
            raise primary

        monkeypatch.setattr(_ResponseStreamAdapter, 'read', fail_read)
    with pytest.raises(expected) as retained:
        if target == 'helper':
            helpers().getVersion('OfflineMine', **options)
        else:
            (NativeRegistry if target == 'native' else Registry)(**options)
    if primary is not None:
        assert retained.value is primary
    traceback = retained.value.__traceback__
    failed_registry = None
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_name == '__init__' and isinstance(frame.f_locals.get('self'), NativeRegistry):
            failed_registry = frame.f_locals['self']
            break
        traceback = traceback.tb_next
    assert failed_registry is not None  # Keep the failed client alive; no GC cleanup can satisfy this test.
    assert session.close_calls == (0 if borrowed else 1)
    assert failed_registry._closed
    assert all(response.close_calls == 1 for response in session.responses)
    failed_registry.close()
    assert session.close_calls == (0 if borrowed else 1)


@pytest.mark.parametrize('secondary', [OSError('close failed'), KeyboardInterrupt('close interrupted')])
def test_registry_constructor_cleanup_retains_primary_error_when_session_close_fails(secondary, monkeypatch):
    from intermine314.service import session as transport

    session = helper_session()
    session.routes[('GET', '/service/instances')] = b'{broken'

    def fail_close():
        session.close_calls += 1
        raise secondary

    monkeypatch.setattr(session, 'close', fail_close)
    monkeypatch.setattr(transport, 'build_session', lambda **kwargs: session)
    with pytest.raises(json.JSONDecodeError) as retained:
        helpers().getVersion('OfflineMine')
    assert retained.value.__traceback__ is not None
    assert session.close_calls == 1
    assert session.responses[0].close_calls == 1
