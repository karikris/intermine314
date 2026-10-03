"""Account API contracts using strictly offline transports; no real mutations."""

import json
from urllib.parse import parse_qs, urlsplit

import pytest

from intermine314.service.errors import ServiceError, WebserviceError
from intermine314.service.service import Service
from intermine314.service.session import InterMineURLOpener, _ResponseStreamAdapter
from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession


def account_session(version=16):
    session = FixtureSession.service(version=version)
    session.routes.update({
        ('POST', '/service/users'): b'{"error":null,"user":{"temporaryToken":"new-token"}}',
        ('POST', '/service/user/deregistration'): b'{"error":null,"token":{"uuid":"remove-token"}}',
        ('DELETE', '/service/user'): '<user name="Müller"/>'.encode(),
        ('GET', '/service/session'): b'{"token":"anonymous-token"}',
    })
    return session


@pytest.mark.parametrize('factory', ['native_service_factory', 'legacy_service_factory'])
def test_register_unicode_configuration_auth_and_borrowed_lifetime(request, factory, caplog):
    session = account_session()
    service = request.getfixturevalue(factory)(
        session=session, token='original-token', prefetch_depth=4, prefetch_id_only=True,
        request_timeout=13, proxy_url='socks5h://127.0.0.1:9050', tor=True,
        strict_tor_proxy_scheme=False, allow_insecure_tor_proxy_scheme=True,
        allow_http_over_tor=True, verify_tls='/custom/ca.pem', user_agent='account-client',
    )
    original_opener = service.opener
    caplog.set_level('DEBUG')
    registered = service.register('Müller 人 🧬 &+', 'sëcret 人 &+')
    assert isinstance(registered, type(service))
    assert registered.compatibility == service.compatibility
    assert registered.root == SERVICE_ROOT
    assert registered.opener.token == 'new-token'
    for name in ('prefetch_depth', 'prefetch_id_only', 'request_timeout', 'proxy_url', 'tor',
                 'strict_tor_proxy_scheme', 'allow_insecure_tor_proxy_scheme',
                 'allow_http_over_tor', 'verify_tls', 'user_agent'):
        assert getattr(registered, name) == getattr(service, name)
    assert registered.opener._session is session
    assert not registered._owns_session
    assert service.opener is original_opener and original_opener.token == 'original-token'
    post = session.requests[1]
    assert post.method == 'POST' and post.path == '/service/users'
    assert parse_qs(post.data.decode()) == {'name': ['Müller 人 🧬 &+'], 'password': ['sëcret 人 &+']}
    assert 'Authorization' not in post.headers and not urlsplit(post.url).query
    assert post.headers['Accept'] == 'application/json'
    assert post.headers['Content-Type'] == 'application/x-www-form-urlencoded; charset=utf-8'
    assert session.requests[2].headers['Authorization'] == registered.opener.auth_header
    for call in session.requests:
        assert call.headers['User-Agent'] == 'account-client'
        assert call.options == {'stream': True, 'timeout': (10.0, 13.0), 'verify': '/custom/ca.pem'}
    assert all(response.close_calls == 1 for response in session.responses)
    assert 'sëcret' not in caplog.text and 'original-token' not in caplog.text and 'new-token' not in caplog.text
    registered.close()
    assert service.get_anonymous_token(SERVICE_ROOT) == 'anonymous-token'
    service.close()
    assert session.close_calls == 0


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('close_first', ['original', 'registered'])
def test_register_owned_sessions_are_independent(monkeypatch, profile, close_first):
    original_session, registered_session = account_session(), account_session()
    sessions = iter([original_session, registered_session])
    configurations = []

    def build_session(opener):
        configurations.append((opener.proxy_url, opener.tor_mode, opener._timeout,
                               opener._verify_tls, opener._user_agent))
        return next(sessions)

    monkeypatch.setattr(InterMineURLOpener, '_build_managed_session', build_session)
    original = Service(SERVICE_ROOT, username='old-user', password='old-password', compatibility=profile,
                       proxy_url='socks5h://127.0.0.1:9050', tor=True,
                       request_timeout=13, verify_tls='/custom/ca.pem', user_agent='account-client')
    registered = original.register('new-user', 'new-password')
    assert configurations == [('socks5h://127.0.0.1:9050', True, (10.0, 13.0),
                               '/custom/ca.pem', 'account-client')] * 2
    assert original.opener._session is original_session
    assert registered.opener._session is registered_session
    assert original._owns_session and registered._owns_session
    assert original.opener.using_authentication
    assert 'Authorization' not in original_session.requests[-1].headers
    first, other = (original, registered) if close_first == 'original' else (registered, original)
    first.close()
    assert other.get_anonymous_token(SERVICE_ROOT) == 'anonymous-token'
    other.close()
    first.close()
    assert original_session.close_calls == registered_session.close_calls == 1


@pytest.mark.parametrize('owned', [False, True])
def test_register_failed_returned_initialization_closes_only_owned_session(monkeypatch, owned):
    original_session, returned_session = account_session(), account_session()
    sessions = iter([original_session, returned_session])
    monkeypatch.setattr(InterMineURLOpener, '_build_managed_session', lambda self: next(sessions))
    original = Service(SERVICE_ROOT, session=None if owned else original_session, token='old-token')
    bad_session = returned_session if owned else original_session
    bad_session.routes[('GET', '/service/version/ws')] = b'not a version'
    with pytest.raises(ServiceError):
        original.register('new-user', 'new-password')
    assert returned_session.close_calls == int(owned)
    assert original_session.close_calls == 0
    assert original.opener.token == 'old-token'
    assert all(response.close_calls == 1 for response in bad_session.responses)
    original.close()
    assert original_session.close_calls == int(owned)


@pytest.mark.parametrize('factory', ['native_service_factory', 'legacy_service_factory'])
@pytest.mark.parametrize('validity', [None, 1, 86400])
def test_deregistration_token_wire_and_bounds(request, factory, validity):
    session = account_session()
    service = request.getfixturevalue(factory)(session=session, token='old-token')
    result = service.get_deregistration_token() if validity is None else service.get_deregistration_token(validity)
    assert result == {'uuid': 'remove-token'}
    call = session.requests[-1]
    assert call.method == 'POST' and call.path == '/service/user/deregistration'
    assert parse_qs(call.data.decode()) == {'validity': [str(300 if validity is None else validity)]}
    assert call.headers['Authorization'] == service.opener.auth_header
    assert call.headers['Accept'] == 'application/json'
    assert session.responses[-1].close_calls == 1
    for invalid in (0, -1, 86401):
        count = len(session.requests)
        with pytest.raises(ValueError, match='1.*86400'):
            service.get_deregistration_token(invalid)
        assert len(session.requests) == count


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('token', ['uuid-in-string &+ 人', {'uuid': 'remove &+ 人'}])
def test_deregister_flushes_before_delete_and_returns_bytes(monkeypatch, profile, token):
    session = account_session()
    service = Service(SERVICE_ROOT, session=session, token='old-token', compatibility=profile)
    old_model = service.model
    service.release
    service.widgets
    service._resolve_query_model()
    caches = ('_model', '_model_xml', '_model_name', '_query_model', '_version', '_release', '_widgets')
    original_request = session.request

    def capture(method, url, **kwargs):
        if method == 'DELETE':
            assert all(getattr(service, name) is None for name in caches)
        return original_request(method, url, **kwargs)

    monkeypatch.setattr(session, 'request', capture)
    assert service.deregister(token) == '<user name="Müller"/>'.encode()
    call = session.requests[-1]
    assert call.method == 'DELETE' and call.path == '/service/user' and call.data is None
    assert parse_qs(urlsplit(call.url).query) == {
        'deregistrationToken': [token['uuid'] if isinstance(token, dict) else token],
        'format': ['xml'], 'token': ['old-token'],
    }
    assert call.headers['Authorization'] == service.opener.auth_header
    assert session.responses[-1].close_calls == 1
    assert service.version == 16
    assert service.model is not old_model and service.model.service == service
    assert service.widgets and service.release


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_flush_clears_implemented_caches_without_request_or_auth_change(profile):
    session = account_session()
    service = Service(SERVICE_ROOT, session=session, token='old-token', compatibility=profile)
    old_model, old_widgets = service.model, service.widgets
    service.release
    count = len(session.requests)
    assert service.flush() is None
    assert len(session.requests) == count and service.opener.token == 'old-token'
    assert service.model is not old_model and service.widgets is not old_widgets


@pytest.mark.parametrize('owned', [False, True])
@pytest.mark.parametrize('failure', ['http', 'read', 'interrupt', 'parse', 'version', 'unsupported'])
def test_random_constructor_failure_closes_only_owned_resources(monkeypatch, owned, failure):
    session = account_session(5 if failure == 'unsupported' else 16)
    monkeypatch.setattr(InterMineURLOpener, '_build_managed_session', lambda self: session)
    error = ServiceError
    if failure == 'http':
        session.routes[('GET', '/service/session')] = (400, b'{"error":"failed"}')
        error = WebserviceError
    elif failure == 'parse':
        session.routes[('GET', '/service/session')] = b'{broken'
        error = json.JSONDecodeError
    elif failure == 'version':
        session.routes[('GET', '/service/version/ws')] = b'bad version'
    elif failure in ('read', 'interrupt'):
        error = OSError if failure == 'read' else KeyboardInterrupt

        def fail_read(*args, **kwargs):
            raise error()

        monkeypatch.setattr(_ResponseStreamAdapter, 'read', fail_read)
    with pytest.raises(error):
        Service(SERVICE_ROOT, token='random', session=None if owned else session)
    assert session.close_calls == int(owned)
    assert all(response.close_calls == 1 for response in session.responses)


@pytest.mark.parametrize('method,args,version,required', [
    ('register', ('user', 'password'), 8, 9),
    ('get_deregistration_token', (), 15, 16),
    ('deregister', ('token',), 15, 16),
])
@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_account_version_gates_prevent_mutations(method, args, version, required, profile):
    session = account_session(version)
    service = Service(SERVICE_ROOT, session=session, compatibility=profile)
    with pytest.raises(ServiceError, match=f'version {required}'):
        getattr(service, method)(*args)
    assert len(session.requests) == 1


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_public_anonymous_and_random_constructor_use_configured_transport(monkeypatch, profile):
    session = account_session()
    monkeypatch.setattr(InterMineURLOpener, '_build_managed_session', lambda self: session)
    service = Service(SERVICE_ROOT, token='random', request_timeout=13, verify_tls=False,
                      proxy_url='socks5h://127.0.0.1:9050', tor=True,
                      user_agent='account-client', compatibility=profile)
    assert service._owns_session and service.opener._owns_session
    assert service.opener._session_finalizer.alive
    assert service.opener.token == 'anonymous-token'
    assert service.get_anonymous_token(SERVICE_ROOT) == 'anonymous-token'
    assert [call.path for call in session.requests] == ['/service/session', '/service/version/ws', '/service/session']
    assert 'Authorization' not in session.requests[0].headers
    assert session.requests[-1].headers['Authorization'] == service.opener.auth_header
    for call in session.requests:
        assert call.options == {'stream': True, 'timeout': (10.0, 13.0), 'verify': False}
        assert call.headers['User-Agent'] == 'account-client'
    assert all(response.close_calls == 1 for response in session.responses)
    service.close()
    service.close()
    assert session.close_calls == 1 and service.opener._session_finalizer is None


@pytest.mark.parametrize('method,args,path,http_method', [
    ('register', ('user', 'password'), '/service/users', 'POST'),
    ('get_deregistration_token', (), '/service/user/deregistration', 'POST'),
    ('deregister', ('token',), '/service/user', 'DELETE'),
    ('get_anonymous_token', (SERVICE_ROOT,), '/service/session', 'GET'),
])
@pytest.mark.parametrize('failure', ['http', 'read', 'interrupt'])
def test_account_responses_close_on_failures(monkeypatch, method, args, path, http_method, failure):
    session = account_session()
    service = Service(SERVICE_ROOT, session=session)
    session.routes[(http_method, path)] = (400, b'{"error":"failed"}') if failure == 'http' else b'body'
    if failure != 'http':
        def fail_read(*args, **kwargs):
            raise OSError('offline read failed') if failure == 'read' else KeyboardInterrupt()

        monkeypatch.setattr(_ResponseStreamAdapter, 'read', fail_read)
    with pytest.raises({'http': WebserviceError, 'read': OSError, 'interrupt': KeyboardInterrupt}[failure]):
        getattr(service, method)(*args)
    assert session.responses[-1].close_calls == 1
    assert session.close_calls == 0


@pytest.mark.parametrize('method,args,path,payload,error', [
    ('register', ('user', 'password'), '/service/users', b'{"error":"name taken"}', ServiceError),
    ('register', ('user', 'password'), '/service/users', b'{broken', json.JSONDecodeError),
    ('register', ('user', 'password'), '/service/users', b'{"error":null,"user":{}}', KeyError),
    ('get_deregistration_token', (), '/service/user/deregistration', b'{"error":"denied"}', ServiceError),
    ('get_deregistration_token', (), '/service/user/deregistration', b'{"error":null}', KeyError),
    ('get_anonymous_token', (SERVICE_ROOT,), '/service/session', b'{broken', json.JSONDecodeError),
    ('get_anonymous_token', (SERVICE_ROOT,), '/service/session', b'{}', KeyError),
])
def test_account_payload_errors_close_without_closing_borrowed_session(method, args, path, payload, error):
    session = account_session()
    session.routes[('GET' if method == 'get_anonymous_token' else 'POST', path)] = payload
    service = Service(SERVICE_ROOT, session=session)
    with pytest.raises(error):
        getattr(service, method)(*args)
    assert session.responses[-1].close_calls == 1 and session.close_calls == 0
