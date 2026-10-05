"""Lazy, independently owned transport snapshots for historical helper modules."""

from __future__ import annotations

from contextlib import ExitStack, closing
from urllib.parse import urlencode

from intermine314.registry import _instance, _registry_client
from intermine314.service.transport import open_readonly


class _HelperState:
    """Own local clients while borrowing injected clients and their sessions.

    Each historical module keeps its own instance. Account credentials are sent
    only as encoded parameters to the resolved service, never to the registry.
    """

    def __init__(self, mine, token):
        self.mine = mine
        self.token = token
        self.registry = None
        self.service = None
        self.opener = None
        self.root = None
        self._resources = ExitStack()

    @staticmethod
    def validate_options(registry, service, opener, registry_options):
        if registry is not None and registry_options:
            raise TypeError('registry_options cannot be combined with an injected registry')
        if service is not None and opener is not None:
            raise TypeError('service and opener cannot be combined')
        # A custom authenticated opener without a clone cannot safely separate
        # the caller's identity from the account token supplied to the helper.
        if opener is not None and not callable(getattr(opener, 'clone', None)):
            if getattr(opener, 'token', None) or getattr(opener, 'using_authentication', False):
                raise TypeError('authenticated opener must support clone()')

    def connect(self, *, registry=None, service=None, opener=None, registry_options):
        from intermine314.service.service import _require_https_when_tor
        from intermine314.service.urls import normalize_service_root

        self.registry = self._resources.enter_context(_registry_client(registry, registry_options))
        self.root = normalize_service_root(_instance(self.registry, self.mine)['url'])
        account_opener = service.opener if service is not None else opener
        _require_https_when_tor(
            self.root,
            tor_enabled=self.registry.tor or bool(getattr(account_opener, 'tor_mode', False)),
            allow_http_over_tor=self.registry.allow_http_over_tor,
            context='Helper service root URL',
        )
        if service is not None:
            if service.root != self.root:
                raise ValueError('injected service must match the resolved mine')
            self.service = service
            opener = service.opener
        elif opener is None:
            client = self.registry
            self.service = self._resources.enter_context(client._new_service(
                self.root, request_timeout=client.request_timeout,
                proxy_url=client.proxy_url, session=client._session,
                verify_tls=client.verify_tls, tor=client.tor,
                strict_tor_proxy_scheme=client.strict_tor_proxy_scheme,
                allow_insecure_tor_proxy_scheme=client.allow_insecure_tor_proxy_scheme,
                allow_http_over_tor=client.allow_http_over_tor,
                user_agent=client.user_agent, compatibility=client.compatibility,
            ))
            self.opener = self.service.opener
            return
        clone = getattr(opener, 'clone', None)
        if callable(clone):
            self.opener = clone()
            self._resources.callback(self.opener.close)
            self.opener.token = None
            self.opener.using_authentication = False
        else:
            self.opener = opener

    def url(self, path, parameters=None):
        parameters = {} if parameters is None else dict(parameters)
        parameters['token'] = self.token
        return self.root + path + '?' + urlencode(parameters)

    def read(self, path, parameters=None, *, method='GET'):
        request = open_readonly if method.upper() in {'GET', 'HEAD'} else lambda opener, *a, **kw: opener.open(*a, **kw)
        with closing(request(self.opener, self.url(path, parameters), method=method)) as response:
            payload = response.read()
        return payload.decode('utf-8') if isinstance(payload, bytes) else payload

    def discard(self, path, parameters=None, *, method):
        with closing(self.opener.open(self.url(path, parameters), method=method)):
            pass

    def close(self):
        self._resources.close()
