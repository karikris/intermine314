from __future__ import annotations

from intermine314.config.constants import (
    DEFAULT_REGISTRY_INSTANCES_URL,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TOR_PROXY_SCHEME,
    DEFAULT_TOR_SOCKS_HOST,
    DEFAULT_TOR_SOCKS_PORT,
)
from intermine314.service.transport import build_session


def tor_proxy_url(
    host: str = DEFAULT_TOR_SOCKS_HOST,
    port: int = DEFAULT_TOR_SOCKS_PORT,
    scheme: str = DEFAULT_TOR_PROXY_SCHEME,
) -> str:
    return f"{scheme}://{host}:{int(port)}"


def tor_session(
    host: str = DEFAULT_TOR_SOCKS_HOST,
    port: int = DEFAULT_TOR_SOCKS_PORT,
    scheme: str = DEFAULT_TOR_PROXY_SCHEME,
    user_agent: str | None = None,
):
    return build_session(proxy_url=tor_proxy_url(host=host, port=port, scheme=scheme), user_agent=user_agent)


def tor_service(
    root: str,
    *,
    host: str = DEFAULT_TOR_SOCKS_HOST,
    port: int = DEFAULT_TOR_SOCKS_PORT,
    scheme: str = DEFAULT_TOR_PROXY_SCHEME,
    session=None,
    allow_http_over_tor: bool = False,
    **service_kwargs,
):
    from intermine314.service.service import Service

    proxy = tor_proxy_url(host=host, port=port, scheme=scheme)
    tor_http_session = session or tor_session(host=host, port=port, scheme=scheme)
    return Service(
        root,
        proxy_url=proxy,
        session=tor_http_session,
        tor=True,
        allow_http_over_tor=bool(allow_http_over_tor),
        **service_kwargs,
    )


def tor_registry(
    registry_url: str = DEFAULT_REGISTRY_INSTANCES_URL,
    *,
    host: str = DEFAULT_TOR_SOCKS_HOST,
    port: int = DEFAULT_TOR_SOCKS_PORT,
    scheme: str = DEFAULT_TOR_PROXY_SCHEME,
    request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    session=None,
    verify_tls: bool = True,
    allow_http_over_tor: bool = False,
):
    from intermine314.service.service import Registry

    proxy = tor_proxy_url(host=host, port=port, scheme=scheme)
    tor_http_session = session or tor_session(host=host, port=port, scheme=scheme)
    return Registry(
        registry_url=registry_url,
        request_timeout=request_timeout,
        proxy_url=proxy,
        session=tor_http_session,
        verify_tls=verify_tls,
        tor=True,
        allow_http_over_tor=bool(allow_http_over_tor),
    )
