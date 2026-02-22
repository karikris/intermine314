from __future__ import annotations

import warnings
from urllib.parse import urlparse

from intermine314.config.constants import (
    DEFAULT_REGISTRY_INSTANCES_URL,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TOR_PROXY_SCHEME,
    DEFAULT_TOR_SOCKS_HOST,
    DEFAULT_TOR_SOCKS_PORT,
)
from intermine314.service.errors import TorConfigurationError
from intermine314.service.transport import build_session

_TOR_DNS_SAFE_PROXY_SCHEME = "socks5h"


def _normalized_proxy_parts(proxy_url: str):
    parsed = urlparse(str(proxy_url))
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").strip("[]").lower()
    port = parsed.port
    return scheme, host, port


def _validate_tor_proxy_scheme(proxy_url: str):
    scheme, _host, _port = _normalized_proxy_parts(proxy_url)
    if scheme != _TOR_DNS_SAFE_PROXY_SCHEME:
        raise TorConfigurationError(
            "Tor routing requires socks5h:// proxy URLs in strict mode to avoid DNS leaks."
        )


def _warn_if_non_dns_safe_proxy_scheme(proxy_url: str):
    scheme, _host, _port = _normalized_proxy_parts(proxy_url)
    if scheme == _TOR_DNS_SAFE_PROXY_SCHEME:
        return
    warnings.warn(
        "Tor routing is configured with a non-DNS-safe proxy scheme. "
        "Use socks5h:// to avoid DNS leaks.",
        RuntimeWarning,
        stacklevel=3,
    )


def _validate_custom_tor_session(session, *, expected_proxy: str):
    if not hasattr(session, "request"):
        raise TorConfigurationError("Custom Tor session must provide a request(...) method.")
    proxies = getattr(session, "proxies", None)
    if not isinstance(proxies, dict):
        raise TorConfigurationError(
            "Custom Tor session must define both http and https proxies using socks5h://."
        )
    http_proxy = proxies.get("http")
    https_proxy = proxies.get("https")
    if not http_proxy or not https_proxy:
        raise TorConfigurationError(
            "Custom Tor session must define both http and https proxies using socks5h://."
        )

    _validate_tor_proxy_scheme(http_proxy)
    _validate_tor_proxy_scheme(https_proxy)
    expected_parts = _normalized_proxy_parts(expected_proxy)
    for key, value in (("http", http_proxy), ("https", https_proxy)):
        if _normalized_proxy_parts(value) != expected_parts:
            raise TorConfigurationError(
                f"Custom Tor session {key} proxy does not match expected Tor proxy {expected_proxy!r}."
            )

    if bool(getattr(session, "trust_env", False)):
        raise TorConfigurationError(
            "Custom Tor session must set trust_env=False in strict mode to avoid environment proxy bypass."
        )


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
    strict: bool = True,
    **service_kwargs,
):
    """
    Build a Service routed through Tor.

    In strict mode, custom sessions are validated to require socks5h proxies and
    trust_env=False to prevent DNS/proxy bypass leaks.
    """
    from intermine314.service.service import Service

    proxy = tor_proxy_url(host=host, port=port, scheme=scheme)
    if strict:
        _validate_tor_proxy_scheme(proxy)
    else:
        _warn_if_non_dns_safe_proxy_scheme(proxy)
    tor_http_session = session or tor_session(host=host, port=port, scheme=scheme)
    if strict and session is not None:
        _validate_custom_tor_session(tor_http_session, expected_proxy=proxy)
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
    max_cached_services=None,
    strict: bool = True,
):
    """
    Build a Registry client routed through Tor.

    In strict mode, custom sessions are validated to require socks5h proxies and
    trust_env=False to prevent DNS/proxy bypass leaks.
    """
    from intermine314.service.service import Registry

    proxy = tor_proxy_url(host=host, port=port, scheme=scheme)
    if strict:
        _validate_tor_proxy_scheme(proxy)
    else:
        _warn_if_non_dns_safe_proxy_scheme(proxy)
    tor_http_session = session or tor_session(host=host, port=port, scheme=scheme)
    if strict and session is not None:
        _validate_custom_tor_session(tor_http_session, expected_proxy=proxy)
    registry_kwargs = dict(
        registry_url=registry_url,
        request_timeout=request_timeout,
        proxy_url=proxy,
        session=tor_http_session,
        verify_tls=verify_tls,
        tor=True,
        allow_http_over_tor=bool(allow_http_over_tor),
    )
    if max_cached_services is not None:
        registry_kwargs["max_cached_services"] = max_cached_services
    return Registry(**registry_kwargs)
