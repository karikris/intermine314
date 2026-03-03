from __future__ import annotations
from urllib.parse import urlparse

from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.service.errors import TorConfigurationError
from intermine314.service.transport import (
    build_session,
    enforce_tor_dns_safe_proxy_url,
)

_SERVICE_DEFAULTS = get_runtime_defaults().service_defaults
_DEFAULT_REGISTRY_INSTANCES_URL = _SERVICE_DEFAULTS.default_registry_instances_url
_DEFAULT_REQUEST_TIMEOUT_SECONDS = _SERVICE_DEFAULTS.default_request_timeout_seconds
_DEFAULT_TOR_PROXY_SCHEME = _SERVICE_DEFAULTS.default_tor_proxy_scheme
_DEFAULT_TOR_SOCKS_HOST = _SERVICE_DEFAULTS.default_tor_socks_host
_DEFAULT_TOR_SOCKS_PORT = _SERVICE_DEFAULTS.default_tor_socks_port


def _normalized_proxy_parts(proxy_url: str):
    parsed = urlparse(str(proxy_url))
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").strip("[]").lower()
    port = parsed.port
    return scheme, host, port


def _validate_tor_proxy_scheme(
    proxy_url: str,
    *,
    strict_tor_proxy_scheme: bool,
    allow_insecure_tor_proxy_scheme: bool,
):
    enforce_tor_dns_safe_proxy_url(
        proxy_url,
        tor_mode=True,
        context="Tor routing proxy URL",
        strict_tor_proxy_scheme=strict_tor_proxy_scheme,
        allow_insecure_tor_proxy_scheme=allow_insecure_tor_proxy_scheme,
    )


def _validate_custom_tor_session(
    session,
    *,
    expected_proxy: str,
    strict_tor_proxy_scheme: bool,
    allow_insecure_tor_proxy_scheme: bool,
):
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

    _validate_tor_proxy_scheme(
        http_proxy,
        strict_tor_proxy_scheme=strict_tor_proxy_scheme,
        allow_insecure_tor_proxy_scheme=allow_insecure_tor_proxy_scheme,
    )
    _validate_tor_proxy_scheme(
        https_proxy,
        strict_tor_proxy_scheme=strict_tor_proxy_scheme,
        allow_insecure_tor_proxy_scheme=allow_insecure_tor_proxy_scheme,
    )
    expected_parts = _normalized_proxy_parts(expected_proxy)
    for key, value in (("http", http_proxy), ("https", https_proxy)):
        if _normalized_proxy_parts(value) != expected_parts:
            raise TorConfigurationError(
                f"Custom Tor session {key} proxy does not match expected Tor proxy {expected_proxy!r}."
            )

    if strict_tor_proxy_scheme and bool(getattr(session, "trust_env", False)):
        raise TorConfigurationError(
            "Custom Tor session must set trust_env=False in strict mode to avoid environment proxy bypass."
        )


def tor_proxy_url(
    host: str = _DEFAULT_TOR_SOCKS_HOST,
    port: int = _DEFAULT_TOR_SOCKS_PORT,
    scheme: str = _DEFAULT_TOR_PROXY_SCHEME,
) -> str:
    return f"{scheme}://{host}:{int(port)}"


def tor_session(
    host: str = _DEFAULT_TOR_SOCKS_HOST,
    port: int = _DEFAULT_TOR_SOCKS_PORT,
    scheme: str = _DEFAULT_TOR_PROXY_SCHEME,
    user_agent: str | None = None,
    strict: bool = True,
    allow_insecure_tor_proxy_scheme: bool = False,
):
    return build_session(
        proxy_url=tor_proxy_url(host=host, port=port, scheme=scheme),
        user_agent=user_agent,
        tor_mode=True,
        strict_tor_proxy_scheme=bool(strict),
        allow_insecure_tor_proxy_scheme=bool(allow_insecure_tor_proxy_scheme),
    )


def tor_service(
    root: str,
    *,
    host: str = _DEFAULT_TOR_SOCKS_HOST,
    port: int = _DEFAULT_TOR_SOCKS_PORT,
    scheme: str = _DEFAULT_TOR_PROXY_SCHEME,
    session=None,
    allow_http_over_tor: bool = False,
    strict: bool = True,
    allow_insecure_tor_proxy_scheme: bool = False,
    **service_kwargs,
):
    """
    Build a Service routed through Tor.

    In strict mode, custom sessions are validated to require socks5h proxies and
    trust_env=False to prevent DNS/proxy bypass leaks.
    """
    from intermine314.service.service import Service

    proxy = tor_proxy_url(host=host, port=port, scheme=scheme)
    _validate_tor_proxy_scheme(
        proxy,
        strict_tor_proxy_scheme=bool(strict),
        allow_insecure_tor_proxy_scheme=bool(allow_insecure_tor_proxy_scheme),
    )
    tor_http_session = session or tor_session(
        host=host,
        port=port,
        scheme=scheme,
        strict=bool(strict),
        allow_insecure_tor_proxy_scheme=bool(allow_insecure_tor_proxy_scheme),
    )
    if strict and session is not None:
        _validate_custom_tor_session(
            tor_http_session,
            expected_proxy=proxy,
            strict_tor_proxy_scheme=bool(strict),
            allow_insecure_tor_proxy_scheme=bool(allow_insecure_tor_proxy_scheme),
        )
    service = Service(
        root,
        proxy_url=proxy,
        session=tor_http_session,
        tor=True,
        strict_tor_proxy_scheme=bool(strict),
        allow_insecure_tor_proxy_scheme=bool(allow_insecure_tor_proxy_scheme),
        allow_http_over_tor=bool(allow_http_over_tor),
        **service_kwargs,
    )
    if session is None:
        adopt = getattr(service, "_adopt_session_ownership", None)
        if callable(adopt):
            adopt()
    return service


def tor_registry(
    registry_url: str = _DEFAULT_REGISTRY_INSTANCES_URL,
    *,
    host: str = _DEFAULT_TOR_SOCKS_HOST,
    port: int = _DEFAULT_TOR_SOCKS_PORT,
    scheme: str = _DEFAULT_TOR_PROXY_SCHEME,
    request_timeout=_DEFAULT_REQUEST_TIMEOUT_SECONDS,
    session=None,
    verify_tls: bool = True,
    allow_http_over_tor: bool = False,
    max_cached_services=None,
    strict: bool = True,
    allow_insecure_tor_proxy_scheme: bool = False,
):
    """
    Build a Registry client routed through Tor.

    In strict mode, custom sessions are validated to require socks5h proxies and
    trust_env=False to prevent DNS/proxy bypass leaks.
    """
    from intermine314.service.service import Registry

    proxy = tor_proxy_url(host=host, port=port, scheme=scheme)
    _validate_tor_proxy_scheme(
        proxy,
        strict_tor_proxy_scheme=bool(strict),
        allow_insecure_tor_proxy_scheme=bool(allow_insecure_tor_proxy_scheme),
    )
    tor_http_session = session or tor_session(
        host=host,
        port=port,
        scheme=scheme,
        strict=bool(strict),
        allow_insecure_tor_proxy_scheme=bool(allow_insecure_tor_proxy_scheme),
    )
    if strict and session is not None:
        _validate_custom_tor_session(
            tor_http_session,
            expected_proxy=proxy,
            strict_tor_proxy_scheme=bool(strict),
            allow_insecure_tor_proxy_scheme=bool(allow_insecure_tor_proxy_scheme),
        )
    registry_kwargs = dict(
        registry_url=registry_url,
        request_timeout=request_timeout,
        proxy_url=proxy,
        session=tor_http_session,
        verify_tls=verify_tls,
        tor=True,
        strict_tor_proxy_scheme=bool(strict),
        allow_insecure_tor_proxy_scheme=bool(allow_insecure_tor_proxy_scheme),
        allow_http_over_tor=bool(allow_http_over_tor),
    )
    if max_cached_services is not None:
        registry_kwargs["max_cached_services"] = max_cached_services
    registry = Registry(**registry_kwargs)
    if session is None:
        adopt = getattr(registry, "_adopt_session_ownership", None)
        if callable(adopt):
            adopt()
    return registry
