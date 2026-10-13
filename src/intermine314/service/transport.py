from __future__ import annotations

import os
import threading
import warnings
import weakref
from contextvars import ContextVar
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from intermine314.config.runtime_defaults import (
    KNOWN_TOR_SOCKS_PORTS,
    TOR_DNS_SAFE_PROXY_SCHEME,
    VALID_TOR_PROXY_SCHEMES,
    get_runtime_defaults,
)
from intermine314.config.transport_policy import resolve_http_retry_policy

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import requests

__all__ = [
    "PROXY_URL_ENV_VAR",
    "TOR_DNS_SAFE_PROXY_SCHEME",
    "TOR_PROXY_SCHEMES",
    "default_tor_proxy_url",
    "resolve_proxy_url",
    "is_tor_proxy_url",
    "allowed_tor_proxy_schemes",
    "enforce_tor_dns_safe_proxy_url",
    "build_session",
]


PROXY_URL_ENV_VAR = "INTERMINE314_PROXY_URL"
_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
TOR_PROXY_SCHEMES = VALID_TOR_PROXY_SCHEMES
_SESSION_LOCKS_GUARD = threading.Lock()
_SESSION_LOCKS = weakref.WeakValueDictionary()


class _SessionLock:
    def __init__(self):
        self.lock = threading.RLock()


def borrowed_session_lock(session):
    """Share a lock across openers without mutating a caller's session."""
    with _SESSION_LOCKS_GUARD:
        state = _SESSION_LOCKS.get(id(session))
        if state is None:
            state = _SESSION_LOCKS[id(session)] = _SessionLock()
        return state


def default_tor_proxy_url(
    *,
    host: str | None = None,
    port: int | None = None,
    scheme: str | None = None,
) -> str:
    defaults = get_runtime_defaults().service_defaults
    resolved_host = str(defaults.default_tor_socks_host) if host is None else str(host)
    resolved_port = int(defaults.default_tor_socks_port) if port is None else int(port)
    resolved_scheme = str(defaults.default_tor_proxy_scheme) if scheme is None else str(scheme)
    return f"{resolved_scheme}://{resolved_host}:{resolved_port}"


def _requests_runtime():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    return requests, HTTPAdapter, Retry


def resolve_proxy_url(proxy_url: str | None = None) -> str | None:
    if proxy_url is not None:
        value = str(proxy_url).strip()
        return value or None
    env_value = os.getenv(PROXY_URL_ENV_VAR, "").strip()
    return env_value or None


def is_tor_proxy_url(proxy_url: str | None = None) -> bool:
    resolved = resolve_proxy_url(proxy_url)
    if not resolved:
        return False
    try:
        parsed = urlparse(resolved)
    except Exception:
        return False

    if (parsed.scheme or "").lower() not in TOR_PROXY_SCHEMES:
        return False

    host = (parsed.hostname or "").strip("[]").lower()
    if host not in _LOCALHOST_HOSTS:
        return False

    if parsed.port is None:
        return True
    return int(parsed.port) in KNOWN_TOR_SOCKS_PORTS


def allowed_tor_proxy_schemes(
    *,
    strict_tor_proxy_scheme: bool = True,
    allow_insecure_tor_proxy_scheme: bool = False,
) -> frozenset[str]:
    if bool(allow_insecure_tor_proxy_scheme):
        return TOR_PROXY_SCHEMES
    if bool(strict_tor_proxy_scheme):
        return frozenset({TOR_DNS_SAFE_PROXY_SCHEME})
    return TOR_PROXY_SCHEMES


def enforce_tor_dns_safe_proxy_url(
    proxy_url: str | None,
    *,
    tor_mode: bool,
    context: str = "Tor proxy URL",
    strict_tor_proxy_scheme: bool = True,
    allow_insecure_tor_proxy_scheme: bool = False,
) -> str | None:
    resolved = resolve_proxy_url(proxy_url)
    if not tor_mode or not resolved:
        return resolved
    try:
        scheme = (urlparse(resolved).scheme or "").lower()
    except Exception:
        scheme = ""
    allowed_schemes = allowed_tor_proxy_schemes(
        strict_tor_proxy_scheme=strict_tor_proxy_scheme,
        allow_insecure_tor_proxy_scheme=allow_insecure_tor_proxy_scheme,
    )
    if scheme not in allowed_schemes:
        from intermine314.service.errors import TorConfigurationError

        allowed_text = ", ".join(f"{value}://" for value in sorted(allowed_schemes))
        raise TorConfigurationError(
            f"{context} must use one of {allowed_text} when Tor mode is enabled. "
            f"Got: {resolved!r}. Use {TOR_DNS_SAFE_PROXY_SCHEME}:// to avoid DNS leaks."
        )
    if scheme != TOR_DNS_SAFE_PROXY_SCHEME:
        warnings.warn(
            f"{context} uses non-DNS-safe proxy scheme {scheme}:// in Tor mode. "
            f"Use {TOR_DNS_SAFE_PROXY_SCHEME}:// to avoid DNS leaks.",
            RuntimeWarning,
            stacklevel=3,
        )
    return resolved


def build_session(
    *,
    proxy_url: str | None,
    user_agent: str | None = None,
    tor_mode: bool = False,
    strict_tor_proxy_scheme: bool = True,
    allow_insecure_tor_proxy_scheme: bool = False,
) -> requests.Session:
    requests, HTTPAdapter, Retry = _requests_runtime()
    proxy_url = enforce_tor_dns_safe_proxy_url(
        proxy_url,
        tor_mode=bool(tor_mode),
        context="build_session proxy_url",
        strict_tor_proxy_scheme=bool(strict_tor_proxy_scheme),
        allow_insecure_tor_proxy_scheme=bool(allow_insecure_tor_proxy_scheme),
    )
    class ManagedSession(requests.Session):
        """Select immutable retry policies per request, including concurrent requests."""

        def __init__(self):
            super().__init__()
            self._retry_safe = ContextVar("intermine314_retry_safe", default=False)
            self._read_adapter = None
            self._transport_closed = False
            self._worker_sessions = {}
            self._worker_lock = threading.RLock()
            self._is_worker_session = False
            self._cookie_sensitive = False

        def _request_with_retry_policy(self, *args, retry_safe=False, **kwargs):
            return self.request(*args, retry_safe=retry_safe, **kwargs)

        def request(self, *args, retry_safe=False, **kwargs):
            from intermine314.export.runtime import in_http_worker

            if in_http_worker() and not self._is_worker_session:
                with self._worker_lock:
                    # Cookie-dependent or custom-auth flows keep a single owner.
                    if (self._cookie_sensitive or self.cookies or self.auth is not None
                            or any(self.hooks.values())
                            or any(adapter is not self._write_adapter for adapter in self.adapters.values())):
                        return self._request_owned(*args, retry_safe=retry_safe, **kwargs)
                    for old_thread in list(self._worker_sessions):
                        if not old_thread.is_alive():
                            self._worker_sessions.pop(old_thread).close()
                    thread = threading.current_thread()
                    worker = self._worker_sessions.get(thread)
                    if worker is None:
                        worker = build_session(proxy_url=proxy_url, user_agent=user_agent,
                                               tor_mode=tor_mode,
                                               strict_tor_proxy_scheme=strict_tor_proxy_scheme,
                                               allow_insecure_tor_proxy_scheme=allow_insecure_tor_proxy_scheme)
                        worker._is_worker_session = True
                        self._worker_sessions[thread] = worker
                    # Snapshot configuration for this request. Worker adapters and
                    # jars stay private; configuration changes between jobs apply.
                    worker.headers = self.headers.copy()
                    worker.proxies = self.proxies.copy()
                    worker.params = self.params.copy()
                    worker.verify, worker.cert = self.verify, self.cert
                    worker.trust_env, worker.stream = self.trust_env, self.stream
                response = worker._request_owned(*args, retry_safe=retry_safe, **kwargs)
                if worker.cookies:
                    with self._worker_lock:
                        self.cookies.update(worker.cookies)
                        self._cookie_sensitive = True
                return response
            return self._request_owned(*args, retry_safe=retry_safe, **kwargs)

        def _request_owned(self, *args, retry_safe=False, **kwargs):
            self._transport_closed = False
            token = self._retry_safe.set(bool(retry_safe))
            try:
                return super().request(*args, **kwargs)
            finally:
                self._retry_safe.reset(token)

        def get_adapter(self, url):
            adapter = super().get_adapter(url)
            if self._retry_safe.get() and adapter is self._write_adapter:
                return self._read_adapter
            return adapter

        def close(self):
            if self._transport_closed:
                return
            self._transport_closed = True
            with self._worker_lock:
                workers, self._worker_sessions = self._worker_sessions, {}
            for worker in workers.values():
                worker.close()
            adapters = set(self.adapters.values())
            if self._read_adapter is not None:
                adapters.add(self._read_adapter)
            for adapter in adapters:
                adapter.close()

    session = ManagedSession()
    if proxy_url:
        session.proxies = {"http": proxy_url, "https": proxy_url}
        session.trust_env = False

    retry_policy = resolve_http_retry_policy()
    retries = Retry(
        total=retry_policy.total,
        backoff_factor=retry_policy.backoff_factor,
        status_forcelist=retry_policy.status_forcelist,
        allowed_methods=retry_policy.allowed_methods,
        raise_on_status=False,
    )
    session._read_adapter = HTTPAdapter(max_retries=retries, pool_connections=32, pool_maxsize=32)
    adapter = HTTPAdapter(max_retries=Retry(total=0), pool_connections=32, pool_maxsize=32)
    session._write_adapter = adapter
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if user_agent:
        session.headers.update({"User-Agent": user_agent})
    return session


def open_readonly(opener, *args, **kwargs):
    """Mark managed reads without changing the protocol of custom openers."""
    read = getattr(opener, "_open_readonly", None)
    return read(*args, **kwargs) if callable(read) else opener.open(*args, **kwargs)
