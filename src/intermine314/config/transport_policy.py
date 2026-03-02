from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HTTPRetryPolicy:
    total: int
    backoff_factor: float
    status_forcelist: tuple[int, ...]
    allowed_methods: tuple[str, ...]


def resolve_http_retry_policy() -> HTTPRetryPolicy:
    from intermine314.config.runtime_defaults import get_runtime_defaults

    defaults = get_runtime_defaults().transport_defaults
    return HTTPRetryPolicy(
        total=int(defaults.default_http_retry_total),
        backoff_factor=float(defaults.default_http_retry_backoff_seconds),
        status_forcelist=tuple(int(code) for code in defaults.default_http_retry_status_codes),
        allowed_methods=tuple(str(method).upper() for method in defaults.default_http_retry_methods),
    )
