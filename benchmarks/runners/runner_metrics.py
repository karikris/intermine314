from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class StartupMeasurement:
    started_at: float


_PROCESS_STARTUP = StartupMeasurement(started_at=time.perf_counter())


def measure_startup() -> StartupMeasurement:
    return _PROCESS_STARTUP


def max_rss_bytes() -> int | None:
    try:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if rss <= 0:
        return None
    if sys.platform.startswith("linux"):
        return rss * 1024
    return rss


def proxy_url_scheme_from_url(proxy_url: str | None) -> str:
    if proxy_url is None:
        return "none"
    text = str(proxy_url).strip()
    if not text:
        return "none"
    try:
        scheme = (urlparse(text).scheme or "").strip().lower()
    except Exception:
        return "unknown"
    if not scheme:
        return "unknown"
    return scheme


def metric_fields(
    *,
    startup: StartupMeasurement,
    status: str,
    error_type: str | None,
    tor_mode: str | None,
    proxy_url_scheme: str | None,
    profile_name: str | None,
) -> dict[str, object]:
    elapsed_ms = max(0.0, (time.perf_counter() - float(startup.started_at)) * 1000.0)
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "max_rss_bytes": max_rss_bytes(),
        "status": str(status),
        "error_type": str(error_type or "none"),
        "tor_mode": str(tor_mode or "disabled"),
        "proxy_url_scheme": str(proxy_url_scheme or "none"),
        "profile_name": str(profile_name or "unknown"),
    }


def attach_metric_fields(
    payload: dict[str, object],
    *,
    startup: StartupMeasurement,
    status: str,
    error_type: str | None,
    tor_mode: str | None,
    proxy_url_scheme: str | None,
    profile_name: str | None,
) -> dict[str, object]:
    payload.update(
        metric_fields(
            startup=startup,
            status=status,
            error_type=error_type,
            tor_mode=tor_mode,
            proxy_url_scheme=proxy_url_scheme,
            profile_name=profile_name,
        )
    )
    return payload
