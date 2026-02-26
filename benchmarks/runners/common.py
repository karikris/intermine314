from __future__ import annotations

import json
import os
import resource
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from benchmarks.runners.runner_metrics import proxy_url_scheme_from_url
from intermine314.service.errors import TorConfigurationError
from intermine314.service.tor import tor_proxy_url
from intermine314.service.transport import (
    build_session,
    enforce_tor_dns_safe_proxy_url,
)
from intermine314.service.urls import normalize_service_root

DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 120.0


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    result = {
        "n": float(len(values)),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }
    if len(values) > 1:
        result["stddev"] = statistics.stdev(values)
    else:
        result["stddev"] = 0.0
    return result


def pythonpath_env(*, source_root: Path | str) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    entries = [str(source_root)]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def ru_maxrss_bytes() -> int | None:
    try:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if rss <= 0:
        return None
    if sys.platform.startswith("linux"):
        return rss * 1024
    return rss


def run_subprocess_json_line(
    *,
    cmd: list[str],
    env: dict[str, str],
    timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
    error_context: str,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{error_context} timed out after {float(timeout_seconds):.1f}s") from exc
    except Exception as exc:
        raise RuntimeError(f"{error_context} failed to start: {type(exc).__name__}") from exc

    if proc.returncode != 0:
        stderr = str(proc.stderr or "").strip()
        raise RuntimeError(f"{error_context} failed with rc={proc.returncode}: {stderr}")

    stdout = str(proc.stdout or "").strip()
    if not stdout:
        raise RuntimeError(f"{error_context} emitted empty stdout")
    line = stdout.splitlines()[-1]
    try:
        payload = json.loads(line)
    except Exception as exc:
        raise RuntimeError(f"{error_context} emitted invalid JSON line: {line[:200]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{error_context} emitted non-object JSON payload")
    return payload


def run_import_baseline_subprocess(
    *,
    import_snippet: str,
    repetitions: int,
    source_root: Path | str,
    python_executable: str | None = None,
    timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
    attempts: int = 1,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    command = [str(python_executable or sys.executable), "-c", str(import_snippet)]
    retries = max(1, int(attempts))
    for _ in range(int(repetitions)):
        last_error: Exception | None = None
        for _attempt in range(retries):
            try:
                payload = run_subprocess_json_line(
                    cmd=command,
                    env=pythonpath_env(source_root=source_root),
                    timeout_seconds=timeout_seconds,
                    error_context="import baseline subprocess",
                )
                runs.append(payload)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise RuntimeError(str(last_error))
    seconds = [float(item["seconds"]) for item in runs]
    peaks = [float(item["tracemalloc_peak_bytes"]) for item in runs]
    module_counts = [float(item["module_count"]) for item in runs]
    return {
        "repetitions": int(repetitions),
        "runs": runs,
        "seconds": stat_summary(seconds),
        "tracemalloc_peak_bytes": stat_summary(peaks),
        "module_count": stat_summary(module_counts),
    }


def service_version_url(service_root: str) -> str:
    normalized = normalize_service_root(service_root)
    return normalized.rstrip("/") + "/version/ws"


def validate_tor_proxy_url(proxy_url: str | None, *, context: str) -> str:
    normalized = enforce_tor_dns_safe_proxy_url(
        proxy_url,
        tor_mode=True,
        context=context,
    )
    if normalized is None or not str(normalized).strip():
        raise TorConfigurationError(f"{context} must define a proxy URL when Tor mode is enabled.")
    return str(normalized)


def tor_proxy_observability_fields(proxy_url: str | None) -> dict[str, str]:
    return {
        "proxy_url": "" if proxy_url is None else str(proxy_url),
        "tor_proxy_scheme": proxy_url_scheme_from_url(proxy_url),
        "tor_dns_safety": "enforced",
    }


def probe_direct(service_root: str, timeout_seconds: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    normalized = normalize_service_root(service_root)
    parsed = urlparse(normalized)
    host = parsed.hostname or "<unknown>"
    scheme = (parsed.scheme or "").lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    result = {
        "mode": "direct",
        "host": host,
        "reason": "ok",
        "err_type": "none",
        "elapsed_s": 0.0,
    }
    try:
        socket.getaddrinfo(host, port)
    except Exception as exc:
        result["reason"] = "dns_failed"
        result["err_type"] = type(exc).__name__
        result["elapsed_s"] = time.perf_counter() - t0
        return result

    try:
        with requests.get(service_version_url(normalized), timeout=timeout_seconds, stream=True) as response:
            if int(response.status_code) >= 400:
                result["reason"] = "connect_failed"
                result["err_type"] = f"http_{int(response.status_code)}"
    except Exception as exc:
        result["reason"] = "connect_failed"
        result["err_type"] = type(exc).__name__
    result["elapsed_s"] = time.perf_counter() - t0
    return result


def probe_tor(
    service_root: str,
    timeout_seconds: float,
    *,
    context: str,
    user_agent: str,
    proxy_url: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    normalized = normalize_service_root(service_root)
    host = urlparse(normalized).hostname or "<unknown>"
    proxy = tor_proxy_url() if proxy_url is None else str(proxy_url)
    result: dict[str, Any] = {
        "mode": "tor",
        "host": host,
        "reason": "ok",
        "err_type": "none",
        "elapsed_s": 0.0,
    }
    result.update(tor_proxy_observability_fields(proxy))
    try:
        safe_proxy = validate_tor_proxy_url(proxy, context=context)
    except Exception as exc:
        result["reason"] = "proxy_failed"
        result["err_type"] = type(exc).__name__
        result["tor_dns_safety"] = "rejected"
        result["elapsed_s"] = time.perf_counter() - t0
        return result

    session = build_session(proxy_url=safe_proxy, user_agent=user_agent, tor_mode=True)
    try:
        with session.get(service_version_url(normalized), timeout=timeout_seconds, stream=True) as response:
            if int(response.status_code) >= 400:
                result["reason"] = "connect_failed"
                result["err_type"] = f"http_{int(response.status_code)}"
    except Exception as exc:
        result["reason"] = "proxy_failed"
        result["err_type"] = type(exc).__name__
    finally:
        try:
            session.close()
        except Exception:
            pass
    result["elapsed_s"] = time.perf_counter() - t0
    return result
