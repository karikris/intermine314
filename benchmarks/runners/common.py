from __future__ import annotations

import json
import os
import resource
import socket
import statistics
import subprocess
import sys
import threading
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


def open_socket_count() -> int | None:
    proc_fd = Path("/proc/self/fd")
    if proc_fd.exists() and proc_fd.is_dir():
        count = 0
        try:
            for fd_path in proc_fd.iterdir():
                try:
                    target = os.readlink(fd_path)
                except Exception:
                    continue
                if str(target).startswith("socket:["):
                    count += 1
            return int(count)
        except Exception:
            return None
    return None


class SocketMonitor:
    def __init__(self, *, sample_interval_seconds: float = 0.1) -> None:
        self.sample_interval_seconds = max(0.01, float(sample_interval_seconds))
        self.start_open_sockets: int | None = None
        self.end_open_sockets: int | None = None
        self.peak_open_sockets: int | None = None
        self.sample_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        count = open_socket_count()
        if count is None:
            return
        self.sample_count += 1
        if self.start_open_sockets is None:
            self.start_open_sockets = int(count)
        if self.peak_open_sockets is None or int(count) > int(self.peak_open_sockets):
            self.peak_open_sockets = int(count)
        self.end_open_sockets = int(count)

    def _run_sampler(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._sample_once()

    def __enter__(self) -> "SocketMonitor":
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run_sampler,
            name="intermine314-socket-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.2, self.sample_interval_seconds * 2.0))
        self._sample_once()

    def as_dict(self) -> dict[str, Any]:
        delta = None
        if self.start_open_sockets is not None and self.end_open_sockets is not None:
            delta = int(self.end_open_sockets) - int(self.start_open_sockets)
        return {
            "start_open_sockets": self.start_open_sockets,
            "end_open_sockets": self.end_open_sockets,
            "peak_open_sockets": self.peak_open_sockets,
            "delta_open_sockets": delta,
            "sample_count": int(self.sample_count),
            "measurement_supported": self.start_open_sockets is not None,
        }


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
    out: dict[str, Any] = {
        "repetitions": int(repetitions),
        "runs": runs,
    }
    numeric_fields = (
        "seconds",
        "tracemalloc_peak_bytes",
        "module_count",
        "max_rss_bytes",
        "imported_module_count",
    )
    for field in numeric_fields:
        values: list[float] = []
        for item in runs:
            value = item.get(field)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            out[field] = stat_summary(values)
    return out


def stable_import_baseline_metrics(
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = baseline if isinstance(baseline, dict) else {}

    def _summary_mean(field: str) -> float | None:
        value = payload.get(field)
        if isinstance(value, dict):
            mean_value = value.get("mean")
            if isinstance(mean_value, (int, float)):
                return float(mean_value)
        return None

    return {
        "import_repetitions": int(payload.get("repetitions", 0) or 0),
        "import_time_seconds_mean": _summary_mean("seconds"),
        "import_time_seconds_median": (
            float(payload.get("seconds", {}).get("median"))
            if isinstance(payload.get("seconds"), dict)
            and isinstance(payload.get("seconds", {}).get("median"), (int, float))
            else None
        ),
        "imported_module_count_mean": _summary_mean("imported_module_count"),
        "module_count_mean": _summary_mean("module_count"),
        "tracemalloc_peak_bytes_mean": _summary_mean("tracemalloc_peak_bytes"),
        "max_rss_bytes_mean": _summary_mean("max_rss_bytes"),
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
