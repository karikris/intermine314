#!/usr/bin/env python3
"""Run the live benchmark workflow using the canonical top-level benchmark entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import time
from typing import Iterable
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bench_targeting import get_target_defaults, load_target_config, normalize_target_settings
from benchmarks.benchmarks import DEFAULT_MINE_URL, main as benchmark_main, parse_args as benchmark_parse_args
from intermine314.service.tor import tor_proxy_url
from intermine314.service.transport import build_session
from intermine314.service.urls import normalize_service_root

SKIP_EXIT_CODE = 2
FAIL_EXIT_CODE = 1
SUCCESS_EXIT_CODE = 0
RUN_LIVE_ENV_VAR = "RUN_LIVE"
LIVE_MODE_ENV_VAR = "INTERMINE314_LIVE_MODE"
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 8.0
VALID_LIVE_MODES = ("direct", "tor", "both")
_SOCKS5H_PREFIX = "socks5h://"


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _truncate(text: str, limit: int = 200) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    return value[: limit - 12] + "...<truncated>"


def _runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--live-mode",
        choices=VALID_LIVE_MODES,
        default=os.getenv(LIVE_MODE_ENV_VAR, "direct"),
        help="Live preflight mode (direct, tor, both).",
    )
    parser.add_argument(
        "--live-preflight-timeout-seconds",
        type=float,
        default=DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
        help="Preflight request timeout in seconds.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight diagnostics only and exit without running benchmarks.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip live preflight checks and run benchmark workflow immediately.",
    )
    return parser


def _argv_context(args: list[str]):
    class _ArgvContext:
        def __enter__(self):
            self._original = list(sys.argv)
            sys.argv = [sys.argv[0]] + list(args)
            return self

        def __exit__(self, exc_type, exc, tb):
            sys.argv = self._original
            return False

    return _ArgvContext()


def _parse_benchmark_args(args: list[str]):
    with _argv_context(args):
        return benchmark_parse_args()


def _run_benchmark(args: list[str]) -> int:
    with _argv_context(args):
        try:
            result = benchmark_main()
        except SystemExit as exc:
            code = exc.code
            if code in (0, None):
                return SUCCESS_EXIT_CODE
            return FAIL_EXIT_CODE
        except Exception as exc:
            print(f"benchmark_failed reason=unexpected_exception err_type={type(exc).__name__}", flush=True)
            return FAIL_EXIT_CODE
    if result in (0, None):
        return SUCCESS_EXIT_CODE
    return FAIL_EXIT_CODE


def _normalize_live_mode(value: str) -> str:
    mode = str(value or "direct").strip().lower()
    if mode not in VALID_LIVE_MODES:
        return "direct"
    return mode


def _mode_sequence(mode: str) -> tuple[str, ...]:
    if mode == "both":
        return ("direct", "tor")
    return (mode,)


def _service_version_url(service_root: str) -> str:
    normalized = normalize_service_root(service_root)
    return normalized.rstrip("/") + "/version/ws"


def _candidate_urls(args) -> list[str]:
    target_config = load_target_config()
    target_defaults = get_target_defaults(target_config)
    target_settings = normalize_target_settings(args.benchmark_target, target_config, target_defaults)

    candidates: list[str] = []
    seen: set[str] = set()

    def _append(url: str | None):
        if url is None:
            return
        text = str(url).strip()
        if not text:
            return
        try:
            key = normalize_service_root(text)
        except Exception:
            key = text
        if key in seen:
            return
        seen.add(key)
        candidates.append(text)

    primary = str(args.mine_url).strip()
    if target_settings is not None:
        target_endpoint = str(target_settings.get("endpoint", "")).strip()
        if target_endpoint and primary == DEFAULT_MINE_URL:
            primary = target_endpoint
    _append(primary)
    if target_settings is not None:
        _append(target_settings.get("endpoint"))
        for fallback in target_settings.get("endpoint_fallbacks", []):
            _append(fallback)
    return candidates


def _print_probe(result: dict) -> None:
    print(
        "preflight "
        + f"mode={result['mode']} "
        + f"host={result['host']} "
        + f"reason={result['reason']} "
        + f"elapsed_s={result['elapsed_s']:.3f} "
        + f"err_type={result['err_type']}",
        flush=True,
    )


def _probe_direct(service_root: str, timeout_seconds: float) -> dict:
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
        with requests.get(_service_version_url(normalized), timeout=timeout_seconds, stream=True) as response:
            if int(response.status_code) >= 400:
                result["reason"] = "connect_failed"
                result["err_type"] = f"http_{int(response.status_code)}"
    except Exception as exc:
        result["reason"] = "connect_failed"
        result["err_type"] = type(exc).__name__
    result["elapsed_s"] = time.perf_counter() - t0
    return result


def _probe_tor(service_root: str, timeout_seconds: float) -> dict:
    t0 = time.perf_counter()
    normalized = normalize_service_root(service_root)
    parsed = urlparse(normalized)
    host = parsed.hostname or "<unknown>"
    result = {
        "mode": "tor",
        "host": host,
        "reason": "ok",
        "err_type": "none",
        "elapsed_s": 0.0,
    }
    proxy = tor_proxy_url()
    if not proxy.startswith(_SOCKS5H_PREFIX):
        result["reason"] = "proxy_failed"
        result["err_type"] = "non_socks5h_proxy"
        result["elapsed_s"] = time.perf_counter() - t0
        return result
    session = build_session(proxy_url=proxy, user_agent="intermine314-live-preflight")
    try:
        with session.get(_service_version_url(normalized), timeout=timeout_seconds, stream=True) as response:
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


def _probe_mode(mode: str, service_root: str, timeout_seconds: float) -> dict:
    if mode == "tor":
        return _probe_tor(service_root, timeout_seconds)
    return _probe_direct(service_root, timeout_seconds)


def _preflight(candidates: Iterable[str], mode: str, timeout_seconds: float) -> tuple[bool, str | None]:
    selected_modes = _mode_sequence(mode)
    for candidate in candidates:
        probe_results = [_probe_mode(m, candidate, timeout_seconds) for m in selected_modes]
        for result in probe_results:
            _print_probe(result)
        if all(result["reason"] == "ok" for result in probe_results):
            return True, candidate
    return False, None


def run(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    runner_args, bench_args = _runner_parser().parse_known_args(raw_args)

    if _is_truthy(os.getenv("CI")) and not _is_truthy(os.getenv(RUN_LIVE_ENV_VAR)):
        print(
            f"preflight_skip reason=ci_disabled env={RUN_LIVE_ENV_VAR} mode={_normalize_live_mode(runner_args.live_mode)}",
            flush=True,
        )
        return SKIP_EXIT_CODE

    if "--help" in bench_args or "-h" in bench_args:
        return _run_benchmark(bench_args)

    mode = _normalize_live_mode(runner_args.live_mode)
    timeout_seconds = max(1.0, float(runner_args.live_preflight_timeout_seconds))

    selected_candidate = None
    if not runner_args.skip_preflight:
        try:
            parsed = _parse_benchmark_args(bench_args)
            candidates = _candidate_urls(parsed)
            ok, selected_candidate = _preflight(candidates, mode=mode, timeout_seconds=timeout_seconds)
            if not ok:
                print(f"preflight_skip reason=environment mode={mode}", flush=True)
                return SKIP_EXIT_CODE
        except SystemExit as exc:
            return SUCCESS_EXIT_CODE if exc.code in (0, None) else FAIL_EXIT_CODE
        except Exception as exc:
            print(
                "preflight_skip "
                + f"reason=environment mode={mode} err_type={type(exc).__name__} err={_truncate(exc)}",
                flush=True,
            )
            return SKIP_EXIT_CODE

    if runner_args.preflight_only:
        return SUCCESS_EXIT_CODE

    benchmark_args = list(bench_args)
    if selected_candidate is not None:
        benchmark_args.extend(["--mine-url", selected_candidate])
    return _run_benchmark(benchmark_args)


if __name__ == "__main__":
    raise SystemExit(run())
