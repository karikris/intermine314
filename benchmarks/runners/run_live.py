#!/usr/bin/env python3
"""Run the live benchmark workflow using the canonical top-level benchmark entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bench_constants import (
    DEFAULT_BENCHMARK_MINE_URL,
    DEFAULT_RUNNER_PREFLIGHT_TIMEOUT_SECONDS,
)
from benchmarks.runners.runner_metrics import (
    attach_metric_fields,
    measure_startup,
    proxy_url_scheme_from_url,
)

from benchmarks.bench_targeting import get_target_defaults, load_target_config, normalize_target_settings
from benchmarks.runners.common import probe_direct, probe_tor
from intermine314.service.tor import tor_proxy_url
from intermine314.service.urls import normalize_service_root

_STARTUP = measure_startup()

SKIP_EXIT_CODE = 2
FAIL_EXIT_CODE = 1
SUCCESS_EXIT_CODE = 0
RUN_LIVE_ENV_VAR = "RUN_LIVE"
LIVE_MODE_ENV_VAR = "INTERMINE314_LIVE_MODE"
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = DEFAULT_RUNNER_PREFLIGHT_TIMEOUT_SECONDS
VALID_LIVE_MODES = ("direct", "tor", "both")

_probe_direct = probe_direct
_probe_tor = lambda service_root, timeout_seconds: probe_tor(  # noqa: E731
    service_root,
    timeout_seconds,
    context="run_live preflight proxy_url",
    user_agent="intermine314-live-preflight",
    proxy_url=tor_proxy_url(),
)


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


def _benchmark_module():
    from benchmarks import benchmarks as benchmark_module

    return benchmark_module


def _parse_benchmark_args(args: list[str]):
    return _benchmark_module().parse_args(args)


def _run_benchmark(args: list[str]) -> int:
    try:
        result = _benchmark_module().main(args)
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


def _mode_proxy_url_scheme(mode: str) -> str:
    if str(mode).strip().lower() in {"tor", "both"}:
        return proxy_url_scheme_from_url(tor_proxy_url())
    return "none"


def _status_from_exit_code(code: int) -> str:
    if int(code) == SUCCESS_EXIT_CODE:
        return "ok"
    if int(code) == SKIP_EXIT_CODE:
        return "skipped"
    return "failed"


def _profile_name_from_benchmark_args(parsed) -> str:
    value = getattr(parsed, "benchmark_profile", None)
    if value is None:
        return "auto"
    text = str(value).strip()
    if not text:
        return "auto"
    return text


def _finalize(
    *,
    exit_code: int,
    status: str,
    error_type: str,
    tor_mode: str,
    proxy_url_scheme: str,
    profile_name: str,
) -> int:
    payload: dict[str, object] = {}
    attach_metric_fields(
        payload,
        startup=_STARTUP,
        status=status,
        error_type=error_type,
        tor_mode=tor_mode,
        proxy_url_scheme=proxy_url_scheme,
        profile_name=profile_name,
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return int(exit_code)


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
        if target_endpoint and primary == DEFAULT_BENCHMARK_MINE_URL:
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
    mode = _normalize_live_mode(runner_args.live_mode)
    proxy_url_scheme = _mode_proxy_url_scheme(mode)
    profile_name = "auto"

    if _is_truthy(os.getenv("CI")) and not _is_truthy(os.getenv(RUN_LIVE_ENV_VAR)):
        print(
            f"preflight_skip reason=ci_disabled env={RUN_LIVE_ENV_VAR} mode={mode}",
            flush=True,
        )
        return _finalize(
            exit_code=SKIP_EXIT_CODE,
            status="skipped",
            error_type="ci_disabled",
            tor_mode=mode,
            proxy_url_scheme=proxy_url_scheme,
            profile_name=profile_name,
        )

    if "--help" in bench_args or "-h" in bench_args:
        code = _run_benchmark(bench_args)
        return _finalize(
            exit_code=code,
            status=_status_from_exit_code(code),
            error_type="none" if code == SUCCESS_EXIT_CODE else "benchmark_failed",
            tor_mode=mode,
            proxy_url_scheme=proxy_url_scheme,
            profile_name=profile_name,
        )

    timeout_seconds = max(1.0, float(runner_args.live_preflight_timeout_seconds))

    selected_candidate = None
    if not runner_args.skip_preflight:
        try:
            parsed = _parse_benchmark_args(bench_args)
            profile_name = _profile_name_from_benchmark_args(parsed)
            candidates = _candidate_urls(parsed)
            ok, selected_candidate = _preflight(candidates, mode=mode, timeout_seconds=timeout_seconds)
            if not ok:
                print(f"preflight_skip reason=environment mode={mode}", flush=True)
                return _finalize(
                    exit_code=SKIP_EXIT_CODE,
                    status="skipped",
                    error_type="preflight_environment",
                    tor_mode=mode,
                    proxy_url_scheme=proxy_url_scheme,
                    profile_name=profile_name,
                )
        except SystemExit as exc:
            code = SUCCESS_EXIT_CODE if exc.code in (0, None) else FAIL_EXIT_CODE
            return _finalize(
                exit_code=code,
                status=_status_from_exit_code(code),
                error_type="none" if code == SUCCESS_EXIT_CODE else "benchmark_argparse_exit",
                tor_mode=mode,
                proxy_url_scheme=proxy_url_scheme,
                profile_name=profile_name,
            )
        except Exception as exc:
            print(
                "preflight_skip "
                + f"reason=environment mode={mode} err_type={type(exc).__name__} err={_truncate(exc)}",
                flush=True,
            )
            return _finalize(
                exit_code=SKIP_EXIT_CODE,
                status="skipped",
                error_type=type(exc).__name__,
                tor_mode=mode,
                proxy_url_scheme=proxy_url_scheme,
                profile_name=profile_name,
            )

    if runner_args.preflight_only:
        return _finalize(
            exit_code=SUCCESS_EXIT_CODE,
            status="ok",
            error_type="none",
            tor_mode=mode,
            proxy_url_scheme=proxy_url_scheme,
            profile_name=profile_name,
        )

    benchmark_args = list(bench_args)
    if selected_candidate is not None:
        benchmark_args.extend(["--mine-url", selected_candidate])
    code = _run_benchmark(benchmark_args)
    return _finalize(
        exit_code=code,
        status=_status_from_exit_code(code),
        error_type="none" if code == SUCCESS_EXIT_CODE else "benchmark_failed",
        tor_mode=mode,
        proxy_url_scheme=proxy_url_scheme,
        profile_name=profile_name,
    )


if __name__ == "__main__":
    raise SystemExit(run())
