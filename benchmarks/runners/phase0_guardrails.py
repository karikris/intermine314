#!/usr/bin/env python3
"""Collect phase-0 guardrails for startup/import and Tor safety boundaries.

Baselines captured:
- import-time, imported-module count, tracemalloc peak, and max RSS for:
  - config surfaces
  - query builder
  - service transport
- Tor DNS-safety invariant behavior (socks5h accepted, socks5 rejected)

Exit codes:
- 0: success
- 1: failure
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bench_constants import DEFAULT_RUNNER_IMPORT_REPETITIONS
from benchmarks.runners.common import (
    now_utc_iso,
    run_import_baseline_subprocess,
    stat_summary,
    tor_proxy_observability_fields,
    validate_tor_proxy_url,
)
from benchmarks.runners.runner_metrics import (
    attach_metric_fields,
    measure_startup,
    proxy_url_scheme_from_url,
)

_STARTUP = measure_startup()

SUCCESS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1
DEFAULT_IMPORT_REPETITIONS = DEFAULT_RUNNER_IMPORT_REPETITIONS
PROFILE_NAME = "phase0_guardrails"

IMPORT_SURFACES: dict[str, str] = {
    "config_surface": (
        "import intermine314.config as _cfg;"
        "import intermine314.config.loader as _loader;"
        "import intermine314.config.runtime_defaults as _runtime_defaults"
    ),
    "query_builder": "import intermine314.query.builder as _query_builder",
    "service_transport": (
        "import intermine314.service.transport as _transport;"
        "import intermine314.service.session as _session"
    ),
}

_now_iso = now_utc_iso
_run_import_baseline_subprocess = partial(
    run_import_baseline_subprocess,
    source_root=SRC,
)


def _build_import_snippet(import_statement: str) -> str:
    return "\n".join(
        [
            "import json",
            "import resource",
            "import sys",
            "import time",
            "import tracemalloc",
            "",
            "def _rss_bytes():",
            "    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)",
            "    if rss <= 0:",
            "        return None",
            "    if sys.platform.startswith('linux'):",
            "        return rss * 1024",
            "    return rss",
            "",
            "before = len(sys.modules)",
            "tracemalloc.start()",
            "t0 = time.perf_counter()",
            str(import_statement),
            "elapsed = time.perf_counter() - t0",
            "_cur, peak = tracemalloc.get_traced_memory()",
            "after = len(sys.modules)",
            "payload = {",
            "    'seconds': elapsed,",
            "    'module_count': after,",
            "    'imported_module_count': after - before,",
            "    'tracemalloc_peak_bytes': peak,",
            "    'max_rss_bytes': _rss_bytes(),",
            "}",
            "print(json.dumps(payload))",
        ]
    )


def _numeric_summary(runs: list[dict[str, Any]], field: str) -> dict[str, float] | None:
    values: list[float] = []
    for run in runs:
        value = run.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return stat_summary(values)


def _collect_import_surface_baseline(name: str, import_statement: str, repetitions: int) -> dict[str, Any]:
    baseline = _run_import_baseline_subprocess(
        import_snippet=_build_import_snippet(import_statement),
        repetitions=int(repetitions),
    )
    runs = [run for run in baseline.get("runs", []) if isinstance(run, dict)]
    enriched = dict(baseline)
    imported_count_summary = _numeric_summary(runs, "imported_module_count")
    if imported_count_summary is not None:
        enriched["imported_module_count"] = imported_count_summary
    max_rss_summary = _numeric_summary(runs, "max_rss_bytes")
    if max_rss_summary is not None:
        enriched["max_rss_bytes"] = max_rss_summary
    enriched["surface"] = str(name)
    return enriched


def _tor_safety_guardrail() -> dict[str, Any]:
    safe_proxy = "socks5h://127.0.0.1:9050"
    unsafe_proxy = "socks5://127.0.0.1:9050"
    normalized_safe = validate_tor_proxy_url(
        safe_proxy,
        context="phase0 guardrails safe proxy",
    )
    unsafe_rejected = False
    unsafe_error_type = "none"
    try:
        validate_tor_proxy_url(
            unsafe_proxy,
            context="phase0 guardrails unsafe proxy",
        )
    except Exception as exc:
        unsafe_rejected = True
        unsafe_error_type = type(exc).__name__
    payload = {
        "safe_proxy_input": safe_proxy,
        "safe_proxy_normalized": normalized_safe,
        "safe_proxy_scheme": proxy_url_scheme_from_url(normalized_safe),
        "unsafe_proxy_input": unsafe_proxy,
        "unsafe_proxy_rejected": unsafe_rejected,
        "unsafe_proxy_rejection_error_type": unsafe_error_type,
    }
    payload.update(tor_proxy_observability_fields(normalized_safe))
    return payload


def _build_report(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    import_guardrails: dict[str, Any] = {}
    for name, import_statement in IMPORT_SURFACES.items():
        import_guardrails[name] = _collect_import_surface_baseline(
            name,
            import_statement,
            args.import_repetitions,
        )
    tor_safety = _tor_safety_guardrail()
    tor_safety_ok = bool(tor_safety.get("unsafe_proxy_rejected"))
    status = "ok" if tor_safety_ok else "failed"
    error_type = "none" if tor_safety_ok else "tor_dns_safety_not_enforced"
    report: dict[str, Any] = {
        "timestamp_utc": _now_iso(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "runtime": {
            "import_repetitions": int(args.import_repetitions),
            "surfaces": list(IMPORT_SURFACES.keys()),
        },
        "import_guardrails": import_guardrails,
        "tor_safety": tor_safety,
    }
    attach_metric_fields(
        report,
        startup=_STARTUP,
        status=status,
        error_type=error_type,
        tor_mode="disabled",
        proxy_url_scheme=str(tor_safety.get("tor_proxy_scheme", "none")),
        profile_name=PROFILE_NAME,
    )
    return (SUCCESS_EXIT_CODE if status == "ok" else FAIL_EXIT_CODE), report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-repetitions", type=int, default=DEFAULT_IMPORT_REPETITIONS)
    parser.add_argument("--json-out", default=None)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.import_repetitions) <= 0:
        raise ValueError("--import-repetitions must be a positive integer")
    code, report = _build_report(args)
    output = json.dumps(report, sort_keys=True, default=str)
    if args.json_out:
        path = Path(str(args.json_out))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
    print(output, flush=True)
    return code


def main() -> int:
    try:
        return run()
    except Exception as exc:
        payload = {"status": "failed", "error": str(exc), "error_type": type(exc).__name__}
        attach_metric_fields(
            payload,
            startup=_STARTUP,
            status="failed",
            error_type=type(exc).__name__,
            tor_mode="disabled",
            proxy_url_scheme="none",
            profile_name=PROFILE_NAME,
        )
        print(json.dumps(payload, sort_keys=True), flush=True)
        return FAIL_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
