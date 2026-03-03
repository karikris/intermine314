#!/usr/bin/env python3
"""Freeze minimal runtime contract and capture phase-0 baseline artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bench_constants import DEFAULT_RUNNER_IMPORT_REPETITIONS
from benchmarks.runners.common import now_utc_iso, run_import_baseline_subprocess
from benchmarks.runners.runner_metrics import (
    attach_metric_fields,
    measure_startup,
    proxy_url_scheme_from_url,
)

_STARTUP = measure_startup()

SUCCESS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1
PROFILE_NAME = "phase0_contract_baseline"
DEFAULT_CONTRACT_FILE = ROOT / "benchmarks" / "contracts" / "minimal_public_contract.json"


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"contract payload must be a JSON object: {path}")
    return payload


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_import_snippet(module_name: str) -> str:
    return "\n".join(
        [
            "import hashlib",
            "import importlib",
            "import json",
            "import resource",
            "import sys",
            "import time",
            "import tracemalloc",
            f"target = {module_name!r}",
            "",
            "def _rss_bytes():",
            "    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)",
            "    if rss <= 0:",
            "        return None",
            "    if sys.platform.startswith('linux'):",
            "        return rss * 1024",
            "    return rss",
            "",
            "before = set(sys.modules)",
            "before_count = len(before)",
            "tracemalloc.start()",
            "t0 = time.perf_counter()",
            "importlib.import_module(target)",
            "elapsed = time.perf_counter() - t0",
            "_cur, peak = tracemalloc.get_traced_memory()",
            "after = set(sys.modules)",
            "imported = sorted(after - before)",
            "payload = {",
            "    'module': target,",
            "    'seconds': elapsed,",
            "    'module_count': len(after),",
            "    'imported_module_count': len(imported),",
            "    'tracemalloc_peak_bytes': peak,",
            "    'max_rss_bytes': _rss_bytes(),",
            "    'import_graph_hash': hashlib.sha256('\\n'.join(imported).encode('utf-8')).hexdigest(),",
            "    'import_graph_sample': imported[:64],",
            "}",
            "print(json.dumps(payload))",
        ]
    )


def _float_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "n": float(n),
        "mean": sum(ordered) / float(n),
        "min": ordered[0],
        "max": ordered[-1],
        "median": float(median),
    }


def _collect_hotspot_imports(modules: list[str], repetitions: int) -> tuple[dict[str, Any], bool]:
    surfaces: dict[str, Any] = {}
    ok = True
    for module_name in modules:
        baseline = run_import_baseline_subprocess(
            import_snippet=_build_import_snippet(module_name),
            repetitions=int(repetitions),
            source_root=SRC,
        )
        runs = [run for run in baseline.get("runs", []) if isinstance(run, dict)]
        graph_hashes = sorted(
            {
                str(run.get("import_graph_hash"))
                for run in runs
                if isinstance(run.get("import_graph_hash"), str) and str(run.get("import_graph_hash")).strip()
            }
        )
        import_time_seconds = [
            float(run["seconds"])
            for run in runs
            if isinstance(run.get("seconds"), (int, float))
        ]
        imported_module_counts = [
            float(run["imported_module_count"])
            for run in runs
            if isinstance(run.get("imported_module_count"), (int, float))
        ]
        max_rss_bytes = [
            float(run["max_rss_bytes"])
            for run in runs
            if isinstance(run.get("max_rss_bytes"), (int, float))
        ]
        sample = []
        if runs and isinstance(runs[0].get("import_graph_sample"), list):
            sample = [str(value) for value in runs[0].get("import_graph_sample", [])]
        surfaces[module_name] = {
            "repetitions": int(repetitions),
            "runs": runs,
            "import_time_seconds": _float_summary(import_time_seconds),
            "imported_module_count": _float_summary(imported_module_counts),
            "max_rss_bytes": _float_summary(max_rss_bytes),
            "import_graph_hash": graph_hashes[0] if len(graph_hashes) == 1 else "mixed",
            "import_graph_hashes": graph_hashes,
            "import_graph_sample": sample,
        }
        if not runs:
            ok = False
    return surfaces, ok


def _module_symbols_snapshot(module_name: str, symbols: list[str]) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for symbol in symbols:
        if hasattr(module, symbol):
            value = getattr(module, symbol)
            resolved[symbol] = f"{value.__class__.__module__}.{value.__class__.__name__}"
        else:
            missing.append(symbol)
    return {
        "module": module_name,
        "symbols": list(symbols),
        "resolved": resolved,
        "missing": missing,
        "ok": len(missing) == 0,
    }


def _public_api_snapshot(contract: dict[str, Any]) -> dict[str, Any]:
    import intermine314

    dir_snapshot = [str(name) for name in dir(intermine314)]
    all_snapshot = [str(name) for name in getattr(intermine314, "__all__", [])]
    public_entrypoints = contract.get("public_entrypoints", {})
    exceptions = contract.get("exceptions", {})
    symbol_checks: list[dict[str, Any]] = []
    for module_name, symbols in list(public_entrypoints.items()) + list(exceptions.items()):
        symbol_checks.append(_module_symbols_snapshot(str(module_name), [str(sym) for sym in list(symbols)]))
    missing = {
        check["module"]: list(check["missing"])
        for check in symbol_checks
        if isinstance(check, dict) and isinstance(check.get("missing"), list) and check.get("missing")
    }
    return {
        "dir_intermine314": dir_snapshot,
        "dir_intermine314_hash": _sha256_json(dir_snapshot),
        "intermine314___all__": sorted(all_snapshot),
        "intermine314___all___hash": _sha256_json(sorted(all_snapshot)),
        "symbol_contract_checks": symbol_checks,
        "missing_symbols": missing,
        "ok": not missing,
    }


def _tor_invariants() -> dict[str, Any]:
    from intermine314.service.errors import TorConfigurationError
    from intermine314.service.tor import tor_proxy_url, tor_registry, tor_service
    from intermine314.service.transport import enforce_tor_dns_safe_proxy_url

    safe_proxy = "socks5h://127.0.0.1:9050"
    unsafe_proxy = "socks5://127.0.0.1:9050"
    safe_proxy_normalized = enforce_tor_dns_safe_proxy_url(
        safe_proxy,
        tor_mode=True,
        context="phase0 contract safe proxy",
        strict_tor_proxy_scheme=True,
        allow_insecure_tor_proxy_scheme=False,
    )
    transport_rejects_unsafe = False
    transport_error_type = "none"
    try:
        enforce_tor_dns_safe_proxy_url(
            unsafe_proxy,
            tor_mode=True,
            context="phase0 contract unsafe proxy",
            strict_tor_proxy_scheme=True,
            allow_insecure_tor_proxy_scheme=False,
        )
    except Exception as exc:
        transport_rejects_unsafe = isinstance(exc, TorConfigurationError)
        transport_error_type = type(exc).__name__
    helper_service_rejects_unsafe = False
    helper_service_error_type = "none"
    try:
        tor_service(
            "https://example.org/service",
            scheme="socks5",
            strict=True,
            allow_insecure_tor_proxy_scheme=False,
        )
    except Exception as exc:
        helper_service_rejects_unsafe = isinstance(exc, TorConfigurationError)
        helper_service_error_type = type(exc).__name__
    helper_registry_rejects_unsafe = False
    helper_registry_error_type = "none"
    try:
        tor_registry(
            "https://example.org/registry/instances",
            scheme="socks5",
            strict=True,
            allow_insecure_tor_proxy_scheme=False,
        )
    except Exception as exc:
        helper_registry_rejects_unsafe = isinstance(exc, TorConfigurationError)
        helper_registry_error_type = type(exc).__name__
    default_proxy = tor_proxy_url()
    default_proxy_scheme = proxy_url_scheme_from_url(default_proxy)
    ok = bool(
        safe_proxy_normalized == safe_proxy
        and transport_rejects_unsafe
        and helper_service_rejects_unsafe
        and helper_registry_rejects_unsafe
        and default_proxy_scheme == "socks5h"
    )
    return {
        "status": "ok" if ok else "failed",
        "safe_proxy_normalized": safe_proxy_normalized,
        "safe_proxy_scheme": proxy_url_scheme_from_url(safe_proxy_normalized),
        "unsafe_proxy_rejected_transport": transport_rejects_unsafe,
        "unsafe_proxy_transport_error_type": transport_error_type,
        "unsafe_proxy_rejected_tor_service": helper_service_rejects_unsafe,
        "unsafe_proxy_tor_service_error_type": helper_service_error_type,
        "unsafe_proxy_rejected_tor_registry": helper_registry_rejects_unsafe,
        "unsafe_proxy_tor_registry_error_type": helper_registry_error_type,
        "default_tor_proxy_url": default_proxy,
        "default_tor_proxy_scheme": default_proxy_scheme,
        "tor_dns_safety": "enforced",
    }


class _TrackingExecutor:
    instances: list["_TrackingExecutor"] = []

    def __init__(self, *args, **kwargs):
        _ = (args, kwargs)
        self.current_pending = 0
        self.max_pending = 0
        self.map_calls = 0
        self.map_buffersizes: list[int] = []
        self.enter_calls = 0
        self.exit_calls = 0
        _TrackingExecutor.instances.append(self)

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        self.exit_calls += 1
        return False

    def map(self, fn, iterable, *, buffersize=None):
        self.map_calls += 1
        max_pending = max(1, int(buffersize or 1))
        self.map_buffersizes.append(max_pending)
        pending = []
        source_iter = iter(iterable)
        while True:
            while len(pending) < max_pending:
                try:
                    arg = next(source_iter)
                except StopIteration:
                    break
                self.current_pending += 1
                self.max_pending = max(self.max_pending, self.current_pending)
                pending.append(fn(arg))
            if not pending:
                break
            result = pending.pop(0)
            self.current_pending -= 1
            yield result


class _FakeOffsetQuery:
    def count(self):
        return 128

    def results(self, row="dict", start=0, size=None):
        stop = int(start) + int(size or 0)
        for value in range(int(start), stop):
            yield {"row": row, "value": value}


def _parallel_invariants() -> dict[str, Any]:
    from intermine314.query.parallel_offset import run_parallel_offset

    source = inspect.getsource(run_parallel_offset)
    source_contains_buffersize = "buffersize=" in source
    _TrackingExecutor.instances.clear()
    iterator = run_parallel_offset(
        _FakeOffsetQuery(),
        row="dict",
        start=0,
        size=20,
        page_size=1,
        max_workers=4,
        order_mode="ordered",
        inflight_limit=2,
        ordered_window_pages=2,
        ordered_max_in_flight=2,
        max_inflight_bytes_estimate=None,
        job_id="phase0-contract",
        thread_name_prefix="phase0",
        executor_cls=_TrackingExecutor,
    )
    first = next(iterator)
    iterator.close()
    executor = _TrackingExecutor.instances[0] if _TrackingExecutor.instances else None
    runtime_buffersize_observed = bool(
        executor is not None
        and executor.map_calls == 1
        and executor.map_buffersizes
        and int(executor.map_buffersizes[0]) >= 1
    )
    context_managed = bool(executor is not None and executor.enter_calls == 1 and executor.exit_calls == 1)
    status_ok = bool(source_contains_buffersize and runtime_buffersize_observed and context_managed)
    return {
        "status": "ok" if status_ok else "failed",
        "source_contains_executor_map_buffersize": source_contains_buffersize,
        "runtime_buffersize_observed": runtime_buffersize_observed,
        "executor_context_managed": context_managed,
        "first_row": first,
        "map_buffersizes": [] if executor is None else executor.map_buffersizes,
        "max_pending": None if executor is None else executor.max_pending,
    }


def _elt_invariants() -> dict[str, Any]:
    from intermine314.export.fetch import fetch_from_mine
    from intermine314.query.builder import Query
    from intermine314.query.data_plane import ManagedDuckDBConnection

    fetch_signature = inspect.signature(fetch_from_mine)
    query_to_duckdb_signature = inspect.signature(Query.to_duckdb)
    fetch_params = list(fetch_signature.parameters.keys())
    query_params = list(query_to_duckdb_signature.parameters.keys())
    fetch_source = inspect.getsource(fetch_from_mine)
    required_fetch_fields = {"parquet_path", "managed", "max_inflight_bytes_estimate"}
    required_query_fields = {"managed"}
    source_tokens = ("query.to_parquet(", "read_parquet", "workflow must be 'elt'")
    source_token_checks = {token: (token in fetch_source) for token in source_tokens}
    managed_connection_support = bool(
        hasattr(ManagedDuckDBConnection, "__enter__") and hasattr(ManagedDuckDBConnection, "__exit__")
    )
    status_ok = bool(
        required_fetch_fields.issubset(fetch_signature.parameters.keys())
        and required_query_fields.issubset(query_to_duckdb_signature.parameters.keys())
        and all(source_token_checks.values())
        and managed_connection_support
    )
    return {
        "status": "ok" if status_ok else "failed",
        "fetch_from_mine_parameters": fetch_params,
        "query_to_duckdb_parameters": query_params,
        "fetch_from_mine_has_required_parameters": required_fetch_fields.issubset(fetch_signature.parameters.keys()),
        "query_to_duckdb_has_managed_parameter": required_query_fields.issubset(
            query_to_duckdb_signature.parameters.keys()
        ),
        "elt_source_token_checks": source_token_checks,
        "managed_duckdb_connection_context_manager": managed_connection_support,
    }


def _collect_single_import_graph(module_name: str) -> dict[str, Any]:
    baseline = run_import_baseline_subprocess(
        import_snippet=_build_import_snippet(module_name),
        repetitions=1,
        source_root=SRC,
    )
    run = {}
    runs = baseline.get("runs", [])
    if isinstance(runs, list) and runs and isinstance(runs[0], dict):
        run = runs[0]
    return {
        "module": module_name,
        "import_graph_hash": run.get("import_graph_hash"),
        "import_graph_sample": run.get("import_graph_sample"),
        "imported_module_count": run.get("imported_module_count"),
        "module_count": run.get("module_count"),
        "seconds": run.get("seconds"),
        "max_rss_bytes": run.get("max_rss_bytes"),
    }


def _build_report(
    contract: dict[str, Any],
    import_repetitions: int,
    *,
    contract_file: Path,
) -> tuple[int, dict[str, Any]]:
    hotspot_modules = [str(name) for name in contract.get("hotspot_modules", [])]
    hotspot_imports, import_ok = _collect_hotspot_imports(hotspot_modules, import_repetitions)
    public_api = _public_api_snapshot(contract)
    top_level_import = _collect_single_import_graph("intermine314")
    top_level_graph_hash = str(top_level_import.get("import_graph_hash") or "unknown")
    invariants = {
        "tor_dns_safety": _tor_invariants(),
        "parallel_scheduler_buffersize": _parallel_invariants(),
        "elt_pipeline": _elt_invariants(),
    }
    failing_invariants = [
        name
        for name, payload in invariants.items()
        if not (isinstance(payload, dict) and str(payload.get("status", "")) == "ok")
    ]
    status = "ok"
    error_type = "none"
    if not public_api.get("ok", False):
        status = "failed"
        error_type = "public_contract_symbol_missing"
    elif failing_invariants:
        status = "failed"
        error_type = f"invariant_failed:{failing_invariants[0]}"
    elif not import_ok:
        status = "failed"
        error_type = "hotspot_import_measurement_failed"

    import_startup_baseline = {
        module_name: {
            "import_time_seconds": payload.get("import_time_seconds"),
            "imported_module_count": payload.get("imported_module_count"),
            "max_rss_bytes": payload.get("max_rss_bytes"),
            "import_graph_hash": payload.get("import_graph_hash"),
        }
        for module_name, payload in hotspot_imports.items()
    }
    report: dict[str, Any] = {
        "timestamp_utc": now_utc_iso(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "contract_file": str(contract_file),
        "contract_schema_version": contract.get("schema_version"),
        "contract": contract,
        "public_api_snapshot": public_api,
        "top_level_import_snapshot": top_level_import,
        "import_startup_hotspots": hotspot_imports,
        "import_startup_baseline": import_startup_baseline,
        "public_api_import_graph_hash": top_level_graph_hash,
        "invariants": invariants,
    }
    attach_metric_fields(
        report,
        startup=_STARTUP,
        status=status,
        error_type=error_type,
        tor_mode="strict",
        proxy_url_scheme="socks5h",
        profile_name=PROFILE_NAME,
    )
    return (SUCCESS_EXIT_CODE if status == "ok" else FAIL_EXIT_CODE), report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-file", default=str(DEFAULT_CONTRACT_FILE))
    parser.add_argument("--import-repetitions", type=int, default=DEFAULT_RUNNER_IMPORT_REPETITIONS)
    parser.add_argument("--json-out", default=None)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract_file = Path(str(args.contract_file))
    if not contract_file.exists():
        raise FileNotFoundError(f"Contract file does not exist: {contract_file}")
    repetitions = int(args.import_repetitions)
    if repetitions <= 0:
        raise ValueError("--import-repetitions must be a positive integer")
    contract = _load_contract(contract_file)
    code, report = _build_report(contract, repetitions, contract_file=contract_file)
    output = json.dumps(report, sort_keys=True, default=str)
    if args.json_out:
        path = Path(str(args.json_out))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
    print(output)
    return code


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
