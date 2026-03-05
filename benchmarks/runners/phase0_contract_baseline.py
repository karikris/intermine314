#!/usr/bin/env python3
"""Freeze minimal runtime contract and capture phase-0 baseline artifacts."""

from __future__ import annotations

import argparse
from concurrent.futures import Future
from dataclasses import fields, is_dataclass
import hashlib
import importlib
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

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


def _resolve_dotted_object(path: str) -> Any:
    parts = [part for part in str(path).split(".") if part]
    if not parts:
        raise ValueError("symbol path must be non-empty")
    last_error: Exception | None = None
    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        attrs = parts[split:]
        try:
            obj = importlib.import_module(module_name)
        except Exception as exc:
            last_error = exc
            continue
        for attr in attrs:
            obj = getattr(obj, attr)
        return obj
    if last_error is not None:
        raise last_error
    raise ImportError(f"Could not import symbol from path: {path!r}")


def _signature_contract_snapshot(contract: dict[str, Any]) -> dict[str, Any]:
    raw_signatures = contract.get("required_signatures", {})
    checks: list[dict[str, Any]] = []
    for symbol_path, required_params in dict(raw_signatures).items():
        required = [str(name) for name in list(required_params)]
        actual: list[str] = []
        missing: list[str] = []
        in_declared_order = False
        error = None
        try:
            symbol = _resolve_dotted_object(str(symbol_path))
            signature = inspect.signature(symbol)
            actual = [str(name) for name in signature.parameters.keys()]
            missing = [name for name in required if name not in actual]
            positions = [actual.index(name) for name in required if name in actual]
            in_declared_order = positions == sorted(positions)
            ok = len(missing) == 0 and in_declared_order
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        checks.append(
            {
                "symbol": str(symbol_path),
                "required_parameters": required,
                "actual_parameters": actual,
                "missing_parameters": missing,
                "parameters_in_declared_order": in_declared_order,
                "error": error,
                "ok": bool(ok),
            }
        )
    failures = {
        str(item.get("symbol")): {
            "missing_parameters": list(item.get("missing_parameters", [])),
            "parameters_in_declared_order": bool(item.get("parameters_in_declared_order", False)),
            "error": item.get("error"),
        }
        for item in checks
        if isinstance(item, dict) and not bool(item.get("ok", False))
    }
    return {
        "checks": checks,
        "failures": failures,
        "ok": len(failures) == 0,
    }


def _dataclass_contract_snapshot(contract: dict[str, Any]) -> dict[str, Any]:
    raw_fields = contract.get("required_dataclass_fields", {})
    checks: list[dict[str, Any]] = []
    for symbol_path, required_fields in dict(raw_fields).items():
        required = [str(name) for name in list(required_fields)]
        actual: list[str] = []
        missing: list[str] = []
        error = None
        try:
            symbol = _resolve_dotted_object(str(symbol_path))
            if not is_dataclass(symbol):
                raise TypeError(f"{symbol_path} is not a dataclass")
            actual = [field.name for field in fields(symbol)]
            missing = [name for name in required if name not in actual]
            ok = len(missing) == 0
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        checks.append(
            {
                "symbol": str(symbol_path),
                "required_fields": required,
                "actual_fields": actual,
                "missing_fields": missing,
                "error": error,
                "ok": bool(ok),
            }
        )
    failures = {
        str(item.get("symbol")): {
            "missing_fields": list(item.get("missing_fields", [])),
            "error": item.get("error"),
        }
        for item in checks
        if isinstance(item, dict) and not bool(item.get("ok", False))
    }
    return {
        "checks": checks,
        "failures": failures,
        "ok": len(failures) == 0,
    }


def _forbidden_symbols_snapshot(contract: dict[str, Any]) -> dict[str, Any]:
    raw_symbols = contract.get("forbidden_symbols", [])
    checks: list[dict[str, Any]] = []
    for symbol_path in [str(item) for item in list(raw_symbols)]:
        present = False
        resolved_type = None
        error = None
        try:
            value = _resolve_dotted_object(symbol_path)
            present = True
            resolved_type = f"{value.__class__.__module__}.{value.__class__.__name__}"
        except (AttributeError, ImportError, ModuleNotFoundError):
            present = False
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        checks.append(
            {
                "symbol": symbol_path,
                "present": bool(present),
                "resolved_type": resolved_type,
                "error": error,
                "ok": (not bool(present)) and error is None,
            }
        )
    failures = {
        str(item.get("symbol")): {
            "present": bool(item.get("present", False)),
            "resolved_type": item.get("resolved_type"),
            "error": item.get("error"),
        }
        for item in checks
        if isinstance(item, dict) and not bool(item.get("ok", False))
    }
    return {
        "checks": checks,
        "failures": failures,
        "ok": len(failures) == 0,
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
        self.submit_calls = 0
        self.submitted_offsets: list[int] = []
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

    def submit(self, fn, *args, **kwargs):
        self.submit_calls += 1
        if len(args) >= 2:
            self.submitted_offsets.append(int(args[1]))
        self.current_pending += 1
        self.max_pending = max(self.max_pending, self.current_pending)
        fut = _TrackingFuture(self)
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            fut.set_exception(exc)
        return fut


class _TrackingFuture(Future):
    def __init__(self, owner: _TrackingExecutor):
        super().__init__()
        self._owner = owner
        self._released = False

    def _release(self) -> None:
        if self._released:
            return
        self._released = True
        self._owner.current_pending = max(0, self._owner.current_pending - 1)

    def result(self, timeout=None):
        try:
            return super().result(timeout=timeout)
        finally:
            self._release()


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.closed = False

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        self.closed = True


class _VerifyTLSTrackingOpener:
    instances: list["_VerifyTLSTrackingOpener"] = []

    def __init__(self, *args, **kwargs):
        _ = args
        self.kwargs = dict(kwargs)
        self._session = object()
        self._owns_session = False
        _VerifyTLSTrackingOpener.instances.append(self)

    def open(self, url, *args, **kwargs):
        _ = (args, kwargs)
        text = str(url)
        if text.endswith("/version/ws"):
            return _FakeResponse(b"8")
        if text.endswith("/service/instances") or text.endswith("/mines.json"):
            payload = b'{"instances":[{"name":"examplemine","url":"https://example.org/service"}]}'
            return _FakeResponse(payload)
        raise RuntimeError(f"Unexpected URL in phase-0 verify_tls probe: {text!r}")

    def close(self):
        return


def _verify_tls_invariants() -> dict[str, Any]:
    from intermine314.service.service import Registry, Service

    ca_bundle = "/tmp/intermine314-ca-bundle.pem"
    services: list[Any] = []
    observed: dict[str, Any] = {}
    error = None
    ok = False
    _VerifyTLSTrackingOpener.instances.clear()
    try:
        with patch("intermine314.service.service.InterMineURLOpener", _VerifyTLSTrackingOpener):
            service_ca = Service("https://example.org/service", verify_tls=ca_bundle)
            services.append(service_ca)
            observed["service_verify_tls_ca"] = _VerifyTLSTrackingOpener.instances[-1].kwargs.get("verify_tls")

            service_false = Service("https://example.org/service", verify_tls=False)
            services.append(service_false)
            observed["service_verify_tls_false"] = _VerifyTLSTrackingOpener.instances[-1].kwargs.get("verify_tls")

            registry_ca = Registry("https://example.org/service/instances", verify_tls=ca_bundle)
            services.append(registry_ca)
            observed["registry_verify_tls_ca"] = _VerifyTLSTrackingOpener.instances[-1].kwargs.get("verify_tls")

            registry_false = Registry("https://example.org/service/instances", verify_tls=False)
            services.append(registry_false)
            observed["registry_verify_tls_false"] = _VerifyTLSTrackingOpener.instances[-1].kwargs.get("verify_tls")
        ok = bool(
            observed.get("service_verify_tls_ca") == ca_bundle
            and observed.get("service_verify_tls_false") is False
            and observed.get("registry_verify_tls_ca") == ca_bundle
            and observed.get("registry_verify_tls_false") is False
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for resource in services:
            try:
                close_fn = getattr(resource, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                continue
    return {
        "status": "ok" if ok else "failed",
        "verify_tls_service_ca_bundle_passthrough": observed.get("service_verify_tls_ca"),
        "verify_tls_service_false_passthrough": observed.get("service_verify_tls_false"),
        "verify_tls_registry_ca_bundle_passthrough": observed.get("registry_verify_tls_ca"),
        "verify_tls_registry_false_passthrough": observed.get("registry_verify_tls_false"),
        "error": error,
    }


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
    source_uses_offset_pages = "offsets = tuple(range(" in source
    source_uses_submit_wait = "executor.submit(" in source and "wait(" in source
    source_uses_bounded_queue = "BoundedInflightQueue(" in source
    _TrackingExecutor.instances.clear()
    ordered_rows = list(
        run_parallel_offset(
            _FakeOffsetQuery(),
            row="dict",
            start=0,
            size=20,
            page_size=1,
            max_workers=4,
            order_mode="ordered",
            inflight_limit=2,
            max_inflight_bytes_estimate=None,
            job_id="phase0-contract-ordered",
            thread_name_prefix="phase0",
            executor_cls=_TrackingExecutor,
        )
    )
    ordered_executor = _TrackingExecutor.instances[0] if _TrackingExecutor.instances else None
    ordered_sequence_ok = [int(row["value"]) for row in ordered_rows] == list(range(20))
    ordered_bounded = bool(
        ordered_executor is not None and int(ordered_executor.max_pending) <= 2 and ordered_executor.submit_calls >= 1
    )
    ordered_context_managed = bool(
        ordered_executor is not None and ordered_executor.enter_calls == 1 and ordered_executor.exit_calls == 1
    )

    _TrackingExecutor.instances.clear()
    byte_capped_rows = list(
        run_parallel_offset(
            _FakeOffsetQuery(),
            row="dict",
            start=0,
            size=6,
            page_size=1,
            max_workers=4,
            order_mode="unordered",
            inflight_limit=8,
            max_inflight_bytes_estimate=64,
            job_id="phase0-contract-bytes",
            thread_name_prefix="phase0",
            executor_cls=_TrackingExecutor,
        )
    )
    byte_capped_executor = _TrackingExecutor.instances[0] if _TrackingExecutor.instances else None
    bytes_cap_limits_pending = bool(
        byte_capped_executor is not None
        and byte_capped_executor.submit_calls >= 1
        and int(byte_capped_executor.max_pending) <= 1
    )
    byte_capped_context_managed = bool(
        byte_capped_executor is not None
        and byte_capped_executor.enter_calls == 1
        and byte_capped_executor.exit_calls == 1
    )

    status_ok = bool(
        source_uses_offset_pages
        and source_uses_submit_wait
        and source_uses_bounded_queue
        and ordered_sequence_ok
        and ordered_bounded
        and ordered_context_managed
        and bytes_cap_limits_pending
        and byte_capped_context_managed
    )
    return {
        "status": "ok" if status_ok else "failed",
        "source_uses_offset_pages": source_uses_offset_pages,
        "source_uses_submit_wait": source_uses_submit_wait,
        "source_uses_bounded_queue": source_uses_bounded_queue,
        "ordered_sequence_ok": ordered_sequence_ok,
        "ordered_rows_count": len(ordered_rows),
        "ordered_inflight_bounded": ordered_bounded,
        "ordered_executor_context_managed": ordered_context_managed,
        "ordered_max_pending": None if ordered_executor is None else int(ordered_executor.max_pending),
        "ordered_submit_calls": None if ordered_executor is None else int(ordered_executor.submit_calls),
        "bytes_cap_rows_count": len(byte_capped_rows),
        "bytes_cap_limits_pending": bytes_cap_limits_pending,
        "bytes_cap_executor_context_managed": byte_capped_context_managed,
        "bytes_cap_max_pending": None if byte_capped_executor is None else int(byte_capped_executor.max_pending),
        "bytes_cap_submit_calls": None if byte_capped_executor is None else int(byte_capped_executor.submit_calls),
    }


def _elt_invariants() -> dict[str, Any]:
    from intermine314.export.fetch import fetch_from_mine
    from intermine314.export.managed import ManagedDuckDBConnection
    from intermine314.query.builder import Query

    fetch_signature = inspect.signature(fetch_from_mine)
    query_to_parquet_signature = inspect.signature(Query.to_parquet)
    query_to_duckdb_signature = inspect.signature(Query.to_duckdb)
    fetch_params = list(fetch_signature.parameters.keys())
    query_parquet_params = list(query_to_parquet_signature.parameters.keys())
    query_params = list(query_to_duckdb_signature.parameters.keys())
    fetch_source = inspect.getsource(fetch_from_mine)
    required_fetch_fields = {"parquet_path", "managed", "max_inflight_bytes_estimate"}
    required_parquet_fields = {"path", "single_file", "parallel_options"}
    required_duckdb_fields = {"managed", "single_file", "table"}
    source_tokens = ("query.to_parquet(", "read_parquet", "workflow must be 'elt'")
    source_token_checks = {token: (token in fetch_source) for token in source_tokens}
    managed_connection_support = bool(
        hasattr(ManagedDuckDBConnection, "__enter__") and hasattr(ManagedDuckDBConnection, "__exit__")
    )
    status_ok = bool(
        required_fetch_fields.issubset(fetch_signature.parameters.keys())
        and required_parquet_fields.issubset(query_to_parquet_signature.parameters.keys())
        and required_duckdb_fields.issubset(query_to_duckdb_signature.parameters.keys())
        and all(source_token_checks.values())
        and managed_connection_support
    )
    return {
        "status": "ok" if status_ok else "failed",
        "fetch_from_mine_parameters": fetch_params,
        "query_to_parquet_parameters": query_parquet_params,
        "query_to_duckdb_parameters": query_params,
        "fetch_from_mine_has_required_parameters": required_fetch_fields.issubset(fetch_signature.parameters.keys()),
        "query_to_parquet_has_required_parameters": required_parquet_fields.issubset(
            query_to_parquet_signature.parameters.keys()
        ),
        "query_to_duckdb_has_required_parameters": required_duckdb_fields.issubset(
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
    forbidden_symbols = _forbidden_symbols_snapshot(contract)
    signature_contract = _signature_contract_snapshot(contract)
    dataclass_contract = _dataclass_contract_snapshot(contract)
    top_level_import = _collect_single_import_graph("intermine314")
    top_level_graph_hash = str(top_level_import.get("import_graph_hash") or "unknown")
    invariants = {
        "tor_dns_safety": _tor_invariants(),
        "verify_tls_passthrough": _verify_tls_invariants(),
        "parallel_offset_scheduler": _parallel_invariants(),
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
    elif not forbidden_symbols.get("ok", False):
        status = "failed"
        error_type = "public_contract_forbidden_symbol_present"
    elif not signature_contract.get("ok", False):
        status = "failed"
        error_type = "public_contract_signature_mismatch"
    elif not dataclass_contract.get("ok", False):
        status = "failed"
        error_type = "public_contract_dataclass_field_missing"
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
        "forbidden_symbols_snapshot": forbidden_symbols,
        "signature_contract_snapshot": signature_contract,
        "dataclass_contract_snapshot": dataclass_contract,
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
