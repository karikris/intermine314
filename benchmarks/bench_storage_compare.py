from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.bench_fetch import (
    RETRIABLE_EXC,
    _configure_legacy_intermine_transport,
    _retry_wait_seconds,
    count_with_retry,
    get_legacy_service_class,
    make_query,
    resolve_benchmark_workers,
    resolve_mine_user_agent,
)
from benchmarks.bench_utils import stat_summary
from intermine314.query.builder import ParallelOptions
from intermine314.service.transport import enforce_tor_dns_safe_proxy_url
from intermine314.service.tor import tor_proxy_url

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"


def _import_or_raise(module_name: str, requirement_msg: str):
    try:
        return __import__(module_name)
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(requirement_msg) from exc


def _sha256_rows(rows: list[dict[str, Any]], columns: list[str]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        hasher.update("\x1f".join(values).encode("utf-8", errors="replace"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _sample_csv_rows(csv_path: Path, *, sample_size: int = 64) -> tuple[list[str], list[dict[str, Any]]]:
    headers: list[str] = []
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))
            if len(rows) >= int(sample_size):
                break
    return headers, rows


def _sample_parquet_rows(parquet_path: Path, *, columns: list[str], sample_size: int = 64) -> list[dict[str, Any]]:
    pl = _import_or_raise("polars", "polars is required for parquet sampling benchmarks")
    selected = [column for column in columns if str(column).strip()]
    if selected:
        frame = pl.read_parquet(parquet_path, columns=selected).head(int(sample_size))
    else:
        frame = pl.read_parquet(parquet_path).head(int(sample_size))
    return frame.to_dicts()


def _legacy_export_csv(
    *,
    mine_url: str,
    rows_target: int,
    page_size: int,
    csv_path: Path,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
    transport_mode: str,
    tor_proxy_url_value: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    legacy_service_cls = get_legacy_service_class()
    if legacy_service_cls is None:
        return {"status": "skipped", "reason": "legacy intermine package is not installed"}

    proxy_url = None
    if str(transport_mode).strip().lower() == "tor":
        proxy_url = enforce_tor_dns_safe_proxy_url(
            str(tor_proxy_url_value or tor_proxy_url()),
            tor_mode=True,
            context="legacy storage compare proxy_url",
        )

    _configure_legacy_intermine_transport(
        mine_url=mine_url,
        user_agent=resolve_mine_user_agent(mine_url),
        proxy_url=proxy_url,
        timeout_seconds=float(timeout_seconds),
    )
    query = make_query(
        legacy_service_cls,
        mine_url,
        query_root_class,
        query_views,
        query_joins,
        service_kwargs={},
    )
    available_rows, retries, _, _ = count_with_retry(
        query,
        max_retries=int(max_retries),
        sleep_seconds=0.0,
        rows_target=int(rows_target),
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()

    processed = 0
    start = 0
    writer = None
    started = time.perf_counter()
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        while processed < int(rows_target):
            remaining = int(rows_target) - processed
            size = min(int(page_size), remaining)
            got = 0
            for attempt in range(1, int(max_retries) + 1):
                try:
                    iterator = query.results(row="dict", start=start, size=size)
                    for row in iterator:
                        if writer is None:
                            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                            writer.writeheader()
                        writer.writerow(row)
                        got += 1
                    break
                except RETRIABLE_EXC:
                    retries += 1
                    wait_s = _retry_wait_seconds(attempt)
                    time.sleep(wait_s)
            else:
                raise RuntimeError(f"legacy csv export failed after retries start={start} size={size}")

            if got == 0 and available_rows is None and start > 0:
                start = 0
                continue
            if got == 0:
                raise RuntimeError(f"legacy csv export returned 0 rows start={start} size={size}")

            processed += got
            start += got
            if available_rows is not None and start >= int(available_rows):
                start = 0
    elapsed = time.perf_counter() - started
    return {
        "status": "ok",
        "path": str(csv_path),
        "rows": int(processed),
        "seconds": float(elapsed),
        "rows_per_s": float(processed / elapsed) if elapsed else 0.0,
        "retries": int(retries),
        "bytes": int(csv_path.stat().st_size) if csv_path.exists() else None,
    }


def _modern_export_parquet(
    *,
    mine_url: str,
    rows_target: int,
    page_size: int,
    workers: int | None,
    parquet_path: Path,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
    transport_mode: str,
    tor_proxy_url_value: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    from benchmarks.bench_fetch import NewService

    proxy_url = None
    tor_mode = str(transport_mode).strip().lower() == "tor"
    if tor_mode:
        proxy_url = enforce_tor_dns_safe_proxy_url(
            str(tor_proxy_url_value or tor_proxy_url()),
            tor_mode=True,
            context="modern parquet export proxy_url",
        )
    service_kwargs = {
        "proxy_url": proxy_url,
        "tor": tor_mode,
        "user_agent": resolve_mine_user_agent(mine_url),
        "request_timeout": (float(timeout_seconds), float(timeout_seconds)),
    }
    query = make_query(
        NewService,
        mine_url,
        query_root_class,
        query_views,
        query_joins,
        service_kwargs=service_kwargs,
    )
    effective_workers = resolve_benchmark_workers(mine_url, int(rows_target), workers)
    options = ParallelOptions(
        page_size=int(page_size),
        max_workers=int(effective_workers),
        ordered="unordered",
        prefetch=None,
        inflight_limit=None,
        max_inflight_bytes_estimate=None,
        profile="default",
        large_query_mode=True,
        pagination="auto",
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        parquet_path.unlink()
    started = time.perf_counter()
    query.to_parquet(
        str(parquet_path),
        start=0,
        size=int(rows_target),
        single_file=True,
        parallel_options=options,
    )
    elapsed = time.perf_counter() - started
    return {
        "status": "ok",
        "path": str(parquet_path),
        "rows_target": int(rows_target),
        "workers": int(effective_workers),
        "seconds": float(elapsed),
        "bytes": int(parquet_path.stat().st_size) if parquet_path.exists() else None,
    }


def _load_legacy_pandas(csv_path: Path) -> dict[str, Any]:
    pd = _import_or_raise("pandas", "pandas is required for legacy csv benchmark comparison")
    started = time.perf_counter()
    frame = pd.read_csv(csv_path)
    elapsed = time.perf_counter() - started
    rows = int(frame.shape[0])
    memory_bytes = int(frame.memory_usage(deep=True).sum())
    return {
        "status": "ok",
        "seconds": float(elapsed),
        "rows": rows,
        "memory_bytes": memory_bytes,
    }


def _load_modern_polars(parquet_path: Path) -> dict[str, Any]:
    pl = _import_or_raise("polars", "polars is required for parquet benchmark comparison")
    started = time.perf_counter()
    frame = pl.read_parquet(parquet_path)
    elapsed = time.perf_counter() - started
    rows = int(frame.height)
    memory_bytes = int(frame.estimated_size())
    return {
        "status": "ok",
        "seconds": float(elapsed),
        "rows": rows,
        "memory_bytes": memory_bytes,
    }


def _load_modern_duckdb(parquet_path: Path) -> dict[str, Any]:
    duckdb = _import_or_raise("duckdb", "duckdb is required for parquet benchmark comparison")
    started = time.perf_counter()
    con = duckdb.connect(database=":memory:")
    try:
        row_count = int(con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(parquet_path)]).fetchone()[0])
    finally:
        con.close()
    elapsed = time.perf_counter() - started
    return {
        "status": "ok",
        "seconds": float(elapsed),
        "rows": int(row_count),
    }


def _legacy_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    entries = [str(_REPO_ROOT), str(_SRC_ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _run_legacy_storage_subprocess(
    *,
    mine_url: str,
    rows_target: int,
    page_size: int,
    csv_path: Path,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
    transport_mode: str,
    tor_proxy_url_value: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    payload = {
        "mine_url": str(mine_url),
        "rows_target": int(rows_target),
        "page_size": int(page_size),
        "csv_path": str(csv_path),
        "query_root_class": str(query_root_class),
        "query_views": list(query_views),
        "query_joins": list(query_joins),
        "transport_mode": str(transport_mode),
        "tor_proxy_url_value": tor_proxy_url_value,
        "timeout_seconds": float(timeout_seconds),
        "max_retries": int(max_retries),
    }
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.bench_storage_compare",
        "--legacy-storage-subprocess-json",
        json.dumps(payload, separators=(",", ":")),
    ]
    proc = subprocess.run(
        cmd,
        env=_legacy_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = str(proc.stderr or "").strip()
        stdout_tail = str(proc.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(
            "legacy storage subprocess failed rc=%s stderr=%s stdout_tail=%s"
            % (proc.returncode, stderr, " | ".join(stdout_tail))
        )
    payload_obj: dict[str, Any] | None = None
    for line in reversed(str(proc.stdout or "").splitlines()):
        text = str(line).strip()
        if not text:
            continue
        try:
            candidate = json.loads(text)
        except Exception:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("legacy_export"), dict):
            payload_obj = candidate
            break
    if payload_obj is None:
        raise RuntimeError("legacy storage subprocess emitted no JSON payload")
    return payload_obj


def run_storage_compare(
    *,
    mine_url: str,
    rows_target: int,
    page_size: int,
    workers: int | None,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
    transport_mode: str,
    tor_proxy_url_value: str | None,
    output_dir: Path,
    timeout_seconds: float,
    max_retries: int,
    repetitions: int = 3,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    legacy_export_seconds: list[float] = []
    modern_export_seconds: list[float] = []
    pandas_load_seconds: list[float] = []
    polars_load_seconds: list[float] = []
    duckdb_scan_seconds: list[float] = []
    row_count_matches: list[bool] = []
    sample_hash_matches: list[bool] = []

    for repetition in range(1, int(repetitions) + 1):
        csv_path = output_dir / f"legacy_{transport_mode}_rep{repetition}.csv"
        parquet_path = output_dir / f"modern_{transport_mode}_rep{repetition}.parquet"

        legacy_payload = _run_legacy_storage_subprocess(
            mine_url=mine_url,
            rows_target=rows_target,
            page_size=page_size,
            csv_path=csv_path,
            query_root_class=query_root_class,
            query_views=query_views,
            query_joins=query_joins,
            transport_mode=transport_mode,
            tor_proxy_url_value=tor_proxy_url_value,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        legacy_export = dict(legacy_payload.get("legacy_export", {}))
        legacy_load = dict(legacy_payload.get("legacy_pandas_load", {}))
        modern_export = _modern_export_parquet(
            mine_url=mine_url,
            rows_target=rows_target,
            page_size=page_size,
            workers=workers,
            parquet_path=parquet_path,
            query_root_class=query_root_class,
            query_views=query_views,
            query_joins=query_joins,
            transport_mode=transport_mode,
            tor_proxy_url_value=tor_proxy_url_value,
            timeout_seconds=timeout_seconds,
        )
        modern_polars_load = _load_modern_polars(parquet_path)
        modern_duckdb_scan = _load_modern_duckdb(parquet_path)

        headers, csv_sample = _sample_csv_rows(csv_path) if legacy_export.get("status") == "ok" else ([], [])
        parquet_sample = (
            _sample_parquet_rows(parquet_path, columns=headers, sample_size=min(len(csv_sample), 64))
            if headers and csv_sample
            else []
        )
        sample_columns = headers
        sample_hash_csv = _sha256_rows(csv_sample, sample_columns) if csv_sample else None
        sample_hash_parquet = _sha256_rows(parquet_sample, sample_columns) if parquet_sample else None
        row_count_match = (
            int(legacy_load.get("rows", -1)) == int(modern_polars_load.get("rows", -2))
            if legacy_load.get("status") == "ok"
            else None
        )
        sample_hash_match = (
            sample_hash_csv == sample_hash_parquet
            if sample_hash_csv is not None and sample_hash_parquet is not None
            else None
        )

        if legacy_export.get("status") == "ok":
            legacy_export_seconds.append(float(legacy_export.get("seconds", 0.0)))
        if modern_export.get("status") == "ok":
            modern_export_seconds.append(float(modern_export.get("seconds", 0.0)))
        if legacy_load.get("status") == "ok":
            pandas_load_seconds.append(float(legacy_load.get("seconds", 0.0)))
        if modern_polars_load.get("status") == "ok":
            polars_load_seconds.append(float(modern_polars_load.get("seconds", 0.0)))
        if modern_duckdb_scan.get("status") == "ok":
            duckdb_scan_seconds.append(float(modern_duckdb_scan.get("seconds", 0.0)))
        if isinstance(row_count_match, bool):
            row_count_matches.append(row_count_match)
        if isinstance(sample_hash_match, bool):
            sample_hash_matches.append(sample_hash_match)

        runs.append(
            {
                "repetition": int(repetition),
                "legacy_export_csv": legacy_export,
                "modern_export_parquet": modern_export,
                "legacy_pandas_load": legacy_load,
                "modern_polars_load": modern_polars_load,
                "modern_duckdb_scan": modern_duckdb_scan,
                "parity": {
                    "row_count_match": row_count_match,
                    "sample_hash_csv": sample_hash_csv,
                    "sample_hash_parquet": sample_hash_parquet,
                    "sample_hash_match": sample_hash_match,
                },
                "artifacts": {
                    "legacy_csv_path": str(csv_path),
                    "modern_parquet_path": str(parquet_path),
                },
            }
        )

    return {
        "schema_version": "legacy_storage_compare_v2",
        "transport_mode": str(transport_mode),
        "repetitions": int(repetitions),
        "runs": runs,
        "summary": {
            "legacy_export_csv_seconds": stat_summary(legacy_export_seconds),
            "modern_export_parquet_seconds": stat_summary(modern_export_seconds),
            "legacy_pandas_load_seconds": stat_summary(pandas_load_seconds),
            "modern_polars_load_seconds": stat_summary(polars_load_seconds),
            "modern_duckdb_scan_seconds": stat_summary(duckdb_scan_seconds),
        },
        "parity": {
            "row_count_match_all": all(row_count_matches) if row_count_matches else None,
            "sample_hash_match_all": all(sample_hash_matches) if sample_hash_matches else None,
        },
        "metadata": {
            "query_root_class": str(query_root_class),
            "query_views": list(query_views),
            "query_joins": list(query_joins),
            "rows_target": int(rows_target),
            "page_size": int(page_size),
            "workers": workers,
        },
    }


def _legacy_storage_subprocess_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run one legacy CSV+pandas storage baseline in a subprocess.")
    parser.add_argument("--legacy-storage-subprocess-json", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(str(args.legacy_storage_subprocess_json))
    if not isinstance(payload, dict):
        raise ValueError("legacy storage subprocess payload must be a JSON object")
    csv_path = Path(str(payload.get("csv_path", "")))
    legacy_export = _legacy_export_csv(
        mine_url=str(payload.get("mine_url", "")),
        rows_target=int(payload.get("rows_target", 0)),
        page_size=int(payload.get("page_size", 0)),
        csv_path=csv_path,
        query_root_class=str(payload.get("query_root_class", "Gene")),
        query_views=[str(value) for value in payload.get("query_views", [])],
        query_joins=[str(value) for value in payload.get("query_joins", [])],
        transport_mode=str(payload.get("transport_mode", "direct")),
        tor_proxy_url_value=payload.get("tor_proxy_url_value"),
        timeout_seconds=float(payload.get("timeout_seconds", 60.0)),
        max_retries=int(payload.get("max_retries", 3)),
    )
    legacy_load = _load_legacy_pandas(csv_path) if legacy_export.get("status") == "ok" else {"status": "skipped"}
    print(
        json.dumps(
            {
                "legacy_export": legacy_export,
                "legacy_pandas_load": legacy_load,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    raise SystemExit(_legacy_storage_subprocess_main())
