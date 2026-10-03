from __future__ import annotations

import hashlib
import json
import subprocess
import time
from contextlib import closing
from glob import escape
from itertools import islice
from pathlib import Path
from typing import Any

from benchmarks.bench_fetch import (
    _configure_legacy_intermine_transport,
    _legacy_response_scope,
    _legacy_result_iterator,
    _legacy_subprocess_env,
    _retriable_exceptions_for_mode,
    _retry_wait_seconds,
    count_with_retry,
    get_legacy_service_class,
    make_query,
    reference_python,
    resolve_benchmark_workers,
    resolve_mine_user_agent,
)
from benchmarks.bench_utils import stat_summary
from intermine314.export import query_parquet
from intermine314.export.parquet import write_parquet_batches
from intermine314.query.builder import ParallelOptions
from intermine314.service.transport import (
    default_tor_proxy_url,
    enforce_tor_dns_safe_proxy_url,
)


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


def _sample_parquet_rows(parquet_path: Path, *, columns: list[str], sample_size: int = 64) -> list[dict[str, Any]]:
    # ORDER BY makes the comparison independent of parallel page completion.
    # SQL limits the materialized Arrow/Polars result, unlike read_parquet().head().
    if sample_size <= 0:
        return []
    selected = [str(column) for column in columns if str(column).strip()]
    projection = ", ".join('"' + col.replace('"', '""') + '"' for col in selected) or "*"
    order = f" ORDER BY {projection}" if selected else ""
    return query_parquet(
        parquet_path, f"SELECT {projection} FROM results{order} LIMIT ?",
        parameters=[int(sample_size)],
    ).to_dicts()


@_legacy_response_scope()
def _legacy_export_parquet(
    *,
    mine_url: str,
    rows_target: int,
    page_size: int,
    parquet_path: Path,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
    transport_mode: str,
    tor_proxy_url_value: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    if rows_target < 0 or page_size < 1 or max_retries < 1:
        raise ValueError("rows_target must be nonnegative; page_size and max_retries must be positive")
    legacy_service_cls = get_legacy_service_class()
    if legacy_service_cls is None:
        return {"status": "skipped", "reason": "legacy intermine package is not installed"}

    proxy_url = None
    if str(transport_mode).strip().lower() == "tor":
        proxy_url = enforce_tor_dns_safe_proxy_url(
            str(tor_proxy_url_value or default_tor_proxy_url()),
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
    pl = _import_or_raise("polars", "polars is required for reference Parquet export")
    processed = 0
    retriable = _retriable_exceptions_for_mode("intermine_batched")

    def pages():
        nonlocal processed, retries
        start = 0
        while processed < rows_target:
            size = min(page_size, rows_target - processed)
            for attempt in range(1, max_retries + 1):
                try:
                    result = query.results(row="dict", start=start, size=size)
                    with _legacy_result_iterator(result) as iterator:
                        # A failed partial page is discarded before retry. The
                        # writer receives only complete bounded pages.
                        batch = list(islice(iterator, size + 1))
                        if len(batch) > size:
                            raise ValueError("reference server returned more than the requested page")
                    break
                except retriable:
                    retries += 1
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"reference Parquet export failed after retries start={start} size={size}"
                        ) from None
                    time.sleep(_retry_wait_seconds(attempt))
            if not batch:
                if available_rows is None and start > 0:
                    start = 0
                    continue
                raise RuntimeError(f"reference Parquet export returned 0 rows start={start} size={size}")
            processed += len(batch)
            start += len(batch)
            yield batch
            if available_rows is not None and start >= available_rows:
                start = 0

    started = time.perf_counter()
    write_parquet_batches(
        batches=pages(), target=parquet_path, columns=list(query.views),
        polars_module=pl, compression="zstd", single_file=True, batch_size=page_size,
    )
    elapsed = time.perf_counter() - started
    return {
        "status": "ok", "path": str(parquet_path), "rows": processed,
        "columns": list(query.views),
        "seconds": elapsed, "rows_per_s": processed / elapsed if elapsed else 0.0,
        "retries": retries, "bytes": parquet_path.stat().st_size,
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
            str(tor_proxy_url_value or default_tor_proxy_url()),
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
        "columns": list(query.views),
        "workers": int(effective_workers),
        "seconds": float(elapsed),
        "bytes": int(parquet_path.stat().st_size) if parquet_path.exists() else None,
    }


def _load_parquet_polars(parquet_path: Path, *, batch_size: int = 10000) -> dict[str, Any]:
    """Measure a complete SQL scan with bounded Arrow -> Polars batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pl = _import_or_raise("polars", "polars is required for Parquet benchmark comparison")
    duckdb = _import_or_raise("duckdb", "duckdb is required for Parquet benchmark comparison")
    started = time.perf_counter()
    rows = peak_bytes = 0
    with closing(duckdb.connect(database=":memory:")) as con:
        with closing(con.execute(
            "SELECT * FROM read_parquet(?)", [escape(str(parquet_path))]
        ).to_arrow_reader(batch_size=batch_size)) as reader:
            for batch in reader:
                frame = pl.from_arrow(batch)
                rows += frame.height
                peak_bytes = max(peak_bytes, frame.estimated_size())
                del frame
    return {
        "status": "ok", "seconds": time.perf_counter() - started, "rows": rows,
        "peak_batch_memory_bytes": peak_bytes, "batch_size": batch_size,
    }


def _load_modern_duckdb(parquet_path: Path) -> dict[str, Any]:
    duckdb = _import_or_raise("duckdb", "duckdb is required for parquet benchmark comparison")
    started = time.perf_counter()
    con = duckdb.connect(database=":memory:")
    try:
        row_count = int(con.execute("SELECT COUNT(*) FROM read_parquet(?)", [escape(str(parquet_path))]).fetchone()[0])
    finally:
        con.close()
    elapsed = time.perf_counter() - started
    return {
        "status": "ok",
        "seconds": float(elapsed),
        "rows": int(row_count),
    }


def _run_legacy_storage_subprocess(
    *,
    mine_url: str,
    rows_target: int,
    page_size: int,
    parquet_path: Path,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
    transport_mode: str,
    tor_proxy_url_value: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    interpreter = reference_python(required=False)
    if interpreter is None:
        return {"legacy_export": {"status": "skipped", "reason": "Set INTERMINE314_REFERENCE_PYTHON for the optional reference environment"},
                "legacy_polars_load": {"status": "skipped"}}
    payload = {
        "mine_url": str(mine_url),
        "rows_target": int(rows_target),
        "page_size": int(page_size),
        "parquet_path": str(parquet_path),
        "query_root_class": str(query_root_class),
        "query_views": list(query_views),
        "query_joins": list(query_joins),
        "transport_mode": str(transport_mode),
        "tor_proxy_url_value": tor_proxy_url_value,
        "timeout_seconds": float(timeout_seconds),
        "max_retries": int(max_retries),
    }
    cmd = [
        interpreter,
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
        stdout_tail_text = " | ".join(stdout_tail)
        raise RuntimeError(
            f"legacy storage subprocess failed rc={proc.returncode} "
            f"stderr={stderr} stdout_tail={stdout_tail_text}"
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
    reference_load_seconds: list[float] = []
    polars_load_seconds: list[float] = []
    duckdb_scan_seconds: list[float] = []
    row_count_matches: list[bool] = []
    sample_hash_matches: list[bool] = []

    for repetition in range(1, int(repetitions) + 1):
        reference_path = output_dir / f"reference_{transport_mode}_rep{repetition}.parquet"
        parquet_path = output_dir / f"modern_{transport_mode}_rep{repetition}.parquet"

        legacy_payload = _run_legacy_storage_subprocess(
            mine_url=mine_url,
            rows_target=rows_target,
            page_size=page_size,
            parquet_path=reference_path,
            query_root_class=query_root_class,
            query_views=query_views,
            query_joins=query_joins,
            transport_mode=transport_mode,
            tor_proxy_url_value=tor_proxy_url_value,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        legacy_export = dict(legacy_payload.get("legacy_export", {}))
        legacy_load = dict(legacy_payload.get("legacy_polars_load", {}))
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
        modern_polars_load = _load_parquet_polars(parquet_path)
        modern_duckdb_scan = _load_modern_duckdb(parquet_path)

        modern_columns = list(modern_export["columns"])
        reference_columns = (
            list(legacy_export["columns"]) if legacy_export.get("status") == "ok" else None
        )
        columns_match = reference_columns == modern_columns if reference_columns is not None else None
        reference_sample = (
            _sample_parquet_rows(reference_path, columns=reference_columns)
            if reference_columns is not None else []
        )
        parquet_sample = (
            _sample_parquet_rows(parquet_path, columns=modern_columns, sample_size=len(reference_sample))
            if reference_sample else []
        )
        sample_hash_reference = _sha256_rows(reference_sample, reference_columns) if reference_sample else None
        sample_hash_parquet = _sha256_rows(parquet_sample, modern_columns) if parquet_sample else None
        row_count_match = (
            int(legacy_load.get("rows", -1)) == int(modern_polars_load.get("rows", -2))
            if legacy_load.get("status") == "ok"
            else None
        )
        sample_hash_match = (
            columns_match and sample_hash_reference == sample_hash_parquet
            if sample_hash_reference is not None and sample_hash_parquet is not None
            else None
        )

        if legacy_export.get("status") == "ok":
            legacy_export_seconds.append(float(legacy_export.get("seconds", 0.0)))
        if modern_export.get("status") == "ok":
            modern_export_seconds.append(float(modern_export.get("seconds", 0.0)))
        if legacy_load.get("status") == "ok":
            reference_load_seconds.append(float(legacy_load.get("seconds", 0.0)))
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
                "legacy_export_parquet": legacy_export,
                "modern_export_parquet": modern_export,
                "legacy_polars_load": legacy_load,
                "modern_polars_load": modern_polars_load,
                "modern_duckdb_scan": modern_duckdb_scan,
                "parity": {
                    "columns_match": columns_match,
                    "row_count_match": row_count_match,
                    "sample_hash_reference": sample_hash_reference,
                    "sample_hash_parquet": sample_hash_parquet,
                    "sample_hash_match": sample_hash_match,
                },
                "artifacts": {
                    "reference_parquet_path": str(reference_path),
                    "modern_parquet_path": str(parquet_path),
                },
            }
        )

    return {
        "schema_version": "parquet_storage_compare_v3",
        "transport_mode": str(transport_mode),
        "repetitions": int(repetitions),
        "runs": runs,
        "summary": {
            "legacy_export_parquet_seconds": stat_summary(legacy_export_seconds),
            "modern_export_parquet_seconds": stat_summary(modern_export_seconds),
            "legacy_reference_load_seconds": stat_summary(reference_load_seconds),
            "modern_polars_load_seconds": stat_summary(polars_load_seconds),
            "modern_duckdb_scan_seconds": stat_summary(duckdb_scan_seconds),
        },
        "parity": {
            "row_count_match_all": all(row_count_matches) if row_count_matches else None,
            "sample_hash_match_all": all(sample_hash_matches) if sample_hash_matches else None,
        },
        "metadata": {
            "reference_python": reference_python(required=False),
            "sample_order": "ascending selected columns; first 64 rows",
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

    parser = argparse.ArgumentParser(description="Run the optional original-client Parquet storage baseline in its reference environment.")
    parser.add_argument("--legacy-storage-subprocess-json", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(str(args.legacy_storage_subprocess_json))
    if not isinstance(payload, dict):
        raise ValueError("legacy storage subprocess payload must be a JSON object")
    parquet_path = Path(str(payload.get("parquet_path", "")))
    legacy_export = _legacy_export_parquet(
        mine_url=str(payload.get("mine_url", "")),
        rows_target=int(payload.get("rows_target", 0)),
        page_size=int(payload.get("page_size", 0)),
        parquet_path=parquet_path,
        query_root_class=str(payload.get("query_root_class", "Gene")),
        query_views=[str(value) for value in payload.get("query_views", [])],
        query_joins=[str(value) for value in payload.get("query_joins", [])],
        transport_mode=str(payload.get("transport_mode", "direct")),
        tor_proxy_url_value=payload.get("tor_proxy_url_value"),
        timeout_seconds=float(payload.get("timeout_seconds", 60.0)),
        max_retries=int(payload.get("max_retries", 3)),
    )
    legacy_load = _load_parquet_polars(parquet_path) if legacy_export.get("status") == "ok" else {"status": "skipped"}
    print(
        json.dumps(
            {
                "legacy_export": legacy_export,
                "legacy_polars_load": legacy_load,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entrypoint
    raise SystemExit(_legacy_storage_subprocess_main())
