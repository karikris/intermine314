from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

from intermine314.config.constants import (
    DEFAULT_KEYSET_BATCH_SIZE,
    DEFAULT_LIST_CHUNK_SIZE,
    DEFAULT_TARGETED_EXPORT_PAGE_SIZE,
    DEFAULT_TARGETED_LIST_DESCRIPTION,
    DEFAULT_TARGETED_LIST_NAME_PREFIX,
    DEFAULT_TARGETED_LIST_TAGS,
)
from intermine314.util.deps import (
    optional_duckdb as _optional_duckdb,
    optional_polars as _optional_polars,
    quote_sql_string as _duckdb_quote,
    require_polars as _require_polars,
)
from intermine314.util.logging import log_structured_event, new_job_id
from intermine314.export.parquet import write_single_parquet_from_parts

VALID_REPORT_MODES = frozenset({"summary", "full"})
_TARGETED_LOG = logging.getLogger("intermine314.export.targeted")


@dataclass(frozen=True)
class TargetedTableSpec:
    name: str
    root_class: str
    views: list[str]
    joins: list[str]
    template_names: list[str]
    template_keywords: list[str]


@dataclass(frozen=True)
class _PreparedTableSpec:
    table: TargetedTableSpec
    template_name: str | None
    template: Any | None
    valid_views: list[str]
    valid_joins: list[str]


def _normalize_report_mode(report_mode: str) -> str:
    mode = str(report_mode or "summary").strip().lower()
    if mode not in VALID_REPORT_MODES:
        choices = ", ".join(sorted(VALID_REPORT_MODES))
        raise ValueError(f"report_mode must be one of: {choices}")
    return mode


def _sample_append(values: list[Any], item: Any, max_size: int) -> None:
    if max_size <= 0:
        return
    if len(values) >= max_size:
        del values[0]
    values.append(item)


def _new_table_report_entry() -> dict[str, Any]:
    return {
        "chunks": [],
        "errors": [],
        "template_candidates": [],
        "chunk_count": 0,
        "error_count": 0,
        "last_error": None,
        "totals": {
            "files": 0,
            "rows": 0,
            "bytes": 0,
            "seconds": 0.0,
            "templates_used": 0,
            "pages": 0,
            "chunk_write_time": 0.0,
        },
    }


def _record_table_error(
    table_report: dict[str, Any],
    error_message: str,
    *,
    report_mode: str,
    report_sample_size: int,
) -> None:
    text = str(error_message)
    table_report["error_count"] = int(table_report.get("error_count", 0)) + 1
    table_report["last_error"] = text
    if report_mode == "full":
        table_report["errors"].append(text)
    else:
        _sample_append(table_report["errors"], text, report_sample_size)


def _record_chunk_report(
    table_report: dict[str, Any],
    chunk_report: dict[str, Any],
    *,
    report_mode: str,
    report_sample_size: int,
    chunk_jsonl_handle=None,
    table_name: str | None = None,
) -> None:
    table_report["chunk_count"] = int(table_report.get("chunk_count", 0)) + 1
    totals = table_report.get("totals", {})
    totals["files"] = int(totals.get("files", 0)) + 1
    rows = chunk_report.get("rows")
    if isinstance(rows, int):
        totals["rows"] = int(totals.get("rows", 0)) + rows
    totals["bytes"] = int(totals.get("bytes", 0)) + int(chunk_report.get("bytes", 0) or 0)
    totals["seconds"] = float(totals.get("seconds", 0.0)) + float(chunk_report.get("seconds", 0.0) or 0.0)
    if chunk_report.get("used_template"):
        totals["templates_used"] = int(totals.get("templates_used", 0)) + 1
    pages = chunk_report.get("pages")
    if isinstance(pages, int):
        totals["pages"] = int(totals.get("pages", 0)) + pages
    chunk_write_time = chunk_report.get("chunk_write_time")
    if isinstance(chunk_write_time, (int, float)):
        totals["chunk_write_time"] = float(totals.get("chunk_write_time", 0.0)) + float(chunk_write_time)
    table_report["totals"] = totals

    if report_mode == "full":
        table_report["chunks"].append(chunk_report)
    else:
        _sample_append(table_report["chunks"], chunk_report, report_sample_size)

    if chunk_jsonl_handle is not None:
        payload = {
            "table": table_name,
            **chunk_report,
        }
        chunk_jsonl_handle.write(json.dumps(payload, sort_keys=True, default=str))
        chunk_jsonl_handle.write("\n")


def _log_targeted_event(level, event, **fields) -> None:
    if not _TARGETED_LOG.isEnabledFor(level):
        return
    log_structured_event(_TARGETED_LOG, level, event, **fields)


def _open_chunk_jsonl(path: str | Path):
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    handle = target_path.open("w", encoding="utf-8")
    return handle, target_path


def default_oakmine_targeted_tables() -> list[TargetedTableSpec]:
    return [
        TargetedTableSpec(
            name="core_protein",
            root_class="Protein",
            views=[
                "Protein.primaryIdentifier",
                "Protein.primaryAccession",
                "Protein.proteinDescription",
                "Protein.length",
                "Protein.molecularWeight",
                "Protein.EC",
                "Protein.symbol",
            ],
            joins=[],
            template_names=[],
            template_keywords=["protein", "core"],
        ),
        TargetedTableSpec(
            name="edge_go",
            root_class="Protein",
            views=[
                "Protein.primaryIdentifier",
                "Protein.GOTerms.primaryIdentifier",
            ],
            joins=[
                "Protein.GOTerms",
            ],
            template_names=[],
            template_keywords=["go", "gene ontology", "protein"],
        ),
        TargetedTableSpec(
            name="edge_domain",
            root_class="Protein",
            views=[
                "Protein.primaryIdentifier",
                "Protein.domainMotifs.interPro",
                "Protein.domainMotifs.crossRef",
            ],
            joins=[
                "Protein.domainMotifs",
            ],
            template_names=[],
            template_keywords=["interpro", "domain", "protein"],
        ),
    ]


def rank_template_names(names: Iterable[str], keywords: Iterable[str]) -> list[str]:
    lowered_keywords = [str(x).strip().lower() for x in keywords if str(x).strip()]
    if not lowered_keywords:
        return sorted({str(name) for name in names})
    scored: list[tuple[int, str]] = []
    for name in names:
        text = str(name)
        lower = text.lower()
        score = sum(1 for kw in lowered_keywords if kw in lower)
        if score > 0:
            scored.append((score, text))
    scored.sort(key=lambda item: (-item[0], item[1].lower()))
    return [name for _, name in scored]


def _coerce_table_specs(table_specs: list[dict[str, Any]] | None) -> list[TargetedTableSpec]:
    if not table_specs:
        return default_oakmine_targeted_tables()
    parsed: list[TargetedTableSpec] = []
    for raw in table_specs:
        name = str(raw.get("name", "")).strip()
        root_class = str(raw.get("root_class", "Protein")).strip() or "Protein"
        views = [str(v).strip() for v in raw.get("views", []) if str(v).strip()]
        joins = [str(v).strip() for v in raw.get("joins", []) if str(v).strip()]
        template_names = [str(v).strip() for v in raw.get("template_names", []) if str(v).strip()]
        template_keywords = [str(v).strip() for v in raw.get("template_keywords", []) if str(v).strip()]
        if not name or not views:
            continue
        parsed.append(
            TargetedTableSpec(
                name=name,
                root_class=root_class,
                views=views,
                joins=joins,
                template_names=template_names,
                template_keywords=template_keywords,
            )
        )
    return parsed


def _count_parquet_rows(path: Path) -> int | None:
    polars_module = _optional_polars()
    if polars_module is None:
        return None
    try:
        return int(polars_module.scan_parquet(str(path)).select(polars_module.len()).collect().item(0, 0))
    except Exception:
        return None


def _supported_paths(service: Any, root_class: str, paths: list[str]) -> list[str]:
    query = service.new_query(root_class)
    supported: list[str] = []
    for path in paths:
        try:
            _ = query.model.make_path(path, query.get_subclass_dict())
            supported.append(path)
        except Exception:
            continue
    return supported


def _select_template_for_table(service: Any, table: TargetedTableSpec, catalog: list[str]) -> str | None:
    template_set = set(service.templates.keys())
    for name in table.template_names:
        if name in template_set:
            return name
    ranked = rank_template_names(catalog, table.template_keywords)
    return ranked[0] if ranked else None


def _template_constraint_for_list(template: Any, root_class: str, list_name: str) -> dict[str, dict[str, str]] | None:
    for con in template.editable_constraints:
        code = str(getattr(con, "code", "")).strip()
        path = str(getattr(con, "path", "")).strip()
        op = str(getattr(con, "op", "")).strip().upper()
        if not code or not path:
            continue
        if path != root_class and not path.startswith(root_class + "."):
            continue
        if op in {"IN", "ONE OF", "LOOKUP", "="}:
            values = {"value": list_name}
            if op != "IN":
                values["op"] = "IN"
            return {code: values}
    return None


def _export_template_rows_to_parquet(
    template: Any,
    constraints: dict[str, dict[str, str]],
    out_path: Path,
    page_size: int,
) -> dict[str, Any]:
    polars_module = _require_polars("template parquet exports")
    total_rows = 0
    part = 0
    page_count = 0
    chunk_write_time = 0.0
    t0 = time.perf_counter()
    rows_iter = iter(template.results(row="dict", **constraints))
    with TemporaryDirectory() as tmp:
        staged_dir = Path(tmp)
        while rows := list(islice(rows_iter, page_size)):
            row_count = len(rows)
            part_t0 = time.perf_counter()
            polars_module.from_dicts(rows).write_parquet(
                str(staged_dir / f"part-{part:05d}.parquet"),
                compression="zstd",
            )
            chunk_write_time += time.perf_counter() - part_t0
            total_rows += row_count
            part += 1
            page_count += 1

        write_single_parquet_from_parts(
            staged_dir=staged_dir,
            target=out_path,
            compression="zstd",
            polars_module=polars_module,
            duckdb_module=_optional_duckdb(),
            duckdb_quote=_duckdb_quote,
        )

    elapsed = time.perf_counter() - t0
    rows_per_chunk = (float(total_rows) / float(page_count)) if page_count > 0 else 0.0
    return {
        "rows": total_rows,
        "seconds": elapsed,
        "pages": page_count,
        "rows_per_chunk": rows_per_chunk,
        "chunk_write_time": chunk_write_time,
    }


def _prepare_table_plans(
    service: Any,
    specs: list[TargetedTableSpec],
    template_catalog: list[str],
    use_templates_first: bool,
    table_report: dict[str, Any],
    *,
    report_mode: str,
    report_sample_size: int,
) -> list[_PreparedTableSpec]:
    prepared: list[_PreparedTableSpec] = []
    for table in specs:
        ranked_templates = rank_template_names(template_catalog, table.template_keywords)
        table_report[table.name]["template_candidates"].extend(ranked_templates)

        template_name = None
        template = None
        if use_templates_first:
            selected = _select_template_for_table(service, table, template_catalog)
            if selected:
                try:
                    template = service.get_template(selected)
                    template_name = selected
                except Exception as exc:
                    _record_table_error(
                        table_report[table.name],
                        f"template:{selected}:prepare:{exc}",
                        report_mode=report_mode,
                        report_sample_size=report_sample_size,
                    )

        valid_views = _supported_paths(service, table.root_class, table.views)
        valid_joins = _supported_paths(service, table.root_class, table.joins)
        if not valid_views:
            _record_table_error(
                table_report[table.name],
                "fallback_query:prepare:no_supported_views",
                report_mode=report_mode,
                report_sample_size=report_sample_size,
            )

        prepared.append(
            _PreparedTableSpec(
                table=table,
                template_name=template_name,
                template=template,
                valid_views=valid_views,
                valid_joins=valid_joins,
            )
        )
    return prepared


def _normalize_identifier_token(row: Any) -> str | None:
    value = row[0] if isinstance(row, (list, tuple)) and row else row
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _iter_identifier_chunks(
    id_iter: Iterable[Any],
    *,
    list_chunk_size: int,
    row_limit: int | None,
) -> Iterable[tuple[list[str], int]]:
    chunk: list[str] = []
    identifier_count = 0
    for row in id_iter:
        token = _normalize_identifier_token(row)
        if token is None:
            continue
        chunk.append(token)
        identifier_count += 1

        if len(chunk) >= list_chunk_size:
            yield chunk, identifier_count
            chunk = []
        if row_limit is not None and identifier_count >= row_limit:
            break

    if chunk:
        yield chunk, identifier_count


def _export_table_chunk(
    *,
    service: Any,
    prepared: _PreparedTableSpec,
    chunk_index: int,
    list_name: str,
    output_root: Path,
    page_size: int,
    max_workers: int | None,
    ordered: str | bool,
    prefetch: int | None,
    inflight_limit: int | None,
    profile: str,
    large_query_mode: bool,
    keyset_batch_size: int,
    table_report: dict[str, Any],
    report_mode: str,
    report_sample_size: int,
) -> dict[str, Any] | None:
    table = prepared.table
    table_dir = output_root / table.name
    table_dir.mkdir(parents=True, exist_ok=True)
    out_path = table_dir / f"chunk-{chunk_index:05d}.parquet"

    used_template = None
    wrote_rows = None
    elapsed = None
    pages = None
    rows_per_chunk = None
    chunk_write_time = None
    valid_views: list[str] = []
    valid_joins: list[str] = []

    if prepared.template is not None:
        try:
            con_values = _template_constraint_for_list(prepared.template, table.root_class, list_name)
            if con_values is not None:
                template_export = _export_template_rows_to_parquet(
                    template=prepared.template,
                    constraints=con_values,
                    out_path=out_path,
                    page_size=page_size,
                )
                wrote_rows = int(template_export.get("rows", 0))
                elapsed = float(template_export.get("seconds", 0.0))
                pages = int(template_export.get("pages", 0))
                rows_per_chunk = float(template_export.get("rows_per_chunk", 0.0))
                chunk_write_time = float(template_export.get("chunk_write_time", 0.0))
                used_template = prepared.template_name
        except Exception as exc:
            _record_table_error(
                table_report,
                f"template:{prepared.template_name}:chunk:{chunk_index}:{exc}",
                report_mode=report_mode,
                report_sample_size=report_sample_size,
            )

    if used_template is None:
        valid_views = list(prepared.valid_views)
        valid_joins = list(prepared.valid_joins)
        if not valid_views:
            _record_table_error(
                table_report,
                f"fallback_query:chunk:{chunk_index}:no_supported_views",
                report_mode=report_mode,
                report_sample_size=report_sample_size,
            )
            return None

        t0 = time.perf_counter()
        query = service.new_query(table.root_class)
        query.add_view(*valid_views)
        for join in valid_joins:
            query.add_join(join, "OUTER")
        query.add_constraint(table.root_class, "IN", list_name)
        query.to_parquet(
            str(out_path),
            single_file=True,
            parallel=True,
            page_size=page_size,
            max_workers=max_workers,
            ordered=ordered,
            prefetch=prefetch,
            inflight_limit=inflight_limit,
            profile=profile,
            large_query_mode=large_query_mode,
            pagination="auto",
            keyset_batch_size=keyset_batch_size,
        )
        elapsed = time.perf_counter() - t0
        wrote_rows = _count_parquet_rows(out_path)

    return {
        "chunk_index": chunk_index,
        "list_name": list_name,
        "path": str(out_path),
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
        "rows": wrote_rows,
        "seconds": elapsed,
        "used_template": used_template,
        "fallback_views": valid_views if used_template is None else None,
        "fallback_joins": valid_joins if used_template is None else None,
        "pages": pages,
        "rows_per_chunk": rows_per_chunk,
        "chunk_write_time": chunk_write_time,
    }


def export_targeted_tables_with_lists(
    *,
    service: Any,
    root_class: str,
    identifier_path: str,
    output_dir: str | Path,
    table_specs: list[dict[str, Any]] | None = None,
    id_limit: int | None = None,
    list_chunk_size: int = DEFAULT_LIST_CHUNK_SIZE,
    page_size: int = DEFAULT_TARGETED_EXPORT_PAGE_SIZE,
    max_workers: int | None = None,
    ordered: str | bool = "window",
    profile: str = "large_query",
    large_query_mode: bool = True,
    prefetch: int | None = None,
    inflight_limit: int | None = None,
    keyset_batch_size: int = DEFAULT_KEYSET_BATCH_SIZE,
    sleep_seconds: float = 0.0,
    template_keywords: list[str] | None = None,
    template_limit: int = 40,
    use_templates_first: bool = True,
    list_type: str | None = None,
    list_name_prefix: str = DEFAULT_TARGETED_LIST_NAME_PREFIX,
    list_description: str = DEFAULT_TARGETED_LIST_DESCRIPTION,
    list_tags: list[str] | None = None,
    report_mode: str = "summary",
    report_sample_size: int = 20,
    chunk_details_jsonl_path: str | Path | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """
    Export large datasets as a core table plus separate edge tables using list chunks.
    """
    if list_chunk_size <= 0:
        raise ValueError("list_chunk_size must be a positive integer")
    if page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    if not isinstance(report_sample_size, int) or isinstance(report_sample_size, bool):
        raise TypeError("report_sample_size must be an integer")
    if report_sample_size < 0:
        raise ValueError("report_sample_size must be a non-negative integer")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_mode = _normalize_report_mode(report_mode)
    run_job_id = str(job_id).strip() if job_id is not None else ""
    if not run_job_id:
        run_job_id = new_job_id("targeted")

    specs = _coerce_table_specs(table_specs)
    if not specs:
        raise ValueError("No valid table specs were supplied")

    list_type = str(list_type or root_class).strip()
    list_name_prefix = str(list_name_prefix).strip() or DEFAULT_TARGETED_LIST_NAME_PREFIX
    list_description = str(list_description).strip() or DEFAULT_TARGETED_LIST_DESCRIPTION
    effective_list_tags = list(list_tags) if list_tags is not None else list(DEFAULT_TARGETED_LIST_TAGS)
    table_report: dict[str, Any] = {table.name: _new_table_report_entry() for table in specs}

    chunk_jsonl_handle = None
    chunk_jsonl_file: Path | None = None
    if chunk_details_jsonl_path is not None and str(chunk_details_jsonl_path).strip():
        chunk_jsonl_handle, chunk_jsonl_file = _open_chunk_jsonl(chunk_details_jsonl_path)

    template_catalog = list(service.list_templates(include=template_keywords or [], limit=template_limit))
    table_plans = _prepare_table_plans(
        service=service,
        specs=specs,
        template_catalog=template_catalog,
        use_templates_first=use_templates_first,
        table_report=table_report,
        report_mode=report_mode,
        report_sample_size=report_sample_size,
    )
    _log_targeted_event(
        logging.INFO,
        "targeted_export_start",
        job_id=run_job_id,
        report_mode=report_mode,
        report_sample_size=report_sample_size,
        root_class=root_class,
        table_count=len(specs),
        chunk_jsonl_path=str(chunk_jsonl_file) if chunk_jsonl_file is not None else None,
    )

    id_query = service.new_query(root_class)
    id_query.add_view(identifier_path)
    row_limit = None if id_limit is None else max(0, int(id_limit))
    id_iter = id_query.run_parallel(
        row="list",
        start=0,
        size=row_limit,
        page_size=page_size,
        max_workers=max_workers,
        ordered=ordered,
        prefetch=prefetch,
        inflight_limit=inflight_limit,
        profile=profile,
        large_query_mode=large_query_mode,
        pagination="auto",
        keyset_path=identifier_path,
        keyset_batch_size=keyset_batch_size,
        job_id=run_job_id,
    )

    created_list_names: list[str] = []
    created_lists_total = 0
    cleanup_error = None
    chunk_count = 0
    identifier_count = 0
    try:
        for chunk_identifiers, identifier_count in _iter_identifier_chunks(
            id_iter,
            list_chunk_size=list_chunk_size,
            row_limit=row_limit,
        ):
            chunk_count += 1
            im_list = None
            try:
                created_lists = service.create_batched_lists(
                    chunk_identifiers,
                    list_type=list_type,
                    chunk_size=list_chunk_size,
                    name_prefix=list_name_prefix,
                    description=list_description,
                    tags=effective_list_tags,
                )
                if not created_lists:
                    continue
                im_list = created_lists[0]
                created_lists_total += 1
                if report_mode == "full":
                    created_list_names.append(im_list.name)
                else:
                    _sample_append(created_list_names, im_list.name, report_sample_size)

                for prepared in table_plans:
                    chunk_report = _export_table_chunk(
                        service=service,
                        prepared=prepared,
                        chunk_index=chunk_count,
                        list_name=im_list.name,
                        output_root=output_root,
                        page_size=page_size,
                        max_workers=max_workers,
                        ordered=ordered,
                        prefetch=prefetch,
                        inflight_limit=inflight_limit,
                        profile=profile,
                        large_query_mode=large_query_mode,
                        keyset_batch_size=keyset_batch_size,
                        table_report=table_report[prepared.table.name],
                        report_mode=report_mode,
                        report_sample_size=report_sample_size,
                    )
                    if chunk_report is not None:
                        _record_chunk_report(
                            table_report[prepared.table.name],
                            chunk_report,
                            report_mode=report_mode,
                            report_sample_size=report_sample_size,
                            chunk_jsonl_handle=chunk_jsonl_handle,
                            table_name=prepared.table.name,
                        )
                        _log_targeted_event(
                            logging.INFO,
                            "targeted_export_chunk",
                            job_id=run_job_id,
                            table=prepared.table.name,
                            chunk_index=chunk_report.get("chunk_index"),
                            rows=chunk_report.get("rows"),
                            duration_ms=round(float(chunk_report.get("seconds", 0.0) or 0.0) * 1000.0, 3),
                            bytes=chunk_report.get("bytes"),
                            pages=chunk_report.get("pages"),
                            rows_per_chunk=chunk_report.get("rows_per_chunk"),
                            chunk_write_time_ms=round(
                                float(chunk_report.get("chunk_write_time", 0.0) or 0.0) * 1000.0,
                                3,
                            ),
                        )
            finally:
                if im_list is not None:
                    try:
                        service.delete_lists([im_list.name])
                    except Exception as exc:  # pragma: no cover - cleanup best effort
                        if cleanup_error is None:
                            cleanup_error = str(exc)
                        else:
                            cleanup_error = cleanup_error + "; " + str(exc)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        if chunk_jsonl_handle is not None:
            chunk_jsonl_handle.close()

    totals = {}
    for table in specs:
        entry = table_report[table.name]
        totals[table.name] = dict(entry.get("totals", {}))
        entry["chunk_details_truncated"] = bool(
            report_mode == "summary" and int(entry.get("chunk_count", 0)) > len(entry.get("chunks", []))
        )
        entry["error_details_truncated"] = bool(
            report_mode == "summary" and int(entry.get("error_count", 0)) > len(entry.get("errors", []))
        )

    result = {
        "strategy": "targeted_list_exports",
        "job_id": run_job_id,
        "report_mode": report_mode,
        "report_sample_size": report_sample_size,
        "root_class": root_class,
        "identifier_path": identifier_path,
        "identifier_count": identifier_count,
        "chunk_size": list_chunk_size,
        "chunk_count": chunk_count if identifier_count else 0,
        "created_lists": created_list_names,
        "created_lists_total": created_lists_total,
        "created_lists_truncated": bool(report_mode == "summary" and created_lists_total > len(created_list_names)),
        "cleanup_error": cleanup_error,
        "template_catalog_size": len(template_catalog),
        "tables": table_report,
        "totals": totals,
        "chunk_details_jsonl_path": str(chunk_jsonl_file) if chunk_jsonl_file is not None else None,
        "output_dir": str(output_root),
    }
    _log_targeted_event(
        logging.INFO,
        "targeted_export_done",
        job_id=run_job_id,
        report_mode=report_mode,
        chunk_count=result["chunk_count"],
        identifier_count=identifier_count,
        table_count=len(specs),
        chunk_jsonl_path=result["chunk_details_jsonl_path"],
    )
    return result
