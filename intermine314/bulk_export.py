from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

from intermine314.constants import (
    DEFAULT_KEYSET_BATCH_SIZE,
    DEFAULT_LIST_CHUNK_SIZE,
    DEFAULT_TARGETED_EXPORT_PAGE_SIZE,
    DEFAULT_TARGETED_LIST_DESCRIPTION,
    DEFAULT_TARGETED_LIST_NAME_PREFIX,
    DEFAULT_TARGETED_LIST_TAGS,
)
from intermine314.optional_deps import (
    optional_duckdb as _optional_duckdb,
    optional_polars as _optional_polars,
    quote_sql_string as _duckdb_quote,
    require_polars as _require_polars,
)
from intermine314.query_export import write_single_parquet_from_parts


@dataclass(frozen=True)
class TargetedTableSpec:
    name: str
    root_class: str
    views: list[str]
    joins: list[str]
    template_names: list[str]
    template_keywords: list[str]


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
) -> tuple[int, float]:
    polars_module = _require_polars("template parquet exports")
    start = 0
    total_rows = 0
    part = 0
    t0 = time.perf_counter()
    with TemporaryDirectory() as tmp:
        staged_dir = Path(tmp)
        while True:
            rows = list(template.results(row="dict", start=start, size=page_size, **constraints))
            if not rows:
                break
            row_count = len(rows)
            polars_module.from_dicts(rows).write_parquet(
                str(staged_dir / f"part-{part:05d}.parquet"),
                compression="zstd",
            )
            total_rows += row_count
            start += row_count
            part += 1
            if row_count < page_size:
                break

        write_single_parquet_from_parts(
            staged_dir=staged_dir,
            target=out_path,
            compression="zstd",
            polars_module=polars_module,
            duckdb_module=_optional_duckdb(),
            duckdb_quote=_duckdb_quote,
        )

    return total_rows, time.perf_counter() - t0


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
) -> dict[str, Any]:
    """
    Export large datasets as a core table plus separate edge tables using list chunks.
    """
    if list_chunk_size <= 0:
        raise ValueError("list_chunk_size must be a positive integer")
    if page_size <= 0:
        raise ValueError("page_size must be a positive integer")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    specs = _coerce_table_specs(table_specs)
    if not specs:
        raise ValueError("No valid table specs were supplied")

    list_type = str(list_type or root_class).strip()
    list_name_prefix = str(list_name_prefix).strip() or DEFAULT_TARGETED_LIST_NAME_PREFIX
    list_description = str(list_description).strip() or DEFAULT_TARGETED_LIST_DESCRIPTION
    effective_list_tags = list(list_tags) if list_tags is not None else list(DEFAULT_TARGETED_LIST_TAGS)
    table_report: dict[str, Any] = {
        table.name: {"chunks": [], "errors": [], "template_candidates": []} for table in specs
    }

    template_catalog = service.list_templates(include=template_keywords or [], limit=template_limit)
    template_catalog = list(template_catalog)

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
    )

    identifiers: list[str] = []
    seen = set()
    for row in id_iter:
        value = row[0] if isinstance(row, (list, tuple)) and row else row
        if value is None:
            continue
        token = str(value).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        identifiers.append(token)

    if row_limit is not None and row_limit < len(identifiers):
        identifiers = identifiers[:row_limit]

    created_list_names: list[str] = []
    list_chunks = []
    if identifiers:
        list_chunks = service.create_batched_lists(
            identifiers,
            list_type=list_type,
            chunk_size=list_chunk_size,
            name_prefix=list_name_prefix,
            description=list_description,
            tags=effective_list_tags,
        )
        created_list_names = [lst.name for lst in list_chunks]

    cleanup_error = None
    try:
        for table in specs:
            for candidate in rank_template_names(template_catalog, table.template_keywords):
                table_report[table.name]["template_candidates"].append(candidate)

        for index, im_list in enumerate(list_chunks, start=1):
            for table in specs:
                table_dir = output_root / table.name
                table_dir.mkdir(parents=True, exist_ok=True)
                out_path = table_dir / f"chunk-{index:05d}.parquet"

                used_template = None
                wrote_rows = None
                elapsed = None
                valid_views: list[str] = []
                valid_joins: list[str] = []

                if use_templates_first:
                    selected = _select_template_for_table(service, table, template_catalog)
                    if selected:
                        try:
                            template = service.get_template(selected)
                            con_values = _template_constraint_for_list(template, table.root_class, im_list.name)
                            if con_values is not None:
                                wrote_rows, elapsed = _export_template_rows_to_parquet(
                                    template=template,
                                    constraints=con_values,
                                    out_path=out_path,
                                    page_size=page_size,
                                )
                                used_template = selected
                        except Exception as exc:
                            table_report[table.name]["errors"].append(
                                f"template:{selected}:chunk:{index}:{exc}"
                            )

                if used_template is None:
                    t0 = time.perf_counter()
                    query = service.new_query(table.root_class)
                    valid_views = _supported_paths(service, table.root_class, table.views)
                    valid_joins = _supported_paths(service, table.root_class, table.joins)
                    if not valid_views:
                        table_report[table.name]["errors"].append(
                            f"fallback_query:chunk:{index}:no_supported_views"
                        )
                        continue
                    query.add_view(*valid_views)
                    for join in valid_joins:
                        query.add_join(join, "OUTER")
                    query.add_constraint(table.root_class, "IN", im_list.name)
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

                table_report[table.name]["chunks"].append(
                    {
                        "chunk_index": index,
                        "list_name": im_list.name,
                        "path": str(out_path),
                        "bytes": out_path.stat().st_size if out_path.exists() else 0,
                        "rows": wrote_rows,
                        "seconds": elapsed,
                        "used_template": used_template,
                        "fallback_views": valid_views if used_template is None else None,
                        "fallback_joins": valid_joins if used_template is None else None,
                    }
                )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        if created_list_names:
            try:
                service.delete_lists(created_list_names)
            except Exception as exc:  # pragma: no cover - cleanup best effort
                cleanup_error = str(exc)

    totals = {}
    for table in specs:
        chunks = table_report[table.name]["chunks"]
        totals[table.name] = {
            "files": len(chunks),
            "rows": sum(chunk["rows"] for chunk in chunks if isinstance(chunk.get("rows"), int)),
            "bytes": sum(int(chunk.get("bytes", 0)) for chunk in chunks),
            "seconds": sum(float(chunk.get("seconds", 0.0) or 0.0) for chunk in chunks),
            "templates_used": sum(1 for chunk in chunks if chunk.get("used_template")),
        }

    return {
        "strategy": "targeted_list_exports",
        "root_class": root_class,
        "identifier_path": identifier_path,
        "identifier_count": len(identifiers),
        "chunk_size": list_chunk_size,
        "chunk_count": int(math.ceil(len(identifiers) / list_chunk_size)) if identifiers else 0,
        "created_lists": created_list_names,
        "cleanup_error": cleanup_error,
        "template_catalog_size": len(template_catalog),
        "tables": table_report,
        "totals": totals,
        "output_dir": str(output_root),
    }
