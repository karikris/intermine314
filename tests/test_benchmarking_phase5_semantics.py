from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _reload_benchmarks_module():
    sys.modules.pop("benchmarks.benchmarks", None)
    return importlib.import_module("benchmarks.benchmarks")


class _ReplayIORuntime:
    def __init__(self, *, parity_equivalent: bool = True):
        self.parity_equivalent = parity_equivalent

    @staticmethod
    def csv_parquet_size_stats(_csv_path: Path, _parquet_path: Path):
        return {"csv_bytes": 10, "parquet_bytes": 5, "saved_bytes": 5, "reduction_pct": 50.0, "csv_to_parquet_ratio": 2.0}

    def compare_csv_parquet_parity(self, **_kwargs):
        return {"equivalent": self.parity_equivalent}

    @staticmethod
    def infer_dataframe_columns(csv_path: Path, root_class: str, views: list[str]):
        del csv_path, root_class, views
        return {"cds_column": None, "length_column": None, "group_column": "id"}

    @staticmethod
    def bench_pandas(_path: Path, repetitions: int, cds_column, length_column, group_column):
        del repetitions, cds_column, length_column, group_column
        return {"load_seconds": {"mean": 1.0}}

    @staticmethod
    def bench_polars(_path: Path, repetitions: int, cds_column, length_column, group_column):
        del repetitions, cds_column, length_column, group_column
        return {"load_seconds": {"mean": 0.5}}

    def bench_parquet_join_engines(self, **_kwargs):
        return {
            "duckdb": {"seconds": {"mean": 0.4}},
            "polars": {"seconds": {"mean": 0.3}},
            "parity": {"all_equivalent": self.parity_equivalent},
        }

    @staticmethod
    def export_for_storage(**_kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("offline replay should not call export_for_storage")

    @staticmethod
    def export_new_only_for_dataframe(**_kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("offline replay should not call export_new_only_for_dataframe")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("artifact", encoding="utf-8")


def _build_args(tmp_path: Path, *, strict_parity: bool) -> SimpleNamespace:
    return SimpleNamespace(
        mine_url="https://example.org/mine",
        baseline_rows=100,
        parallel_rows=200,
        dataframe_repetitions=1,
        matrix_storage_dir=str(tmp_path / "matrix"),
        csv_old_path=str(tmp_path / "old.csv"),
        parquet_compare_path=str(tmp_path / "compare.parquet"),
        csv_new_path=str(tmp_path / "new.csv"),
        parquet_new_path=str(tmp_path / "new.parquet"),
        parity_sample_mode="head",
        parity_sample_size=2,
        strict_parity=strict_parity,
        offline_replay_stage_io=True,
    )


def test_storage_stage_offline_replay_uses_existing_artifacts(tmp_path):
    benchmarks_module = _reload_benchmarks_module()
    args = _build_args(tmp_path, strict_parity=False)

    for base in (args.csv_old_path, args.parquet_compare_path, args.csv_new_path, args.parquet_new_path):
        _touch(benchmarks_module._query_output_path(base, "simple"))

    io_runtime = _ReplayIORuntime(parity_equivalent=True)
    result = benchmarks_module._run_storage_dataframe_join_benchmark(
        io_runtime=io_runtime,
        args=args,
        query_kind="simple",
        io_page_size=100,
        mode_runtime_kwargs={},
        direct_phase_plan={"include_legacy_baseline": True},
        benchmark_workers_for_storage=2,
        benchmark_workers_for_dataframe=2,
        query_root_class_local="Gene",
        query_views_local=["Gene.primaryIdentifier"],
        query_joins_local=[],
    )

    assert result["benchmark_semantics"]["offline_replay_stage_io"] is True
    assert result["parity"]["storage_compare_equivalent"] is True
    assert result["parity"]["new_only_large_equivalent"] is True
    assert result["parity"]["join_engines_equivalent"] is True


def test_storage_stage_strict_parity_fails_when_mismatch(tmp_path):
    benchmarks_module = _reload_benchmarks_module()
    args = _build_args(tmp_path, strict_parity=True)

    for base in (args.csv_old_path, args.parquet_compare_path, args.csv_new_path, args.parquet_new_path):
        _touch(benchmarks_module._query_output_path(base, "complex"))

    io_runtime = _ReplayIORuntime(parity_equivalent=False)
    with pytest.raises(RuntimeError):
        benchmarks_module._run_storage_dataframe_join_benchmark(
            io_runtime=io_runtime,
            args=args,
            query_kind="complex",
            io_page_size=100,
            mode_runtime_kwargs={},
            direct_phase_plan={"include_legacy_baseline": True},
            benchmark_workers_for_storage=2,
            benchmark_workers_for_dataframe=2,
            query_root_class_local="Gene",
            query_views_local=["Gene.primaryIdentifier"],
            query_joins_local=[],
        )
