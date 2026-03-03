import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.bench_fetch import (
    ModeRun,
    build_matrix_scenarios,
    initial_chunk_pages,
    make_query,
    parse_positive_int_csv,
    resolve_execution_plan,
    run_replicated_fetch_benchmarks,
    tune_chunk_pages,
)
import pytest


class _BenchmarkFakeQuery:
    def count(self):
        return 0

    def results(self, row="dict", start=0, size=None):
        stop = start + int(size or 0)
        for value in range(start, stop):
            yield {"value": value, "row": row}


class _BenchmarkTrackingExecutor:
    instances = []

    def __init__(self, *args, **kwargs):
        self.current_pending = 0
        self.max_pending = 0
        self.map_calls = 0
        self.map_buffersizes = []
        _BenchmarkTrackingExecutor.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
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


class TestBenchmarkFetch:
    def test_parse_positive_int_csv_supports_auto(self):
        assert parse_positive_int_csv("auto", "--workers", allow_auto=True) == []

    def test_parse_positive_int_csv_validates_values(self):
        with pytest.raises(ValueError):
            parse_positive_int_csv("4,0", "--workers")

    def test_initial_chunk_pages_obeys_ordering_and_bounds(self):
        ordered = initial_chunk_pages(
            workers=4,
            ordered_mode="ordered",
            large_query_mode=False,
            prefetch=None,
            inflight_limit=None,
            min_pages=2,
            max_pages=8,
        )
        unordered = initial_chunk_pages(
            workers=4,
            ordered_mode="unordered",
            large_query_mode=False,
            prefetch=None,
            inflight_limit=None,
            min_pages=2,
            max_pages=8,
        )
        clamped = initial_chunk_pages(
            workers=4,
            ordered_mode="ordered",
            large_query_mode=False,
            prefetch=1,
            inflight_limit=None,
            min_pages=2,
            max_pages=8,
        )
        assert ordered == 4
        assert unordered == 8
        assert clamped == 2

    def test_tune_chunk_pages_can_scale_up_and_down(self):
        up = tune_chunk_pages(
            current_pages=4,
            rows_fetched=4000,
            block_seconds=0.5,
            page_size=1000,
            target_seconds=2.0,
            min_pages=2,
            max_pages=16,
        )
        down = tune_chunk_pages(
            current_pages=8,
            rows_fetched=4000,
            block_seconds=10.0,
            page_size=1000,
            target_seconds=2.0,
            min_pages=2,
            max_pages=16,
        )
        assert up > 4
        assert down < 8

    def test_build_matrix_scenarios_prefers_target_overrides(self):
        args = SimpleNamespace(
            matrix_rows="MATRIX_ROWS",
            matrix_profile="benchmark_profile_3",
        )
        target_settings = {
            "matrix_rows": "1000,2000,3000,4000,5000",
            "matrix_profile": "benchmark_profile_4",
        }
        scenarios = build_matrix_scenarios(args, target_settings)
        assert len(scenarios) == 5
        assert [s["rows_target"] for s in scenarios] == [1000, 2000, 3000, 4000, 5000]
        assert scenarios[0]["profile"] == "benchmark_profile_4"
        assert scenarios[4]["profile"] == "benchmark_profile_4"

    def test_resolve_execution_plan_with_explicit_workers(self):
        plan = resolve_execution_plan(
            mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
            rows_target=10000,
            explicit_workers=[4, 8],
            benchmark_profile="auto",
            phase_default_include_legacy=True,
        )
        assert plan["name"] == "workers_override"
        assert plan["workers"] == [4, 8]
        assert plan["include_legacy_baseline"]

    def test_resolve_execution_plan_uses_named_profile(self):
        plan = resolve_execution_plan(
            mine_url="https://bar.utoronto.ca/thalemine/service",
            rows_target=10000,
            explicit_workers=[],
            benchmark_profile="benchmark_profile_3",
            phase_default_include_legacy=False,
        )
        assert plan["name"] == "benchmark_profile_3"
        assert not plan["include_legacy_baseline"]
        assert len(plan["workers"]) > 0

    def test_make_query_requires_legacy_package_for_legacy_mode(self):
        with pytest.raises(RuntimeError):
            make_query(None, "https://example.org/service", "Gene", ["Gene.primaryIdentifier"], [])

    def test_run_replicated_fetch_reports_stage_timings(self):
        def _fake_run_mode_with_runtime(
            *,
            mode,
            mine_url,
            rows_target,
            page_size,
            workers,
            runtime_kwargs,
            query_root_class,
            query_views,
            query_joins,
        ):
            del mine_url, rows_target, page_size, runtime_kwargs, query_root_class, query_views, query_joins
            return ModeRun(
                mode=mode,
                repetition=-1,
                seconds=1.0,
                rows=100,
                rows_per_s=100.0,
                retries=0,
                available_rows_per_pass=100,
                effective_workers=workers,
                block_stats={},
                stage_timings={
                    "query_init_seconds": 0.1,
                    "count_seconds": 0.2,
                    "stream_seconds": 1.0,
                },
            )

        with patch("benchmarks.bench_fetch.run_mode_with_runtime", side_effect=_fake_run_mode_with_runtime):
            result = run_replicated_fetch_benchmarks(
                phase_name="stage_timing_smoke",
                mine_url="https://example.org/mine",
                rows_target=100,
                repetitions=2,
                workers=[2],
                include_legacy_baseline=False,
                page_size=50,
                legacy_batch_size=50,
                parallel_window_factor=2,
                auto_chunking=True,
                chunk_target_seconds=2.0,
                chunk_min_pages=1,
                chunk_max_pages=8,
                ordered_mode="ordered",
                ordered_window_pages=10,
                parallel_profile="default",
                large_query_mode=False,
                prefetch=None,
                inflight_limit=None,
                max_inflight_bytes_estimate=None,
                randomize_mode_order=False,
                sleep_seconds=0.0,
                max_retries=1,
                query_root_class="Gene",
                query_views=["Gene.primaryIdentifier"],
                query_joins=[],
            )

        mode_summary = result["results"]["intermine314_w2"]
        assert "stage_timings_seconds" in mode_summary
        assert mode_summary["stage_timings_seconds"]["query_init_seconds"]["mean"] == 0.1


def test_benchmarks_main_emits_storage_compare_payload(monkeypatch, tmp_path, capsys):
    import benchmarks.benchmarks as bench_entry

    monkeypatch.setattr(
        bench_entry,
        "resolve_execution_plan",
        lambda **_kwargs: {
            "name": "benchmark_profile_3",
            "workers": [3, 6, 9],
            "include_legacy_baseline": False,
        },
    )
    monkeypatch.setattr(
        bench_entry,
        "run_fetch_phase",
        lambda **kwargs: {"phase_name": kwargs["phase_name"], "status": "ok"},
    )

    captured: list[dict] = []
    fake_storage_module = types.ModuleType("benchmarks.bench_storage_compare")

    def _fake_run_storage_compare(**kwargs):
        captured.append(dict(kwargs))
        return {
            "schema_version": "legacy_storage_compare_v1",
            "status": "ok",
            "transport_mode": kwargs["transport_mode"],
            "parity": {"row_count_match": True, "sample_hash_match": True},
        }

    fake_storage_module.run_storage_compare = _fake_run_storage_compare
    monkeypatch.setitem(sys.modules, "benchmarks.bench_storage_compare", fake_storage_module)

    code = bench_entry.main(
        [
            "--mine-url",
            "https://example.org/service",
            "--rows",
            "5000",
            "--repetitions",
            "1",
            "--workers",
            "auto",
            "--transport-modes",
            "direct,tor",
            "--storage-compare",
            "--storage-output-dir",
            str(tmp_path),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime"]["storage_compare"] is True
    assert payload["runtime"]["storage_compare_workers"] == 9
    assert sorted(payload["storage_compare_by_transport"].keys()) == ["direct", "tor"]
    assert payload["storage_compare_by_transport"]["direct"]["status"] == "ok"
    assert payload["storage_compare_by_transport"]["tor"]["status"] == "ok"

    assert len(captured) == 2
    assert {item["transport_mode"] for item in captured} == {"direct", "tor"}
    assert all(item["workers"] == 9 for item in captured)
    assert all(str(item["output_dir"]).startswith(str(tmp_path)) for item in captured)
