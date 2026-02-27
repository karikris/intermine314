import unittest
import time
from concurrent.futures import Future
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
from intermine314.query import builder as query_builder


class _BenchmarkFakeQuery:
    def count(self):
        return 0

    def results(self, row="dict", start=0, size=None):
        stop = start + int(size or 0)
        for value in range(start, stop):
            yield {"value": value, "row": row}


class _BenchmarkImmediateFuture(Future):
    def __init__(self, fn, arg, executor):
        super().__init__()
        self._executor = executor
        self._released = False
        try:
            value = fn(arg)
        except Exception as exc:
            self.set_exception(exc)
        else:
            self.set_result(value)

    def result(self, timeout=None):
        if not self._released:
            self._executor.current_pending -= 1
            self._released = True
        return super().result(timeout=timeout)


class _BenchmarkTrackingExecutor:
    instances = []

    def __init__(self, *args, **kwargs):
        self.current_pending = 0
        self.max_pending = 0
        _BenchmarkTrackingExecutor.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, arg):
        self.current_pending += 1
        self.max_pending = max(self.max_pending, self.current_pending)
        return _BenchmarkImmediateFuture(fn, arg, self)

    def map(self, *args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("benchmark fallback path should use bounded submit window")


class TestBenchmarkFetch(unittest.TestCase):
    def test_parse_positive_int_csv_supports_auto(self):
        self.assertEqual(parse_positive_int_csv("auto", "--workers", allow_auto=True), [])

    def test_parse_positive_int_csv_validates_values(self):
        with self.assertRaises(ValueError):
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
        self.assertEqual(ordered, 4)
        self.assertEqual(unordered, 8)
        self.assertEqual(clamped, 2)

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
        self.assertGreater(up, 4)
        self.assertLess(down, 8)

    def test_build_matrix_scenarios_prefers_target_overrides(self):
        args = SimpleNamespace(
            matrix_small_rows="SMALL_MATRIX_ROWS",
            matrix_large_rows="LARGE_MATRIX_ROWS",
            matrix_small_profile="benchmark_profile_3",
            matrix_large_profile="benchmark_profile_1",
        )
        target_settings = {
            "matrix_small_rows": "1000,2000,3000",
            "matrix_large_rows": "4000,5000,6000",
            "matrix_small_profile": "benchmark_profile_4",
            "matrix_large_profile": "benchmark_profile_2",
        }
        scenarios = build_matrix_scenarios(args, target_settings)
        self.assertEqual(len(scenarios), 6)
        self.assertEqual([s["rows_target"] for s in scenarios[:3]], [1000, 2000, 3000])
        self.assertEqual([s["rows_target"] for s in scenarios[3:]], [4000, 5000, 6000])
        self.assertEqual(scenarios[0]["profile"], "benchmark_profile_4")
        self.assertEqual(scenarios[3]["profile"], "benchmark_profile_2")

    def test_resolve_execution_plan_with_explicit_workers(self):
        plan = resolve_execution_plan(
            mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
            rows_target=10000,
            explicit_workers=[4, 8],
            benchmark_profile="auto",
            phase_default_include_legacy=True,
        )
        self.assertEqual(plan["name"], "workers_override")
        self.assertEqual(plan["workers"], [4, 8])
        self.assertTrue(plan["include_legacy_baseline"])

    def test_resolve_execution_plan_uses_named_profile(self):
        plan = resolve_execution_plan(
            mine_url="https://bar.utoronto.ca/thalemine/service",
            rows_target=10000,
            explicit_workers=[],
            benchmark_profile="benchmark_profile_3",
            phase_default_include_legacy=False,
        )
        self.assertEqual(plan["name"], "benchmark_profile_3")
        self.assertFalse(plan["include_legacy_baseline"])
        self.assertGreater(len(plan["workers"]), 0)

    def test_make_query_requires_legacy_package_for_legacy_mode(self):
        with self.assertRaises(RuntimeError):
            make_query(None, "https://example.org/service", "Gene", ["Gene.primaryIdentifier"], [])

    def test_run_replicated_fetch_reports_stage_timings(self):
        def _fake_run_mode_with_runtime(
            *,
            mode,
            mine_url,
            rows_target,
            page_size,
            workers,
            csv_out_path,
            runtime_kwargs,
            query_root_class,
            query_views,
            query_joins,
        ):
            del mine_url, rows_target, page_size, csv_out_path, runtime_kwargs, query_root_class, query_views, query_joins
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

    def test_long_ordered_export_benchmark_and_memory_guard(self):
        _BenchmarkTrackingExecutor.instances.clear()
        fake_query = _BenchmarkFakeQuery()
        started = time.perf_counter()
        with patch("intermine314.query.builder._EXECUTOR_MAP_SUPPORTS_BUFFERSIZE", False):
            with patch("intermine314.query.builder.ThreadPoolExecutor", _BenchmarkTrackingExecutor):
                rows = 0
                for _ in query_builder.Query._run_parallel_offset(
                    fake_query,
                    row="dict",
                    start=0,
                    size=8000,
                    page_size=1,
                    max_workers=8,
                    order_mode="ordered",
                    inflight_limit=4,
                    ordered_window_pages=2,
                ):
                    rows += 1
        elapsed = time.perf_counter() - started

        self.assertEqual(rows, 8000)
        self.assertEqual(len(_BenchmarkTrackingExecutor.instances), 1)
        self.assertLessEqual(_BenchmarkTrackingExecutor.instances[0].max_pending, 4)
        # Benchmark smoke check: synthetic long ordered export should remain fast.
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
