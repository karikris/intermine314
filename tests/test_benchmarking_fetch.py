import unittest
from types import SimpleNamespace

from benchmarks.bench_fetch import (
    build_matrix_scenarios,
    initial_chunk_pages,
    make_query,
    parse_positive_int_csv,
    resolve_execution_plan,
    tune_chunk_pages,
)


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


if __name__ == "__main__":
    unittest.main()
