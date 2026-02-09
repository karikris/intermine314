import unittest

from intermine314.mine_registry import (
    DEFAULT_BENCHMARK_LARGE_PROFILE,
    DEFAULT_BENCHMARK_PROFILES,
    DEFAULT_BENCHMARK_SMALL_PROFILE,
    resolve_benchmark_plan,
    resolve_named_benchmark_profile,
    resolve_preferred_workers,
)


FALLBACK_WORKERS = 4
SMALL_DATASET_ROWS = 10000
THRESHOLD_ROWS = 50000
LARGE_DATASET_ROWS = 50001
VERY_LARGE_DATASET_ROWS = 200000

LEGUMEMINE_ROOT = "https://mines.legumeinfo.org/legumemine/service"
MAIZEMINE_ROOT = "https://maizemine.rnet.missouri.edu/maizemine/service"
STANDARD_MINE_ROOTS = (
    "https://bar.utoronto.ca/thalemine/service",
    "https://urgi.versailles.inra.fr/WheatMine/service",
    "https://urgi.versailles.inra.fr/OakMine_PM1N/service",
    "https://urgi.versailles.inrae.fr/WheatMine/service",
    "https://urgi.versailles.inrae.fr/OakMine_PM1N/service",
)


class TestMineRegistry(unittest.TestCase):
    def _assert_workers_for_roots(self, roots, size, expected):
        for root in roots:
            with self.subTest(root=root, size=size):
                self.assertEqual(resolve_preferred_workers(root, size, FALLBACK_WORKERS), expected)

    def test_named_benchmark_profiles_match_defaults(self):
        for name, profile in DEFAULT_BENCHMARK_PROFILES.items():
            with self.subTest(profile=name):
                resolved = resolve_named_benchmark_profile(name)
                self.assertEqual(resolved["include_legacy_baseline"], bool(profile["include_legacy_baseline"]))
                self.assertEqual(resolved["workers"], profile["workers"])

    def test_legumemine_uses_fixed_worker_count(self):
        self._assert_workers_for_roots([LEGUMEMINE_ROOT], SMALL_DATASET_ROWS, 4)
        self._assert_workers_for_roots([LEGUMEMINE_ROOT], THRESHOLD_ROWS, 4)
        self._assert_workers_for_roots([LEGUMEMINE_ROOT], LARGE_DATASET_ROWS, 4)
        self._assert_workers_for_roots([LEGUMEMINE_ROOT], None, 4)

    def test_maizemine_uses_capped_worker_count(self):
        self._assert_workers_for_roots([MAIZEMINE_ROOT], SMALL_DATASET_ROWS, 8)
        self._assert_workers_for_roots([MAIZEMINE_ROOT], THRESHOLD_ROWS, 8)
        self._assert_workers_for_roots([MAIZEMINE_ROOT], LARGE_DATASET_ROWS, 8)
        self._assert_workers_for_roots([MAIZEMINE_ROOT], None, 8)

    def test_standard_mines_reduce_workers_above_threshold(self):
        self._assert_workers_for_roots(STANDARD_MINE_ROOTS, THRESHOLD_ROWS, 16)
        self._assert_workers_for_roots(STANDARD_MINE_ROOTS, LARGE_DATASET_ROWS, 12)
        self._assert_workers_for_roots(STANDARD_MINE_ROOTS, None, 12)

    def test_legumemine_benchmark_plan_switch(self):
        small = resolve_benchmark_plan(LEGUMEMINE_ROOT, SMALL_DATASET_ROWS)
        large = resolve_benchmark_plan(LEGUMEMINE_ROOT, VERY_LARGE_DATASET_ROWS)
        self.assertEqual(small["name"], "benchmark_profile_4")
        self.assertEqual(small["workers"], [4, 6, 8])
        self.assertTrue(small["include_legacy_baseline"])
        self.assertEqual(large["name"], "benchmark_profile_2")
        self.assertEqual(large["workers"], [4, 6, 8])
        self.assertFalse(large["include_legacy_baseline"])

    def test_standard_mines_benchmark_profile_switch(self):
        for root in STANDARD_MINE_ROOTS:
            with self.subTest(root=root):
                small = resolve_benchmark_plan(root, SMALL_DATASET_ROWS)
                large = resolve_benchmark_plan(root, VERY_LARGE_DATASET_ROWS)
                self.assertEqual(small["name"], DEFAULT_BENCHMARK_SMALL_PROFILE)
                self.assertEqual(large["name"], DEFAULT_BENCHMARK_LARGE_PROFILE)

    def test_maizemine_benchmark_profiles_use_split_profiles(self):
        small = resolve_benchmark_plan(MAIZEMINE_ROOT, SMALL_DATASET_ROWS)
        large = resolve_benchmark_plan(MAIZEMINE_ROOT, VERY_LARGE_DATASET_ROWS)
        self.assertEqual(small["name"], "benchmark_profile_4")
        self.assertEqual(small["workers"], [4, 6, 8])
        self.assertTrue(small["include_legacy_baseline"])
        self.assertEqual(large["name"], "benchmark_profile_2")
        self.assertEqual(large["workers"], [4, 6, 8])
        self.assertFalse(large["include_legacy_baseline"])

    def test_unknown_mine_uses_fallback(self):
        unknown_root = "https://example.org/unknown/service"
        self.assertEqual(resolve_preferred_workers(unknown_root, THRESHOLD_ROWS, 6), 6)
        self.assertEqual(resolve_preferred_workers("", THRESHOLD_ROWS, 6), 6)

        fallback_plan = resolve_benchmark_plan(unknown_root, THRESHOLD_ROWS)
        self.assertEqual(fallback_plan["name"], DEFAULT_BENCHMARK_SMALL_PROFILE)
        self.assertEqual(fallback_plan["workers"], DEFAULT_BENCHMARK_PROFILES[DEFAULT_BENCHMARK_SMALL_PROFILE]["workers"])


if __name__ == "__main__":
    unittest.main()
