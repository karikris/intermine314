import unittest

from intermine314.mine_registry import resolve_benchmark_plan, resolve_named_benchmark_profile, resolve_preferred_workers


class TestMineRegistry(unittest.TestCase):
    def test_legumemine_threshold_profile(self):
        root = "https://mines.legumeinfo.org/legumemine/service"
        self.assertEqual(resolve_preferred_workers(root, 10000, 16), 4)
        self.assertEqual(resolve_preferred_workers(root, 50000, 16), 4)
        self.assertEqual(resolve_preferred_workers(root, 50001, 16), 4)
        self.assertEqual(resolve_preferred_workers(root, None, 16), 4)

    def test_named_benchmark_profiles(self):
        profile_1 = resolve_named_benchmark_profile("benchmark_profile_1")
        self.assertTrue(profile_1["include_legacy_baseline"])
        self.assertEqual(profile_1["workers"], [2, 4, 6, 8, 10, 12, 14, 16, 18])

        profile_2 = resolve_named_benchmark_profile("benchmark_profile_2")
        self.assertFalse(profile_2["include_legacy_baseline"])
        self.assertEqual(profile_2["workers"], [4, 8, 12, 16])

        profile_3 = resolve_named_benchmark_profile("benchmark_profile_3")
        self.assertFalse(profile_3["include_legacy_baseline"])
        self.assertEqual(profile_3["workers"], [4, 6, 8])

        profile_4 = resolve_named_benchmark_profile("benchmark_profile_4")
        self.assertTrue(profile_4["include_legacy_baseline"])
        self.assertEqual(profile_4["workers"], [4, 6, 8])

    def test_legumemine_benchmark_plan_switch(self):
        root = "https://mines.legumeinfo.org/legumemine/service"
        small = resolve_benchmark_plan(root, 10000)
        large = resolve_benchmark_plan(root, 100000)
        self.assertEqual(small["name"], "benchmark_small_workers_override")
        self.assertEqual(small["workers"], [4])
        self.assertFalse(small["include_legacy_baseline"])
        self.assertEqual(large["name"], "benchmark_profile_3")
        self.assertEqual(large["workers"], [4, 6, 8])
        self.assertFalse(large["include_legacy_baseline"])

    def test_other_supported_mines_default_to_16(self):
        self.assertEqual(resolve_preferred_workers("https://maizemine.rnet.missouri.edu/maizemine/service", 50000, 4), 16)
        self.assertEqual(resolve_preferred_workers("https://bar.utoronto.ca/thalemine/service", 50000, 4), 16)
        self.assertEqual(resolve_preferred_workers("https://urgi.versailles.inra.fr/WheatMine/service", 50000, 4), 16)
        self.assertEqual(resolve_preferred_workers("https://urgi.versailles.inra.fr/OakMine_PM1N/service", 50000, 4), 16)

    def test_unknown_mine_uses_fallback(self):
        self.assertEqual(resolve_preferred_workers("https://example.org/unknown/service", 50000, 6), 6)
        self.assertEqual(resolve_preferred_workers("", 50000, 6), 6)
        fallback_plan = resolve_benchmark_plan("https://example.org/unknown/service", 50000)
        self.assertEqual(fallback_plan["name"], "benchmark_profile_1")
        self.assertEqual(fallback_plan["workers"], [2, 4, 6, 8, 10, 12, 14, 16, 18])
