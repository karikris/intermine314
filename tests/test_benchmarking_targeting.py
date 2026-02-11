import unittest
from unittest.mock import patch

from benchmarks.bench_targeting import (
    normalize_target_settings,
    normalize_targeted_settings,
    profile_for_rows,
    resolve_reachable_mine_url,
)


class TestBenchmarkTargeting(unittest.TestCase):
    def test_profile_for_rows_auto_switch(self):
        target_settings = {
            "profile_switch_rows": 50000,
            "profile_small": "benchmark_profile_3",
            "profile_large": "benchmark_profile_1",
        }
        self.assertEqual(profile_for_rows("auto", target_settings, 10000), "benchmark_profile_3")
        self.assertEqual(profile_for_rows("auto", target_settings, 100000), "benchmark_profile_1")
        self.assertEqual(profile_for_rows("benchmark_profile_4", target_settings, 100000), "benchmark_profile_4")

    def test_normalize_targeted_settings_applies_table_profile(self):
        target_defaults = {
            "table_profiles": {
                "core": {
                    "root_class": "Gene",
                    "views": ["Gene.primaryIdentifier"],
                    "joins": [],
                }
            }
        }
        target_settings = {
            "targeted_exports": {
                "enabled": True,
                "tables": [
                    {
                        "name": "core_gene",
                        "table_profile": "core",
                        "views": "Gene.symbol",
                        "joins": "Gene.organism",
                    }
                ],
            }
        }
        normalized = normalize_targeted_settings(target_settings, target_defaults)
        table = normalized["tables"][0]
        self.assertEqual(table["root_class"], "Gene")
        self.assertEqual(table["views"], ["Gene.symbol"])
        self.assertEqual(table["joins"], ["Gene.organism"])
        self.assertNotIn("table_profile", table)

    def test_normalize_target_settings_merges_query_profile_and_defaults(self):
        target_defaults = {
            "query_profiles": {
                "gene": {
                    "root_class": "Gene",
                    "views": ["Gene.primaryIdentifier"],
                    "joins": [],
                }
            }
        }
        target_config = {
            "defaults": {
                "targeted_exports": {
                    "enabled": True,
                    "template_limit": 40,
                }
            },
            "targets": {
                "demo": {
                    "query_profile": "gene",
                    "views": "Gene.primaryIdentifier,Gene.name",
                    "joins": "Gene.organism",
                    "targeted_exports": {
                        "template_limit": 10,
                    },
                }
            },
        }

        settings = normalize_target_settings("demo", target_config, target_defaults)
        self.assertIsNotNone(settings)
        self.assertEqual(settings["root_class"], "Gene")
        self.assertEqual(settings["views"], ["Gene.primaryIdentifier", "Gene.name"])
        self.assertEqual(settings["joins"], ["Gene.organism"])
        self.assertEqual(settings["targeted_exports"]["enabled"], True)
        self.assertEqual(settings["targeted_exports"]["template_limit"], 10)

    def test_resolve_reachable_mine_url_passes_timeout(self):
        calls = []

        class _Service:
            def __init__(self, root, request_timeout=None):
                calls.append((root, request_timeout))
                self._version = 35

            @property
            def version(self):
                return self._version

        with patch("benchmarks.bench_targeting.NewService", _Service):
            resolved, errors = resolve_reachable_mine_url(
                "https://example.org/service",
                None,
                request_timeout=9,
            )

        self.assertEqual(resolved, "https://example.org/service")
        self.assertEqual(errors, [])
        self.assertEqual(calls, [("https://example.org/service", 9)])

    def test_resolve_reachable_mine_url_tries_fallback_with_timeout(self):
        calls = []

        class _Service:
            def __init__(self, root, request_timeout=None):
                calls.append((root, request_timeout))
                if root == "https://primary.example/service":
                    raise RuntimeError("down")
                self._version = 35

            @property
            def version(self):
                return self._version

        settings = {
            "endpoint_fallbacks": [
                "https://fallback.example/service",
            ]
        }

        with patch("benchmarks.bench_targeting.NewService", _Service):
            resolved, errors = resolve_reachable_mine_url(
                "https://primary.example/service",
                settings,
                request_timeout=5,
            )

        self.assertEqual(resolved, "https://fallback.example/service")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["url"], "https://primary.example/service")
        self.assertEqual(calls, [("https://primary.example/service", 5), ("https://fallback.example/service", 5)])


if __name__ == "__main__":
    unittest.main()
