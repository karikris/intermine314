from unittest.mock import patch

from benchmarks.bench_targeting import (
    normalize_target_settings,
    normalize_targeted_settings,
    resolve_benchmark_profile,
    resolve_reachable_mine_url,
)


class TestBenchmarkTargeting:
    def test_resolve_benchmark_profile_uses_target_profile(self):
        target_settings = {
            "benchmark_profile": "server_restricted",
            "server_restricted": True,
        }
        assert resolve_benchmark_profile("auto", target_settings) == "server_restricted"
        assert resolve_benchmark_profile("non_restricted", target_settings) == "non_restricted"

    def test_resolve_benchmark_profile_can_derive_from_server_restricted_flag(self):
        assert resolve_benchmark_profile("auto", {"server_restricted": True}) == "server_restricted"
        assert resolve_benchmark_profile("auto", {"server_restricted": False}) == "non_restricted"

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
        assert table["root_class"] == "Gene"
        assert table["views"] == ["Gene.symbol"]
        assert table["joins"] == ["Gene.organism"]
        assert "table_profile" not in table

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
        assert settings is not None
        assert settings["root_class"] == "Gene"
        assert settings["views"] == ["Gene.primaryIdentifier", "Gene.name"]
        assert settings["joins"] == ["Gene.organism"]
        assert settings["targeted_exports"]["enabled"] == True
        assert settings["targeted_exports"]["template_limit"] == 10

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

        assert resolved == "https://example.org/service"
        assert errors == []
        assert calls == [("https://example.org/service", 9)]

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

        assert resolved == "https://fallback.example/service"
        assert len(errors) == 1
        assert errors[0]["url"] == "https://primary.example/service"
        assert calls == [("https://primary.example/service", 5), ("https://fallback.example/service", 5)]
