import intermine314.registry.mines as mine_registry
from intermine314.registry.mines import (
    DEFAULT_BENCHMARK_LARGE_PROFILE,
    DEFAULT_BENCHMARK_PROFILES,
    DEFAULT_BENCHMARK_SMALL_PROFILE,
    PRODUCTION_PROFILE_ELT_DEFAULT,
    PRODUCTION_PROFILE_ELT_FULL,
    PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
    PRODUCTION_PROFILE_ETL_DEFAULT,
    PRODUCTION_PROFILE_ETL_FULL,
    PRODUCTION_PROFILE_ETL_SERVER_LIMITED,
    THALEMINE_BROWSER_USER_AGENT,
    resolve_benchmark_plan,
    resolve_mine_user_agent,
    resolve_named_benchmark_profile,
    resolve_production_plan,
    resolve_production_resource_profile,
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


class TestMineRegistry:
    def _assert_workers_for_roots(self, roots, size, expected):
        for root in roots:
            assert resolve_preferred_workers(root, size, FALLBACK_WORKERS) == expected

    def test_named_benchmark_profiles_match_defaults(self):
        for name, profile in DEFAULT_BENCHMARK_PROFILES.items():
            resolved = resolve_named_benchmark_profile(name)
            assert resolved["include_legacy_baseline"] == bool(profile["include_legacy_baseline"])
            assert resolved["workers"] == profile["workers"]

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
        self._assert_workers_for_roots(STANDARD_MINE_ROOTS, LARGE_DATASET_ROWS, 16)
        self._assert_workers_for_roots(STANDARD_MINE_ROOTS, None, 16)

    def test_legumemine_benchmark_plan_switch(self):
        small = resolve_benchmark_plan(LEGUMEMINE_ROOT, SMALL_DATASET_ROWS)
        large = resolve_benchmark_plan(LEGUMEMINE_ROOT, VERY_LARGE_DATASET_ROWS)
        assert small["name"] == "benchmark_profile_4"
        assert small["workers"] == [4, 6, 8]
        assert small["include_legacy_baseline"]
        assert large["name"] == "benchmark_profile_2"
        assert large["workers"] == [4, 6, 8]
        assert not large["include_legacy_baseline"]

    def test_standard_mines_benchmark_profile_switch(self):
        for root in STANDARD_MINE_ROOTS:
            small = resolve_benchmark_plan(root, SMALL_DATASET_ROWS)
            large = resolve_benchmark_plan(root, VERY_LARGE_DATASET_ROWS)
            assert small["name"] == DEFAULT_BENCHMARK_SMALL_PROFILE
            assert large["name"] == DEFAULT_BENCHMARK_LARGE_PROFILE

    def test_maizemine_benchmark_profiles_use_split_profiles(self):
        small = resolve_benchmark_plan(MAIZEMINE_ROOT, SMALL_DATASET_ROWS)
        large = resolve_benchmark_plan(MAIZEMINE_ROOT, VERY_LARGE_DATASET_ROWS)
        assert small["name"] == "benchmark_profile_4"
        assert small["workers"] == [4, 6, 8]
        assert small["include_legacy_baseline"]
        assert large["name"] == "benchmark_profile_2"
        assert large["workers"] == [4, 6, 8]
        assert not large["include_legacy_baseline"]

    def test_unknown_mine_uses_fallback(self):
        unknown_root = "https://example.org/unknown/service"
        assert resolve_preferred_workers(unknown_root, THRESHOLD_ROWS, 6) == 6
        assert resolve_preferred_workers("", THRESHOLD_ROWS, 6) == 6

        fallback_plan = resolve_benchmark_plan(unknown_root, THRESHOLD_ROWS)
        assert fallback_plan["name"] == DEFAULT_BENCHMARK_SMALL_PROFILE
        assert fallback_plan["workers"] == DEFAULT_BENCHMARK_PROFILES[DEFAULT_BENCHMARK_SMALL_PROFILE]["workers"]

    def test_thalemine_profile_has_transport_user_agent(self):
        user_agent = resolve_mine_user_agent("https://bar.utoronto.ca/thalemine/service")
        assert user_agent == THALEMINE_BROWSER_USER_AGENT
        assert resolve_mine_user_agent("https://example.org/unknown/service") is None

    def test_legumemine_production_profiles_map_to_default_tier(self):
        elt_small = resolve_production_plan(LEGUMEMINE_ROOT, SMALL_DATASET_ROWS, workflow="elt")
        elt_large = resolve_production_plan(LEGUMEMINE_ROOT, VERY_LARGE_DATASET_ROWS, workflow="elt")
        etl_small = resolve_production_plan(LEGUMEMINE_ROOT, SMALL_DATASET_ROWS, workflow="etl")
        etl_large = resolve_production_plan(LEGUMEMINE_ROOT, VERY_LARGE_DATASET_ROWS, workflow="etl")
        assert elt_small["name"] == PRODUCTION_PROFILE_ELT_DEFAULT
        assert elt_large["name"] == PRODUCTION_PROFILE_ELT_DEFAULT
        assert etl_small["name"] == PRODUCTION_PROFILE_ETL_DEFAULT
        assert etl_large["name"] == PRODUCTION_PROFILE_ETL_DEFAULT
        assert elt_small["workers"] == 4
        assert etl_small["workers"] == 4
        assert elt_small["resource_profile"] == "default"
        assert etl_small["resource_profile"] == "default"

    def test_maizemine_production_profiles_map_to_server_limited_tier(self):
        elt = resolve_production_plan(MAIZEMINE_ROOT, SMALL_DATASET_ROWS, workflow="elt")
        etl = resolve_production_plan(MAIZEMINE_ROOT, LARGE_DATASET_ROWS, workflow="etl")
        assert elt["name"] == PRODUCTION_PROFILE_ELT_SERVER_LIMITED
        assert etl["name"] == PRODUCTION_PROFILE_ETL_SERVER_LIMITED
        assert elt["workers"] == 8
        assert etl["workers"] == 8

    def test_standard_mines_production_profiles_map_to_full_tier(self):
        for root in STANDARD_MINE_ROOTS:
            elt = resolve_production_plan(root, VERY_LARGE_DATASET_ROWS, workflow="elt")
            etl = resolve_production_plan(root, VERY_LARGE_DATASET_ROWS, workflow="etl")
            assert elt["name"] == PRODUCTION_PROFILE_ELT_FULL
            assert etl["name"] == PRODUCTION_PROFILE_ETL_FULL
            assert elt["workers"] == 16
            assert etl["workers"] == 16

    def test_named_production_profile_override(self):
        plan = resolve_production_plan(
            LEGUMEMINE_ROOT,
            SMALL_DATASET_ROWS,
            workflow="elt",
            production_profile=PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
        )
        assert plan["name"] == PRODUCTION_PROFILE_ELT_SERVER_LIMITED
        assert plan["workers"] == 8
        assert plan["resource_profile"] == "default"

    def test_resolve_production_resource_profile_defaults_to_default(self):
        profile = resolve_production_resource_profile(
            LEGUMEMINE_ROOT,
            SMALL_DATASET_ROWS,
            workflow="elt",
        )
        assert profile == "default"

    def test_loaded_registry_profiles_include_pre_normalized_matching_fields(self):
        registry = mine_registry._load_registry()
        legumemine_profile = registry["mines"]["legumemine"]

        assert "host_patterns_normalized" in legumemine_profile
        assert "path_prefixes_normalized" in legumemine_profile
        assert legumemine_profile["host_patterns_normalized"] == ("mines.legumeinfo.org",)
        assert legumemine_profile["path_prefixes_normalized"] == ("/legumemine",)

    def test_matches_profile_prefers_pre_normalized_fields(self):
        class _NoIter:
            def __iter__(self):
                raise AssertionError("raw pattern iterables should not be used when normalized fields exist")

        profile = {
            "host_patterns": _NoIter(),
            "path_prefixes": _NoIter(),
            "host_patterns_normalized": ("bar.utoronto.ca",),
            "path_prefixes_normalized": ("/thalemine",),
        }

        assert mine_registry._matches_profile(profile, "bar.utoronto.ca", "/thalemine")
        assert not mine_registry._matches_profile(profile, "example.org", "/thalemine")
