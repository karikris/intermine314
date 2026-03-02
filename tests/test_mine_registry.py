import intermine314.registry.mines as mine_registry
from intermine314.registry.mines import (
    DEFAULT_BENCHMARK_LARGE_PROFILE,
    DEFAULT_BENCHMARK_PROFILES,
    DEFAULT_BENCHMARK_SMALL_PROFILE,
    PRODUCTION_PROFILE_ELT_DEFAULT,
    PRODUCTION_PROFILE_ELT_FULL,
    PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
    THALEMINE_BROWSER_USER_AGENT,
    resolve_benchmark_plan,
    resolve_mine_user_agent,
    resolve_production_plan,
    resolve_production_resource_profile,
    resolve_preferred_workers,
)

LEGUMEMINE_ROOT = "https://mines.legumeinfo.org/legumemine/service"
MAIZEMINE_ROOT = "https://maizemine.rnet.missouri.edu/maizemine/service"
THALEMINE_ROOT = "https://bar.utoronto.ca/thalemine/service"
UNKNOWN_ROOT = "https://example.org/unknown/service"
BENCHMARK_MATRIX_ROWS = (5_000, 10_000, 25_000, 50_000, 100_000)


def test_registry_preferred_workers_resolve_by_mine_tier():
    for rows in BENCHMARK_MATRIX_ROWS:
        assert resolve_preferred_workers(LEGUMEMINE_ROOT, rows, 4) == 4
        assert resolve_preferred_workers(MAIZEMINE_ROOT, rows, 4) == 8
        assert resolve_preferred_workers(THALEMINE_ROOT, rows, 4) == 16
        assert resolve_preferred_workers(UNKNOWN_ROOT, rows, 6) == 6


def test_registry_benchmark_plan_selection_and_fallback_are_stable():
    thale_5k = resolve_benchmark_plan(THALEMINE_ROOT, 5_000)
    thale_100k = resolve_benchmark_plan(THALEMINE_ROOT, 100_000)
    assert thale_5k["name"] == DEFAULT_BENCHMARK_SMALL_PROFILE
    assert thale_100k["name"] == DEFAULT_BENCHMARK_LARGE_PROFILE

    unknown = resolve_benchmark_plan(UNKNOWN_ROOT, 10_000)
    assert unknown["name"] == DEFAULT_BENCHMARK_SMALL_PROFILE
    assert unknown["workers"] == DEFAULT_BENCHMARK_PROFILES[DEFAULT_BENCHMARK_SMALL_PROFILE]["workers"]


def test_registry_production_plan_selection_and_resource_profile_defaults():
    legume = resolve_production_plan(LEGUMEMINE_ROOT, 10_000, workflow="elt")
    maize = resolve_production_plan(MAIZEMINE_ROOT, 10_000, workflow="elt")
    thale = resolve_production_plan(THALEMINE_ROOT, 100_000, workflow="elt")
    assert legume["name"] == PRODUCTION_PROFILE_ELT_DEFAULT
    assert maize["name"] == PRODUCTION_PROFILE_ELT_SERVER_LIMITED
    assert thale["name"] == PRODUCTION_PROFILE_ELT_FULL
    assert resolve_production_resource_profile(LEGUMEMINE_ROOT, 25_000, workflow="elt") == "default"


def test_registry_profile_data_exposes_user_agent_and_normalized_match_fields():
    assert resolve_mine_user_agent(THALEMINE_ROOT) == THALEMINE_BROWSER_USER_AGENT
    assert resolve_mine_user_agent(UNKNOWN_ROOT) is None

    registry = mine_registry._load_registry()
    legume = registry["mines"]["legumemine"]
    assert legume["host_patterns_normalized"] == ("mines.legumeinfo.org",)
    assert legume["path_prefixes_normalized"] == ("/legumemine",)
