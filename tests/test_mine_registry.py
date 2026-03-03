import intermine314.registry.mines as mine_registry
from intermine314.registry.mines import (
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


def test_registry_production_plan_selection_and_resource_profile_defaults():
    legume = resolve_production_plan(LEGUMEMINE_ROOT, 10_000, workflow="elt")
    maize = resolve_production_plan(MAIZEMINE_ROOT, 10_000, workflow="elt")
    thale = resolve_production_plan(THALEMINE_ROOT, 100_000, workflow="elt")
    assert legume["name"] == "elt_default_w4"
    assert maize["name"] == "elt_server_limited_w8"
    assert thale["name"] == "elt_full_w16"
    assert resolve_production_resource_profile(LEGUMEMINE_ROOT, 25_000, workflow="elt") == "default"


def test_registry_profile_data_exposes_user_agent_and_normalized_match_fields():
    registry = mine_registry._load_registry()
    thale = registry["mines"]["thalemine"]
    assert resolve_mine_user_agent(THALEMINE_ROOT) == thale["user_agent"]
    assert resolve_mine_user_agent(UNKNOWN_ROOT) is None

    legume = registry["mines"]["legumemine"]
    assert legume["host_patterns_normalized"] == ("mines.legumeinfo.org",)
    assert legume["path_prefixes_normalized"] == ("/legumemine",)
