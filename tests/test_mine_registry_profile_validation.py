from pathlib import Path
from copy import deepcopy

import pytest

import intermine314.registry.mines as mine_registry


def _reset_registry_state(monkeypatch):
    monkeypatch.setattr(mine_registry, "_CACHE", None)
    monkeypatch.setattr(mine_registry, "_INVALID_CONFIG_ATTEMPTS", 0)
    monkeypatch.setattr(mine_registry, "_INVALID_CONFIG_ATTEMPTS_BY_FIELD", {})


def test_registry_production_plan_emits_canonical_literals():
    plan_elt = mine_registry.resolve_production_plan(
        "https://mines.legumeinfo.org/legumemine/service",
        10_000,
        workflow="elt",
    )
    plan_etl = mine_registry.resolve_production_plan(
        "https://mines.legumeinfo.org/legumemine/service",
        10_000,
        workflow="etl",
    )

    assert plan_elt["pipeline"] == mine_registry.PIPELINE_PARQUET_DUCKDB
    assert plan_etl["pipeline"] == mine_registry.PIPELINE_POLARS_DUCKDB
    assert plan_elt["parallel_profile"] == mine_registry.PRODUCTION_PARALLEL_PROFILE_DEFAULT
    assert plan_elt["ordered"] == mine_registry.PRODUCTION_ORDERED_DEFAULT


def test_registry_invalid_pipeline_fails_fast_during_normalization(monkeypatch):
    _reset_registry_state(monkeypatch)
    monkeypatch.setattr(
        mine_registry,
        "load_mine_parallel_preferences",
        lambda: {
            "production_profiles": {
                "elt_default_w4": {
                    "workflow": "elt",
                    "pipeline": "duckdb_only",
                }
            }
        },
    )
    monkeypatch.setattr(mine_registry, "resolve_mine_parallel_preferences_path", lambda: Path("/tmp/test.toml"))

    with pytest.raises(ValueError, match="production_profiles.elt_default_w4.pipeline"):
        mine_registry._load_registry()
    assert mine_registry._INVALID_CONFIG_ATTEMPTS == 1
    assert mine_registry._INVALID_CONFIG_ATTEMPTS_BY_FIELD["production_profiles.elt_default_w4.pipeline"] == 1


def test_registry_invalid_parallel_profile_fails_fast(monkeypatch):
    _reset_registry_state(monkeypatch)
    monkeypatch.setattr(
        mine_registry,
        "load_mine_parallel_preferences",
        lambda: {
            "production_profiles": {
                "elt_default_w4": {
                    "workflow": "elt",
                    "parallel_profile": "extreme_mode",
                }
            }
        },
    )
    monkeypatch.setattr(mine_registry, "resolve_mine_parallel_preferences_path", lambda: Path("/tmp/test.toml"))

    with pytest.raises(ValueError, match="production_profiles.elt_default_w4.parallel_profile"):
        mine_registry._load_registry()
    assert mine_registry._INVALID_CONFIG_ATTEMPTS == 1
    assert mine_registry._INVALID_CONFIG_ATTEMPTS_BY_FIELD["production_profiles.elt_default_w4.parallel_profile"] == 1


def test_registry_invalid_ordered_mode_fails_fast(monkeypatch):
    _reset_registry_state(monkeypatch)
    monkeypatch.setattr(
        mine_registry,
        "load_mine_parallel_preferences",
        lambda: {
            "production_profiles": {
                "elt_default_w4": {
                    "workflow": "elt",
                    "ordered": "sideways",
                }
            }
        },
    )
    monkeypatch.setattr(mine_registry, "resolve_mine_parallel_preferences_path", lambda: Path("/tmp/test.toml"))

    with pytest.raises(ValueError, match="production_profiles.elt_default_w4.ordered"):
        mine_registry._load_registry()
    assert mine_registry._INVALID_CONFIG_ATTEMPTS == 1
    assert mine_registry._INVALID_CONFIG_ATTEMPTS_BY_FIELD["production_profiles.elt_default_w4.ordered"] == 1


def test_registry_preferences_metrics_include_pipeline_and_profile_distribution(monkeypatch):
    _reset_registry_state(monkeypatch)
    metrics = mine_registry.registry_preferences_metrics()

    pipeline_distribution = metrics["pipeline_distribution"]
    parallel_profile_distribution = metrics["parallel_profile_distribution"]
    production_workers_distribution = metrics["production_workers_distribution"]
    mine_default_workers_distribution = metrics["mine_default_workers_distribution"]
    mine_threshold_rows_distribution = metrics["mine_threshold_rows_distribution"]

    assert pipeline_distribution[mine_registry.PIPELINE_PARQUET_DUCKDB] >= 1
    assert pipeline_distribution[mine_registry.PIPELINE_POLARS_DUCKDB] >= 1
    assert parallel_profile_distribution[mine_registry.PRODUCTION_PARALLEL_PROFILE_DEFAULT] >= 1
    assert sum(production_workers_distribution.values()) >= 1
    assert sum(mine_default_workers_distribution.values()) >= 1
    assert sum(mine_threshold_rows_distribution.values()) >= 1
    assert metrics["invalid_config_attempts_by_field"] == {}


def test_registry_overlay_merges_without_mutating_python_defaults(monkeypatch):
    _reset_registry_state(monkeypatch)
    baseline_benchmark = deepcopy(
        mine_registry.DEFAULT_BENCHMARK_PROFILES[mine_registry.DEFAULT_BENCHMARK_SMALL_PROFILE]
    )
    baseline_production = deepcopy(mine_registry.DEFAULT_PRODUCTION_PROFILES["elt_default_w4"])
    baseline_mine = deepcopy(mine_registry.DEFAULT_REGISTRY["thalemine"])

    monkeypatch.setattr(
        mine_registry,
        "load_mine_parallel_preferences",
        lambda: {
            "defaults": {
                "mine": {
                    "default_workers": 11,
                }
            },
            "benchmark_profiles": {
                mine_registry.DEFAULT_BENCHMARK_SMALL_PROFILE: {
                    "workers": [2, 4],
                }
            },
            "production_profiles": {
                "elt_default_w4": {
                    "workers": 11,
                }
            },
            "mines": {
                "legumemine": {
                    "default_workers": 5,
                }
            },
        },
    )
    monkeypatch.setattr(mine_registry, "resolve_mine_parallel_preferences_path", lambda: Path("/tmp/test.toml"))

    loaded = mine_registry._load_registry()
    assert loaded["benchmark_profiles"][mine_registry.DEFAULT_BENCHMARK_SMALL_PROFILE]["workers"] == [2, 4]
    assert loaded["production_profiles"]["elt_default_w4"]["workers"] == 11
    assert loaded["mines"]["legumemine"]["default_workers"] == 5
    assert loaded["mines"]["thalemine"]["default_workers"] == 11
    assert loaded["_config_source"] == "mine_parallel_preferences_toml"

    assert mine_registry.DEFAULT_BENCHMARK_PROFILES[mine_registry.DEFAULT_BENCHMARK_SMALL_PROFILE] == baseline_benchmark
    assert mine_registry.DEFAULT_PRODUCTION_PROFILES["elt_default_w4"] == baseline_production
    assert mine_registry.DEFAULT_REGISTRY["thalemine"] == baseline_mine


def test_resolve_preferred_workers_uses_single_match_pass(monkeypatch):
    _reset_registry_state(monkeypatch)
    original_match = mine_registry._match_mine_profile
    calls = []

    def _counted_match(service_root, *, mines=None):
        calls.append({"service_root": service_root, "passed_mines": mines is not None})
        return original_match(service_root, mines=mines)

    monkeypatch.setattr(mine_registry, "_match_mine_profile", _counted_match)

    workers = mine_registry.resolve_preferred_workers(
        "https://mines.legumeinfo.org/legumemine/service",
        10_000,
        99,
    )
    assert workers == 4
    assert len(calls) == 1
    assert calls[0]["passed_mines"] is True


def test_plan_resolvers_load_registry_once_per_call(monkeypatch):
    _reset_registry_state(monkeypatch)
    original_load = mine_registry._load_registry
    load_calls = []

    def _counted_load():
        load_calls.append("load")
        return original_load()

    monkeypatch.setattr(mine_registry, "_load_registry", _counted_load)

    _ = mine_registry.resolve_production_plan(
        "https://mines.legumeinfo.org/legumemine/service",
        10_000,
        workflow="elt",
    )
    assert len(load_calls) == 1

    _ = mine_registry.resolve_benchmark_plan(
        "https://mines.legumeinfo.org/legumemine/service",
        10_000,
    )
    assert len(load_calls) == 2

    _ = mine_registry.resolve_preferred_workers(
        "https://mines.legumeinfo.org/legumemine/service",
        10_000,
        7,
    )
    assert len(load_calls) == 3
