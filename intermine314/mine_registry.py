from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

from intermine314.constants import (
    DEFAULT_PARALLEL_WORKERS,
    DEFAULT_PRODUCTION_PROFILE_SWITCH_ROWS,
    DEFAULT_WORKERS_TIER,
    FULL_WORKERS_TIER,
    PRODUCTION_PROFILE_ELT_DEFAULT,
    PRODUCTION_PROFILE_ELT_FULL,
    PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
    PRODUCTION_PROFILE_ETL_DEFAULT,
    PRODUCTION_PROFILE_ETL_FULL,
    PRODUCTION_PROFILE_ETL_SERVER_LIMITED,
    PRODUCTION_WORKFLOW_ELT,
    PRODUCTION_WORKFLOW_ETL,
    PRODUCTION_WORKFLOWS,
    SERVER_LIMITED_WORKERS_TIER,
)

try:
    import tomllib
except Exception:  # pragma: no cover - Python 3.14 includes tomllib
    tomllib = None


DEFAULT_BENCHMARK_SMALL_PROFILE = "benchmark_profile_3"
DEFAULT_BENCHMARK_LARGE_PROFILE = "benchmark_profile_1"
DEFAULT_BENCHMARK_FALLBACK_PROFILE = DEFAULT_BENCHMARK_SMALL_PROFILE
DEFAULT_PROFILE_SWITCH_ROWS = DEFAULT_PRODUCTION_PROFILE_SWITCH_ROWS

DEFAULT_BENCHMARK_PROFILES = {
    DEFAULT_BENCHMARK_LARGE_PROFILE: {
        "include_legacy_baseline": False,
        "workers": [4, 8, 12, 16],
    },
    "benchmark_profile_2": {
        "include_legacy_baseline": False,
        "workers": [4, 6, 8],
    },
    "benchmark_profile_3": {
        "include_legacy_baseline": True,
        "workers": [4, 8, 12, 16],
    },
    "benchmark_profile_4": {
        "include_legacy_baseline": True,
        "workers": [4, 6, 8],
    },
}

DEFAULT_PRODUCTION_PROFILES = {
    PRODUCTION_PROFILE_ELT_DEFAULT: {
        "workflow": PRODUCTION_WORKFLOW_ELT,
        "workers": DEFAULT_WORKERS_TIER,
        "pipeline": "parquet_duckdb",
        "parallel_profile": "large_query",
        "ordered": "unordered",
        "large_query_mode": True,
    },
    PRODUCTION_PROFILE_ELT_SERVER_LIMITED: {
        "workflow": PRODUCTION_WORKFLOW_ELT,
        "workers": SERVER_LIMITED_WORKERS_TIER,
        "pipeline": "parquet_duckdb",
        "parallel_profile": "large_query",
        "ordered": "unordered",
        "large_query_mode": True,
    },
    PRODUCTION_PROFILE_ELT_FULL: {
        "workflow": PRODUCTION_WORKFLOW_ELT,
        "workers": FULL_WORKERS_TIER,
        "pipeline": "parquet_duckdb",
        "parallel_profile": "large_query",
        "ordered": "unordered",
        "large_query_mode": True,
    },
    PRODUCTION_PROFILE_ETL_DEFAULT: {
        "workflow": PRODUCTION_WORKFLOW_ETL,
        "workers": DEFAULT_WORKERS_TIER,
        "pipeline": "polars_duckdb",
        "parallel_profile": "large_query",
        "ordered": "unordered",
        "large_query_mode": True,
    },
    PRODUCTION_PROFILE_ETL_SERVER_LIMITED: {
        "workflow": PRODUCTION_WORKFLOW_ETL,
        "workers": SERVER_LIMITED_WORKERS_TIER,
        "pipeline": "polars_duckdb",
        "parallel_profile": "large_query",
        "ordered": "unordered",
        "large_query_mode": True,
    },
    PRODUCTION_PROFILE_ETL_FULL: {
        "workflow": PRODUCTION_WORKFLOW_ETL,
        "workers": FULL_WORKERS_TIER,
        "pipeline": "polars_duckdb",
        "parallel_profile": "large_query",
        "ordered": "unordered",
        "large_query_mode": True,
    },
}

DEFAULT_PRODUCTION_PROFILE_BY_WORKFLOW = {
    PRODUCTION_WORKFLOW_ELT: PRODUCTION_PROFILE_ELT_DEFAULT,
    PRODUCTION_WORKFLOW_ETL: PRODUCTION_PROFILE_ETL_DEFAULT,
}

FULL_PRODUCTION_PROFILE_BY_WORKFLOW = {
    PRODUCTION_WORKFLOW_ELT: PRODUCTION_PROFILE_ELT_FULL,
    PRODUCTION_WORKFLOW_ETL: PRODUCTION_PROFILE_ETL_FULL,
}

SERVER_LIMITED_PROFILE_BY_WORKFLOW = {
    PRODUCTION_WORKFLOW_ELT: PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
    PRODUCTION_WORKFLOW_ETL: PRODUCTION_PROFILE_ETL_SERVER_LIMITED,
}


def _to_int_list(values):
    result = []
    for value in values:
        try:
            ivalue = int(value)
        except Exception:
            continue
        if ivalue <= 0:
            continue
        result.append(ivalue)
    return result


def _normalize_workflow(workflow):
    value = str(workflow or PRODUCTION_WORKFLOW_ELT).strip().lower()
    if value not in PRODUCTION_WORKFLOWS:
        choices = ", ".join(PRODUCTION_WORKFLOWS)
        raise ValueError(f"workflow must be one of: {choices}")
    return value


def _workers_to_production_profile(workflow, workers):
    workflow = _normalize_workflow(workflow)
    if int(workers) <= DEFAULT_WORKERS_TIER:
        return DEFAULT_PRODUCTION_PROFILE_BY_WORKFLOW[workflow]
    if int(workers) <= SERVER_LIMITED_WORKERS_TIER:
        return SERVER_LIMITED_PROFILE_BY_WORKFLOW[workflow]
    return FULL_PRODUCTION_PROFILE_BY_WORKFLOW[workflow]


def _standard_mine_profile(
    *,
    display_name,
    host_patterns,
    path_prefixes,
    default_workers=FULL_WORKERS_TIER,
    production_large_workers=FULL_WORKERS_TIER,
    production_switch_rows=DEFAULT_PROFILE_SWITCH_ROWS,
    benchmark_small_profile=DEFAULT_BENCHMARK_SMALL_PROFILE,
    benchmark_large_profile=DEFAULT_BENCHMARK_LARGE_PROFILE,
    benchmark_switch_rows=DEFAULT_PROFILE_SWITCH_ROWS,
    production_elt_small_profile=None,
    production_elt_large_profile=None,
    production_etl_small_profile=None,
    production_etl_large_profile=None,
):
    default_workers = int(default_workers)
    production_large_workers = int(production_large_workers)
    if production_elt_small_profile is None:
        production_elt_small_profile = _workers_to_production_profile(PRODUCTION_WORKFLOW_ELT, default_workers)
    if production_elt_large_profile is None:
        production_elt_large_profile = _workers_to_production_profile(PRODUCTION_WORKFLOW_ELT, production_large_workers)
    if production_etl_small_profile is None:
        production_etl_small_profile = _workers_to_production_profile(PRODUCTION_WORKFLOW_ETL, default_workers)
    if production_etl_large_profile is None:
        production_etl_large_profile = _workers_to_production_profile(PRODUCTION_WORKFLOW_ETL, production_large_workers)

    return {
        "display_name": display_name,
        "host_patterns": host_patterns,
        "path_prefixes": path_prefixes,
        # Legacy worker-resolution fields still used by existing query defaults.
        "default_workers": default_workers,
        "large_query_threshold_rows": int(production_switch_rows),
        "workers_above_threshold": production_large_workers,
        "workers_when_size_unknown": production_large_workers,
        # Production profile mapping (ELT + ETL).
        "production_profile_switch_rows": int(production_switch_rows),
        "production_elt_small_profile": str(production_elt_small_profile),
        "production_elt_large_profile": str(production_elt_large_profile),
        "production_etl_small_profile": str(production_etl_small_profile),
        "production_etl_large_profile": str(production_etl_large_profile),
        # Benchmark profile mapping.
        "benchmark_profile": benchmark_small_profile,
        "benchmark_small_profile": benchmark_small_profile,
        "benchmark_switch_threshold_rows": int(benchmark_switch_rows),
        "benchmark_large_profile": benchmark_large_profile,
    }


DEFAULT_REGISTRY = {
    "legumemine": {
        "display_name": "LegumeMine",
        "host_patterns": ["mines.legumeinfo.org"],
        "path_prefixes": ["/legumemine"],
        "default_workers": DEFAULT_WORKERS_TIER,
        "large_query_threshold_rows": DEFAULT_PROFILE_SWITCH_ROWS,
        "workers_above_threshold": DEFAULT_WORKERS_TIER,
        "workers_when_size_unknown": DEFAULT_WORKERS_TIER,
        "production_profile_switch_rows": DEFAULT_PROFILE_SWITCH_ROWS,
        "production_elt_small_profile": PRODUCTION_PROFILE_ELT_DEFAULT,
        "production_elt_large_profile": PRODUCTION_PROFILE_ELT_DEFAULT,
        "production_etl_small_profile": PRODUCTION_PROFILE_ETL_DEFAULT,
        "production_etl_large_profile": PRODUCTION_PROFILE_ETL_DEFAULT,
        "benchmark_profile": DEFAULT_BENCHMARK_SMALL_PROFILE,
        "benchmark_small_profile": DEFAULT_BENCHMARK_SMALL_PROFILE,
        "benchmark_switch_threshold_rows": DEFAULT_PROFILE_SWITCH_ROWS,
        "benchmark_small_workers": [],
        "benchmark_small_include_legacy_baseline": False,
        "benchmark_large_profile": "benchmark_profile_2",
    },
    "maizemine": _standard_mine_profile(
        display_name="MaizeMine",
        host_patterns=["maizemine.rnet.missouri.edu"],
        path_prefixes=["/maizemine"],
        default_workers=SERVER_LIMITED_WORKERS_TIER,
        production_large_workers=SERVER_LIMITED_WORKERS_TIER,
        benchmark_small_profile="benchmark_profile_4",
        benchmark_large_profile="benchmark_profile_2",
        production_elt_small_profile=PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
        production_elt_large_profile=PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
        production_etl_small_profile=PRODUCTION_PROFILE_ETL_SERVER_LIMITED,
        production_etl_large_profile=PRODUCTION_PROFILE_ETL_SERVER_LIMITED,
    ),
    "thalemine": _standard_mine_profile(
        display_name="ThaleMine",
        host_patterns=["bar.utoronto.ca"],
        path_prefixes=["/thalemine"],
    ),
    "oakmine": _standard_mine_profile(
        display_name="OakMine",
        host_patterns=["urgi.versailles.inra.fr", "urgi.versailles.inrae.fr"],
        path_prefixes=["/OakMine_PM1N"],
    ),
    "wheatmine": _standard_mine_profile(
        display_name="WheatMine",
        host_patterns=["urgi.versailles.inra.fr", "urgi.versailles.inrae.fr"],
        path_prefixes=["/WheatMine"],
    ),
}

_CACHE = None


def _config_path():
    return Path(__file__).resolve().parent.parent / "config" / "mine-parallel-preferences.toml"


def _default_mines_copy():
    return deepcopy(DEFAULT_REGISTRY)


def _default_benchmark_profiles_copy():
    return deepcopy(DEFAULT_BENCHMARK_PROFILES)


def _default_production_profiles_copy():
    return deepcopy(DEFAULT_PRODUCTION_PROFILES)


def _merge_mines(loaded):
    merged = _default_mines_copy()

    defaults_block = loaded.get("defaults", {}) if isinstance(loaded, dict) else {}
    mine_defaults = defaults_block.get("mine", {}) if isinstance(defaults_block, dict) else {}
    if not isinstance(mine_defaults, dict):
        mine_defaults = {}

    raw_mines = loaded.get("mines", {}) if isinstance(loaded, dict) else {}
    if not isinstance(raw_mines, dict):
        raw_mines = {}

    for name, profile in raw_mines.items():
        if not isinstance(profile, dict):
            continue
        base = dict(merged.get(name, {}))
        if mine_defaults:
            base.update(mine_defaults)
        base.update(profile)
        merged[name] = base

    if mine_defaults:
        for name, profile in merged.items():
            if name not in raw_mines:
                base = dict(profile)
                base.update(mine_defaults)
                merged[name] = base
    return merged


def _merge_benchmark_profiles(loaded):
    merged = _default_benchmark_profiles_copy()
    raw_profiles = loaded.get("benchmark_profiles", {}) if isinstance(loaded, dict) else {}
    if not isinstance(raw_profiles, dict):
        return merged

    for name, profile in raw_profiles.items():
        if not isinstance(profile, dict):
            continue
        base = dict(merged.get(name, {}))
        base.update(profile)
        merged[name] = base
    return merged


def _merge_production_profiles(loaded):
    merged = _default_production_profiles_copy()
    raw_profiles = loaded.get("production_profiles", {}) if isinstance(loaded, dict) else {}
    if not isinstance(raw_profiles, dict):
        return merged

    for name, profile in raw_profiles.items():
        if not isinstance(profile, dict):
            continue
        base = dict(merged.get(name, {}))
        base.update(profile)
        merged[name] = base
    return merged


def _load_registry():
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    data = {
        "mines": _default_mines_copy(),
        "benchmark_profiles": _default_benchmark_profiles_copy(),
        "production_profiles": _default_production_profiles_copy(),
    }
    cfg = _config_path()
    if cfg.exists() and tomllib is not None:
        try:
            loaded = tomllib.loads(cfg.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data["mines"] = _merge_mines(loaded)
                data["benchmark_profiles"] = _merge_benchmark_profiles(loaded)
                data["production_profiles"] = _merge_production_profiles(loaded)
        except Exception:
            pass
    _CACHE = data
    return _CACHE


def _normalize_service_root(service_root):
    if not service_root:
        return "", ""
    parsed = urlparse(service_root)
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/service"):
        path = path[: -len("/service")]
    return host, path


def _matches_profile(profile, host, path):
    host_patterns = [str(h).lower() for h in profile.get("host_patterns", [])]
    if host_patterns and host not in host_patterns:
        return False
    prefixes = profile.get("path_prefixes", [])
    if not prefixes:
        return True
    path_lower = path.lower()
    return any(path_lower.startswith(str(prefix).lower()) for prefix in prefixes)


def _match_mine_profile(service_root):
    host, path = _normalize_service_root(service_root)
    if not host:
        return None

    mines = _load_registry().get("mines", {})
    for _, profile in mines.items():
        if _matches_profile(profile, host, path):
            return profile
    return None


def _normalize_benchmark_profile(name, profiles, fallback_name):
    if not profiles:
        return fallback_name
    if name in profiles:
        return name
    if fallback_name in profiles:
        return fallback_name
    return next(iter(profiles))


def _profile_to_plan(profile_name, profile_data):
    workers = _to_int_list(profile_data.get("workers", []))
    if not workers:
        workers = _to_int_list(DEFAULT_BENCHMARK_PROFILES[DEFAULT_BENCHMARK_FALLBACK_PROFILE]["workers"])
    include_legacy = bool(profile_data.get("include_legacy_baseline", False))
    return {
        "name": profile_name,
        "workers": workers,
        "include_legacy_baseline": include_legacy,
    }


def _normalize_production_profile(name, profiles, workflow):
    workflow = _normalize_workflow(workflow)
    fallback = DEFAULT_PRODUCTION_PROFILE_BY_WORKFLOW[workflow]
    if not profiles:
        return fallback
    if name in profiles:
        return name
    if fallback in profiles:
        return fallback
    return next(iter(profiles))


def _production_profile_to_plan(profile_name, profile_data, workflow):
    workflow = _normalize_workflow(workflow)
    default_workers = DEFAULT_WORKERS_TIER
    if profile_name in DEFAULT_PRODUCTION_PROFILES:
        default_workers = int(DEFAULT_PRODUCTION_PROFILES[profile_name].get("workers", DEFAULT_WORKERS_TIER))

    workers = profile_data.get("workers", default_workers)
    try:
        workers = int(workers)
    except Exception:
        workers = default_workers
    if workers <= 0:
        workers = default_workers

    plan_workflow = _normalize_workflow(profile_data.get("workflow", workflow))
    if plan_workflow != workflow:
        plan_workflow = workflow
    if plan_workflow == PRODUCTION_WORKFLOW_ELT:
        default_pipeline = "parquet_duckdb"
    else:
        default_pipeline = "polars_duckdb"

    return {
        "name": profile_name,
        "workflow": plan_workflow,
        "workers": workers,
        "pipeline": str(profile_data.get("pipeline", default_pipeline)),
        "parallel_profile": str(profile_data.get("parallel_profile", "large_query")),
        "ordered": profile_data.get("ordered", "unordered"),
        "large_query_mode": bool(profile_data.get("large_query_mode", True)),
        "prefetch": profile_data.get("prefetch"),
        "inflight_limit": profile_data.get("inflight_limit"),
    }


def _mine_profile_name_for_workflow(mine_profile, *, size, workflow, fallback_profile):
    workflow = _normalize_workflow(workflow)
    switch = mine_profile.get("production_profile_switch_rows", mine_profile.get("large_query_threshold_rows"))
    try:
        threshold = int(switch) if switch is not None else DEFAULT_PROFILE_SWITCH_ROWS
    except Exception:
        threshold = DEFAULT_PROFILE_SWITCH_ROWS

    small_key = f"production_{workflow}_small_profile"
    large_key = f"production_{workflow}_large_profile"
    small_profile = mine_profile.get(small_key)
    large_profile = mine_profile.get(large_key)

    if not small_profile:
        small_profile = _workers_to_production_profile(workflow, int(mine_profile.get("default_workers", DEFAULT_PARALLEL_WORKERS)))
    if not large_profile:
        large_workers = int(mine_profile.get("workers_above_threshold", mine_profile.get("default_workers", DEFAULT_PARALLEL_WORKERS)))
        large_profile = _workers_to_production_profile(workflow, large_workers)

    if size is None:
        unknown_workers = int(
            mine_profile.get(
                "workers_when_size_unknown",
                mine_profile.get("workers_above_threshold", mine_profile.get("default_workers", DEFAULT_PARALLEL_WORKERS)),
            )
        )
        return _workers_to_production_profile(workflow, unknown_workers)

    if int(size) > threshold:
        return str(large_profile or fallback_profile)
    return str(small_profile or fallback_profile)


def resolve_production_plan(
    service_root,
    size,
    *,
    workflow=PRODUCTION_WORKFLOW_ELT,
    production_profile="auto",
):
    workflow = _normalize_workflow(workflow)
    registry = _load_registry()
    profiles = registry.get("production_profiles", {}) or DEFAULT_PRODUCTION_PROFILES

    if production_profile is not None and str(production_profile).strip().lower() != "auto":
        resolved_name = _normalize_production_profile(str(production_profile), profiles, workflow)
        return _production_profile_to_plan(resolved_name, profiles[resolved_name], workflow)

    fallback_name = DEFAULT_PRODUCTION_PROFILE_BY_WORKFLOW[workflow]
    mine_profile = _match_mine_profile(service_root)
    if mine_profile is None:
        fallback_resolved = _normalize_production_profile(fallback_name, profiles, workflow)
        return _production_profile_to_plan(fallback_resolved, profiles[fallback_resolved], workflow)

    profile_name = _mine_profile_name_for_workflow(
        mine_profile,
        size=size,
        workflow=workflow,
        fallback_profile=fallback_name,
    )
    resolved_name = _normalize_production_profile(profile_name, profiles, workflow)
    return _production_profile_to_plan(resolved_name, profiles[resolved_name], workflow)


def resolve_preferred_workers(service_root, size, fallback_workers):
    mine_profile = _match_mine_profile(service_root)
    if mine_profile is None:
        return fallback_workers
    plan = resolve_production_plan(service_root, size, workflow=PRODUCTION_WORKFLOW_ELT, production_profile="auto")
    try:
        return int(plan["workers"])
    except Exception:
        return fallback_workers


def resolve_benchmark_plan(service_root, size, fallback_profile=DEFAULT_BENCHMARK_FALLBACK_PROFILE):
    registry = _load_registry()
    profiles = registry.get("benchmark_profiles", {}) or DEFAULT_BENCHMARK_PROFILES
    fallback_name = _normalize_benchmark_profile(fallback_profile, profiles, DEFAULT_BENCHMARK_FALLBACK_PROFILE)
    mine_profile = _match_mine_profile(service_root)

    if mine_profile is None:
        return _profile_to_plan(fallback_name, profiles[fallback_name])

    threshold = mine_profile.get("benchmark_switch_threshold_rows")
    if (
        threshold is not None
        and size is not None
        and size <= int(threshold)
        and isinstance(mine_profile.get("benchmark_small_workers"), list)
    ):
        workers = _to_int_list(mine_profile.get("benchmark_small_workers", []))
        if workers:
            return {
                "name": "benchmark_small_workers_override",
                "workers": workers,
                "include_legacy_baseline": bool(mine_profile.get("benchmark_small_include_legacy_baseline", False)),
            }

    profile_name = mine_profile.get("benchmark_small_profile", mine_profile.get("benchmark_profile", fallback_name))
    if threshold is not None and size is not None and size > int(threshold):
        profile_name = mine_profile.get("benchmark_large_profile", profile_name)
    profile_name = _normalize_benchmark_profile(str(profile_name), profiles, fallback_name)
    return _profile_to_plan(profile_name, profiles[profile_name])


def resolve_named_benchmark_profile(profile_name, fallback_profile=DEFAULT_BENCHMARK_FALLBACK_PROFILE):
    registry = _load_registry()
    profiles = registry.get("benchmark_profiles", {}) or DEFAULT_BENCHMARK_PROFILES
    fallback_name = _normalize_benchmark_profile(fallback_profile, profiles, DEFAULT_BENCHMARK_FALLBACK_PROFILE)
    resolved_name = _normalize_benchmark_profile(str(profile_name), profiles, fallback_name)
    return _profile_to_plan(resolved_name, profiles[resolved_name])


def resolve_benchmark_phase_plan(
    *,
    service_root,
    rows_target,
    explicit_workers,
    benchmark_profile,
    include_legacy_baseline,
):
    workers = _to_int_list(explicit_workers or [])
    if workers:
        return {
            "name": "workers_override",
            "workers": workers,
            "include_legacy_baseline": bool(include_legacy_baseline),
        }

    if benchmark_profile is not None and str(benchmark_profile).strip().lower() != "auto":
        profile_plan = resolve_named_benchmark_profile(str(benchmark_profile), DEFAULT_BENCHMARK_FALLBACK_PROFILE)
    else:
        profile_plan = resolve_benchmark_plan(service_root, rows_target, DEFAULT_BENCHMARK_FALLBACK_PROFILE)

    return {
        "name": profile_plan["name"],
        "workers": list(profile_plan["workers"]),
        "include_legacy_baseline": bool(include_legacy_baseline) and bool(profile_plan["include_legacy_baseline"]),
    }
