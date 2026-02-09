from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

from intermine314.constants import DEFAULT_PARALLEL_WORKERS

try:
    import tomllib
except Exception:  # pragma: no cover - Python 3.14 includes tomllib
    tomllib = None


DEFAULT_BENCHMARK_SMALL_PROFILE = "benchmark_profile_3"
DEFAULT_BENCHMARK_LARGE_PROFILE = "benchmark_profile_1"
DEFAULT_BENCHMARK_FALLBACK_PROFILE = DEFAULT_BENCHMARK_SMALL_PROFILE
DEFAULT_PROFILE_SWITCH_ROWS = 50000
DEFAULT_PRODUCTION_LARGE_WORKERS = 12

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

def _standard_mine_profile(
    *,
    display_name,
    host_patterns,
    path_prefixes,
    default_workers=DEFAULT_PARALLEL_WORKERS,
    production_large_workers=DEFAULT_PRODUCTION_LARGE_WORKERS,
    production_switch_rows=DEFAULT_PROFILE_SWITCH_ROWS,
    benchmark_small_profile=DEFAULT_BENCHMARK_SMALL_PROFILE,
    benchmark_large_profile=DEFAULT_BENCHMARK_LARGE_PROFILE,
    benchmark_switch_rows=DEFAULT_PROFILE_SWITCH_ROWS,
):
    return {
        "display_name": display_name,
        "host_patterns": host_patterns,
        "path_prefixes": path_prefixes,
        "default_workers": int(default_workers),
        "large_query_threshold_rows": int(production_switch_rows),
        "workers_above_threshold": int(production_large_workers),
        "workers_when_size_unknown": int(production_large_workers),
        # Backward-compatible alias for legacy configs.
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
        "default_workers": 4,
        "large_query_threshold_rows": 50000,
        "workers_above_threshold": 4,
        "workers_when_size_unknown": 4,
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
        default_workers=8,
        production_large_workers=8,
        benchmark_small_profile="benchmark_profile_4",
        benchmark_large_profile="benchmark_profile_2",
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


def _default_mines_copy():
    return deepcopy(DEFAULT_REGISTRY)


def _default_benchmark_profiles_copy():
    return deepcopy(DEFAULT_BENCHMARK_PROFILES)


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

    # Also apply mine defaults to built-in mines that are not explicitly listed in config.
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


def _load_registry():
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    data = {"mines": _default_mines_copy(), "benchmark_profiles": _default_benchmark_profiles_copy()}
    cfg = _config_path()
    if cfg.exists() and tomllib is not None:
        try:
            loaded = tomllib.loads(cfg.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data["mines"] = _merge_mines(loaded)
                data["benchmark_profiles"] = _merge_benchmark_profiles(loaded)
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


def _choose_workers(profile, size, fallback):
    default_workers = int(profile.get("default_workers", fallback))
    threshold = profile.get("large_query_threshold_rows")
    if threshold is None:
        return default_workers

    threshold = int(threshold)
    if size is None:
        return int(profile.get("workers_when_size_unknown", profile.get("workers_above_threshold", default_workers)))
    if size > threshold:
        return int(profile.get("workers_above_threshold", default_workers))
    return default_workers


def resolve_preferred_workers(service_root, size, fallback_workers):
    profile = _match_mine_profile(service_root)
    if profile is None:
        return fallback_workers
    return _choose_workers(profile, size, fallback_workers)


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
