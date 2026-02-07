from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

try:
    import tomllib
except Exception:  # pragma: no cover - Python 3.14 includes tomllib
    tomllib = None


DEFAULT_BENCHMARK_FALLBACK_PROFILE = "benchmark_profile_1"

DEFAULT_BENCHMARK_PROFILES = {
    "benchmark_profile_1": {
        "include_legacy_baseline": True,
        "workers": [2, 4, 6, 8, 10, 12, 14, 16, 18],
    },
    "benchmark_profile_2": {
        "include_legacy_baseline": False,
        "workers": [4, 8, 12, 16],
    },
    "benchmark_profile_3": {
        "include_legacy_baseline": False,
        "workers": [4, 6, 8],
    },
    "benchmark_profile_4": {
        "include_legacy_baseline": True,
        "workers": [4, 6, 8],
    },
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
        "benchmark_profile": "benchmark_profile_1",
        "benchmark_switch_threshold_rows": 50000,
        "benchmark_small_workers": [4],
        "benchmark_small_include_legacy_baseline": False,
        "benchmark_large_profile": "benchmark_profile_3",
    },
    "maizemine": {
        "display_name": "MaizeMine",
        "host_patterns": ["maizemine.rnet.missouri.edu"],
        "path_prefixes": ["/maizemine"],
        "default_workers": 16,
        "benchmark_profile": "benchmark_profile_1",
    },
    "thalemine": {
        "display_name": "ThaleMine",
        "host_patterns": ["bar.utoronto.ca"],
        "path_prefixes": ["/thalemine"],
        "default_workers": 16,
        "benchmark_profile": "benchmark_profile_1",
    },
    "oakmine": {
        "display_name": "OakMine",
        "host_patterns": ["urgi.versailles.inra.fr"],
        "path_prefixes": ["/OakMine_PM1N"],
        "default_workers": 16,
        "benchmark_profile": "benchmark_profile_1",
    },
    "wheatmine": {
        "display_name": "WheatMine",
        "host_patterns": ["urgi.versailles.inra.fr"],
        "path_prefixes": ["/WheatMine"],
        "default_workers": 16,
        "benchmark_profile": "benchmark_profile_1",
    },
}

_CACHE = None


def _config_path():
    return Path(__file__).resolve().parent.parent / "config" / "mine-parallel-preferences.toml"


def _to_int_list(values):
    result = []
    for value in values:
        ivalue = int(value)
        if ivalue <= 0:
            continue
        result.append(ivalue)
    return result


def _load_registry():
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    data = {"mines": DEFAULT_REGISTRY, "benchmark_profiles": DEFAULT_BENCHMARK_PROFILES}
    cfg = _config_path()
    if cfg.exists() and tomllib is not None:
        try:
            loaded = tomllib.loads(cfg.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                if isinstance(loaded.get("mines"), dict):
                    data["mines"] = loaded["mines"]
                if isinstance(loaded.get("benchmark_profiles"), dict):
                    data["benchmark_profiles"] = loaded["benchmark_profiles"]
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

    profile_name = mine_profile.get("benchmark_profile", fallback_name)
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
