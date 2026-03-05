from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.parallel.policy import (
    VALID_ORDER_MODES,
    VALID_PARALLEL_PROFILES,
)

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_PROFILE_DIR = _REPO_ROOT / "benchmarks" / "profiles"
_MINE_PARALLEL_PREFERENCES_FILE = "mine-parallel-preferences.toml"
_PARALLEL_PROFILES_FILE = "parallel-profiles.toml"
_MAX_CONFIG_FILE_BYTES = 1_048_576

_WORKFLOW_ELT = "elt"
_PIPELINE_ELT = "parquet_duckdb"
_DEFAULT_WORKERS_TIER = 4
_SERVER_LIMITED_WORKERS_TIER = 8
_FULL_WORKERS_TIER = 16
_PRODUCTION_PROFILE_ELT_DEFAULT = "elt_default_w4"
_PRODUCTION_PROFILE_ELT_SERVER_LIMITED = "elt_server_limited_w8"
_PRODUCTION_PROFILE_ELT_FULL = "elt_full_w16"
_DEFAULT_PARALLEL_PROFILE = "large_query"
_DEFAULT_ORDERED_MODE = "unordered"
_DEFAULT_RESOURCE_PROFILE = "default"

_THALEMINE_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class ConfigDocument:
    payload: Mapping[str, Any]
    ok: bool
    source: str
    path: str | None
    error_kind: str | None


@dataclass(frozen=True)
class ResolvedParallelPolicy:
    workflow: str
    pipeline: str
    production_profile: str
    workers: int
    profile: str
    ordered: str
    large_query_mode: bool
    resource_profile: str


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _to_mutable_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _parse_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    if parsed <= 0:
        return int(default)
    return int(parsed)


def _normalize_mode(value: Any, *, default: str) -> str:
    if isinstance(value, bool):
        return "ordered" if value else "unordered"
    text = str(value or "").strip().lower()
    if not text:
        return str(default)
    if text not in VALID_ORDER_MODES:
        return str(default)
    return text


def _path_cache_key(path: Path) -> tuple[str, int, int]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=32)
def _load_toml_cached(path_str: str, mtime_ns: int, size_bytes: int) -> dict[str, Any]:
    _ = (mtime_ns, size_bytes)
    path = Path(path_str)
    with path.open("rb") as handle:
        return tomllib.load(handle)  # type: ignore[union-attr]


def _load_document(path: Path, *, source: str) -> ConfigDocument:
    path_str = str(path)
    if tomllib is None:
        return ConfigDocument(payload={}, ok=False, source=source, path=path_str, error_kind="tomllib_unavailable")
    try:
        cache_key = _path_cache_key(path)
    except FileNotFoundError:
        return ConfigDocument(payload={}, ok=False, source=source, path=path_str, error_kind="missing")
    except Exception:
        return ConfigDocument(payload={}, ok=False, source=source, path=path_str, error_kind="unreadable")
    if cache_key[2] > _MAX_CONFIG_FILE_BYTES:
        return ConfigDocument(payload={}, ok=False, source=source, path=cache_key[0], error_kind="oversized")
    try:
        loaded = _load_toml_cached(*cache_key)
    except Exception:
        return ConfigDocument(payload={}, ok=False, source=source, path=cache_key[0], error_kind="invalid_toml")
    if not isinstance(loaded, dict):
        return ConfigDocument(payload={}, ok=False, source=source, path=cache_key[0], error_kind="invalid_shape")
    return ConfigDocument(payload=loaded, ok=True, source=source, path=cache_key[0], error_kind=None)


def _default_document(filename: str) -> ConfigDocument:
    return _load_document(_BENCHMARK_PROFILE_DIR / filename, source="benchmarks_toml")


def _override_document(env_var: str) -> ConfigDocument | None:
    override = os.getenv(env_var, "").strip()
    if not override:
        return None
    return _load_document(Path(override), source="override_toml")


def _effective_document(default_doc: ConfigDocument, override_doc: ConfigDocument | None) -> ConfigDocument:
    if override_doc is not None and bool(override_doc.ok):
        return override_doc
    return default_doc


def load_mine_parallel_preferences_document() -> ConfigDocument:
    default_doc = _default_document(_MINE_PARALLEL_PREFERENCES_FILE)
    override_doc = _override_document("INTERMINE314_BENCH_MINE_PARALLEL_PREFERENCES_PATH")
    return _effective_document(default_doc, override_doc)


def load_parallel_profiles_document() -> ConfigDocument:
    default_doc = _default_document(_PARALLEL_PROFILES_FILE)
    override_doc = _override_document("INTERMINE314_BENCH_PARALLEL_PROFILES_PATH")
    return _effective_document(default_doc, override_doc)


def _default_production_profiles(
    *,
    workers_default: int,
    workers_server_limited: int,
    workers_full: int,
) -> dict[str, dict[str, Any]]:
    return {
        _PRODUCTION_PROFILE_ELT_DEFAULT: {
            "workers": int(workers_default),
            "parallel_profile": _DEFAULT_PARALLEL_PROFILE,
            "ordered": _DEFAULT_ORDERED_MODE,
            "large_query_mode": True,
        },
        _PRODUCTION_PROFILE_ELT_SERVER_LIMITED: {
            "workers": int(workers_server_limited),
            "parallel_profile": _DEFAULT_PARALLEL_PROFILE,
            "ordered": _DEFAULT_ORDERED_MODE,
            "large_query_mode": True,
        },
        _PRODUCTION_PROFILE_ELT_FULL: {
            "workers": int(workers_full),
            "parallel_profile": _DEFAULT_PARALLEL_PROFILE,
            "ordered": _DEFAULT_ORDERED_MODE,
            "large_query_mode": True,
        },
    }


def _default_mines() -> dict[str, dict[str, Any]]:
    return {
        "legumemine": {
            "display_name": "LegumeMine",
            "host_patterns": ["mines.legumeinfo.org"],
            "path_prefixes": ["/legumemine"],
            "production_profile": _PRODUCTION_PROFILE_ELT_DEFAULT,
            "resource_profile": _DEFAULT_RESOURCE_PROFILE,
        },
        "maizemine": {
            "display_name": "MaizeMine",
            "host_patterns": ["maizemine.rnet.missouri.edu"],
            "path_prefixes": ["/maizemine"],
            "production_profile": _PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
            "resource_profile": _DEFAULT_RESOURCE_PROFILE,
        },
        "thalemine": {
            "display_name": "ThaleMine",
            "host_patterns": ["bar.utoronto.ca"],
            "path_prefixes": ["/thalemine"],
            "production_profile": _PRODUCTION_PROFILE_ELT_FULL,
            "resource_profile": _DEFAULT_RESOURCE_PROFILE,
            "user_agent": _THALEMINE_BROWSER_USER_AGENT,
        },
        "oakmine": {
            "display_name": "OakMine",
            "host_patterns": ["urgi.versailles.inra.fr", "urgi.versailles.inrae.fr"],
            "path_prefixes": ["/OakMine_PM1N"],
            "production_profile": _PRODUCTION_PROFILE_ELT_FULL,
            "resource_profile": _DEFAULT_RESOURCE_PROFILE,
        },
        "wheatmine": {
            "display_name": "WheatMine",
            "host_patterns": ["urgi.versailles.inra.fr", "urgi.versailles.inrae.fr"],
            "path_prefixes": ["/WheatMine"],
            "production_profile": _PRODUCTION_PROFILE_ELT_FULL,
            "resource_profile": _DEFAULT_RESOURCE_PROFILE,
        },
    }


def _normalize_service_root(service_root: object) -> tuple[str, str]:
    if not service_root:
        return "", ""
    parsed = urlparse(str(service_root))
    host = (parsed.hostname or "").strip().lower()
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/service"):
        path = path[: -len("/service")]
    return host, path.lower()


def _mine_matches(profile: Mapping[str, Any], host: str, path: str) -> bool:
    host_patterns_raw = profile.get("host_patterns")
    host_patterns = tuple(str(item).strip().lower() for item in host_patterns_raw) if isinstance(host_patterns_raw, (list, tuple)) else ()
    if host_patterns and host not in host_patterns:
        return False
    prefixes_raw = profile.get("path_prefixes")
    prefixes = tuple(str(item).strip().lower() for item in prefixes_raw) if isinstance(prefixes_raw, (list, tuple)) else ()
    if not prefixes:
        return True
    return any(path.startswith(prefix) for prefix in prefixes)


def _match_mine(service_root: object, mines: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any] | None:
    host, path = _normalize_service_root(service_root)
    if not host:
        return None
    for mine in mines.values():
        if isinstance(mine, Mapping) and _mine_matches(mine, host, path):
            return mine
    return None


def _load_parallel_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], Mapping[str, Any]]:
    runtime = get_runtime_defaults()
    workers_default = _parse_positive_int(runtime.registry_defaults.default_workers_tier, _DEFAULT_WORKERS_TIER)
    workers_server_limited = _parse_positive_int(
        runtime.registry_defaults.server_limited_workers_tier,
        _SERVER_LIMITED_WORKERS_TIER,
    )
    workers_full = _parse_positive_int(runtime.registry_defaults.full_workers_tier, _FULL_WORKERS_TIER)

    production_profiles = _default_production_profiles(
        workers_default=workers_default,
        workers_server_limited=workers_server_limited,
        workers_full=workers_full,
    )
    mines = _default_mines()

    mine_doc = load_mine_parallel_preferences_document()
    mine_root = _to_mapping(mine_doc.payload)
    mine_defaults = _to_mapping(_to_mapping(mine_root.get("defaults")).get("mine"))

    for name, profile in _to_mapping(mine_root.get("production_profiles")).items():
        if not isinstance(profile, Mapping):
            continue
        key = str(name).strip().lower()
        base = production_profiles.get(key, {})
        production_profiles[key] = {**base, **dict(profile)}

    raw_mines = _to_mapping(mine_root.get("mines"))
    if mine_defaults:
        for key in list(mines.keys()):
            mines[key] = {**mine_defaults, **mines[key]}
    for name, mine_profile in raw_mines.items():
        if not isinstance(mine_profile, Mapping):
            continue
        key = str(name).strip().lower()
        base = _to_mapping(mines.get(key))
        mines[key] = {**base, **dict(mine_defaults), **dict(mine_profile)}
    return production_profiles, mines, mine_defaults


def resolve_parallel_policy(
    mine_url: object,
    size: object,
    user_overrides: Mapping[str, Any] | None = None,
) -> ResolvedParallelPolicy:
    if size is not None:
        try:
            int(size)
        except Exception as exc:
            raise ValueError(f"Invalid size: {size!r}") from exc

    overrides = _to_mapping(user_overrides)
    workflow = str(overrides.get("workflow", _WORKFLOW_ELT) or _WORKFLOW_ELT).strip().lower()
    if workflow != _WORKFLOW_ELT:
        raise ValueError("workflow must be 'elt'")

    runtime = get_runtime_defaults()
    workers_default = _parse_positive_int(runtime.registry_defaults.default_workers_tier, _DEFAULT_WORKERS_TIER)
    default_parallel_profile = str(runtime.query_defaults.default_parallel_profile or "default").strip().lower() or "default"
    if default_parallel_profile not in VALID_PARALLEL_PROFILES:
        default_parallel_profile = "default"
    default_ordered_mode = _normalize_mode(
        runtime.query_defaults.default_parallel_ordered_mode,
        default="ordered",
    )
    default_large_query_mode = bool(runtime.query_defaults.default_large_query_mode)

    production_profiles, mines, mine_defaults = _load_parallel_inputs()
    matched_mine = _match_mine(mine_url, mines)

    explicit_production_profile = str(overrides.get("production_profile", "")).strip().lower()
    if explicit_production_profile and explicit_production_profile != "auto":
        production_profile_name = explicit_production_profile
    elif isinstance(matched_mine, Mapping):
        production_profile_name = str(matched_mine.get("production_profile", "")).strip().lower()
    else:
        production_profile_name = _PRODUCTION_PROFILE_ELT_DEFAULT

    if production_profile_name not in production_profiles:
        if _PRODUCTION_PROFILE_ELT_DEFAULT in production_profiles:
            production_profile_name = _PRODUCTION_PROFILE_ELT_DEFAULT
        else:
            production_profile_name = next(iter(production_profiles))
    production_profile = _to_mapping(production_profiles.get(production_profile_name))

    explicit_profile = overrides.get("profile")
    if explicit_profile is None:
        profile_name = str(production_profile.get("parallel_profile", default_parallel_profile) or default_parallel_profile).strip().lower()
    else:
        profile_name = str(explicit_profile or "").strip().lower()
    if profile_name not in VALID_PARALLEL_PROFILES:
        if explicit_profile is not None:
            choices = ", ".join(sorted(VALID_PARALLEL_PROFILES))
            raise ValueError(f"profile must be one of: {choices}")
        profile_name = default_parallel_profile

    parallel_profile_doc = load_parallel_profiles_document()
    parallel_profiles_root = _to_mapping(parallel_profile_doc.payload)
    parallel_profile_defaults = _to_mapping(_to_mapping(parallel_profiles_root.get("profiles")).get(profile_name))

    workers = _parse_positive_int(
        overrides.get("max_workers", production_profile.get("workers")),
        workers_default,
    )

    ordered_value = overrides.get("ordered")
    if ordered_value is None:
        ordered_value = production_profile.get("ordered")
    if ordered_value is None:
        ordered_value = parallel_profile_defaults.get("ordered")
    ordered = _normalize_mode(ordered_value, default=default_ordered_mode)

    if "large_query_mode" in overrides:
        large_query_mode = bool(overrides.get("large_query_mode"))
    elif "large_query_mode" in production_profile:
        large_query_mode = bool(production_profile.get("large_query_mode"))
    elif "large_query_mode" in parallel_profile_defaults:
        large_query_mode = bool(parallel_profile_defaults.get("large_query_mode"))
    else:
        large_query_mode = bool(default_large_query_mode)

    mine_resource_profile = matched_mine.get("resource_profile") if isinstance(matched_mine, Mapping) else None
    if "resource_profile" in overrides:
        resource_profile = str(overrides.get("resource_profile") or _DEFAULT_RESOURCE_PROFILE).strip().lower() or _DEFAULT_RESOURCE_PROFILE
    elif mine_resource_profile is not None:
        resource_profile = str(mine_resource_profile or _DEFAULT_RESOURCE_PROFILE).strip().lower() or _DEFAULT_RESOURCE_PROFILE
    else:
        resource_profile = str(mine_defaults.get("resource_profile", _DEFAULT_RESOURCE_PROFILE) or _DEFAULT_RESOURCE_PROFILE).strip().lower() or _DEFAULT_RESOURCE_PROFILE

    return ResolvedParallelPolicy(
        workflow=_WORKFLOW_ELT,
        pipeline=_PIPELINE_ELT,
        production_profile=production_profile_name,
        workers=int(workers),
        profile=profile_name,
        ordered=ordered,
        large_query_mode=bool(large_query_mode),
        resource_profile=resource_profile,
    )


def load_mine_profiles() -> dict[str, dict[str, Any]]:
    _production_profiles, mines, _mine_defaults = _load_parallel_inputs()
    normalized: dict[str, dict[str, Any]] = {}
    for name, profile in mines.items():
        if not isinstance(profile, Mapping):
            continue
        normalized[str(name).strip().lower()] = _to_mutable_dict(profile)
    return normalized


__all__ = [
    "ConfigDocument",
    "ResolvedParallelPolicy",
    "load_mine_parallel_preferences_document",
    "load_parallel_profiles_document",
    "load_mine_profiles",
    "resolve_parallel_policy",
]
