from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from urllib.parse import urlparse

from intermine314.config.loader import (
    load_mine_parallel_preferences,
    load_packaged_mine_parallel_preferences_detailed,
    resolve_mine_parallel_preferences_path,
)
from intermine314.config.runtime_defaults import QueryDefaults, RegistryDefaults
from intermine314.parallel.policy import (
    VALID_ORDER_MODES,
    VALID_PARALLEL_PROFILES,
    normalize_order_mode,
)
from intermine314.util.logging import log_structured_event

_QUERY_DEFAULTS = QueryDefaults()
_REGISTRY_DEFAULTS = RegistryDefaults()
_DEFAULT_PARALLEL_WORKERS = int(_QUERY_DEFAULTS.default_parallel_workers)
_DEFAULT_PROFILE_SWITCH_ROWS = int(_REGISTRY_DEFAULTS.default_production_profile_switch_rows)
_DEFAULT_WORKERS_TIER = int(_REGISTRY_DEFAULTS.default_workers_tier)
_SERVER_LIMITED_WORKERS_TIER = int(_REGISTRY_DEFAULTS.server_limited_workers_tier)
_FULL_WORKERS_TIER = int(_REGISTRY_DEFAULTS.full_workers_tier)

_WORKFLOW_ELT = "elt"
_PIPELINE_PARQUET_DUCKDB = "parquet_duckdb"
_RESOURCE_PROFILE_DEFAULT = "default"
_PRODUCTION_PARALLEL_PROFILE_DEFAULT = "large_query"
_PRODUCTION_ORDERED_DEFAULT = "unordered"

PRODUCTION_PROFILE_ELT_DEFAULT = "elt_default_w4"
PRODUCTION_PROFILE_ELT_SERVER_LIMITED = "elt_server_limited_w8"
PRODUCTION_PROFILE_ELT_FULL = "elt_full_w16"

THALEMINE_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

_REGISTRY_MINES_LOG = logging.getLogger("intermine314.registry.mines")
_CACHE: dict[str, object] | None = None
_INVALID_CONFIG_ATTEMPTS = 0
_INVALID_CONFIG_ATTEMPTS_BY_FIELD: dict[str, int] = {}


def _log_event(event: str, *, level: int = logging.DEBUG, **fields: object) -> None:
    if not _REGISTRY_MINES_LOG.isEnabledFor(level):
        return
    log_structured_event(_REGISTRY_MINES_LOG, level, event, **fields)


def _record_invalid_config(kind: str, *, path: str | None = None, value: object = None) -> None:
    global _INVALID_CONFIG_ATTEMPTS
    _INVALID_CONFIG_ATTEMPTS += 1
    field_key = str(path or kind)
    _INVALID_CONFIG_ATTEMPTS_BY_FIELD[field_key] = int(_INVALID_CONFIG_ATTEMPTS_BY_FIELD.get(field_key, 0)) + 1
    _log_event(
        "registry_preferences_invalid_config",
        level=logging.WARNING,
        kind=str(kind),
        path=path,
        value=str(value),
        invalid_config_attempts=int(_INVALID_CONFIG_ATTEMPTS),
    )


def _decode_size(size: object, *, path: str) -> int | None:
    if size is None:
        return None
    try:
        return int(size)
    except Exception as exc:
        raise ValueError(f"Invalid size at {path}: {size!r}") from exc


def _decode_positive_int(value: object, *, path: str, default: int) -> int:
    raw = default if value is None else value
    try:
        parsed = int(raw)
    except Exception as exc:
        _record_invalid_config("positive_int", path=path, value=raw)
        raise ValueError(f"Invalid integer at {path}: {raw!r}") from exc
    if parsed <= 0:
        _record_invalid_config("positive_int", path=path, value=raw)
        raise ValueError(f"Expected positive integer at {path}: {raw!r}")
    return parsed


def _decode_string(value: object, *, path: str, default: str) -> str:
    raw = default if value is None else value
    text = str(raw or "").strip()
    if not text:
        _record_invalid_config("string", path=path, value=raw)
        raise ValueError(f"Expected non-empty string at {path}")
    return text


def _decode_optional_string(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        _record_invalid_config("optional_string", path=path, value=value)
        raise ValueError(f"Expected non-empty optional string at {path}")
    return text


def _decode_string_tuple(value: object, *, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        _record_invalid_config("string_list", path=path, value=value)
        raise ValueError(f"Expected iterable of strings at {path}")
    try:
        iterator = iter(value)
    except Exception as exc:
        _record_invalid_config("string_list", path=path, value=value)
        raise ValueError(f"Expected iterable of strings at {path}") from exc
    return tuple(str(item) for item in iterator)


def _normalize_match_values(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(value).lower() for value in values)


def _normalize_workflow(workflow: object) -> str:
    value = str(workflow or _WORKFLOW_ELT).strip().lower()
    if value != _WORKFLOW_ELT:
        raise ValueError("workflow must be 'elt'")
    return value


def _normalize_pipeline(value: object, *, path: str) -> str:
    pipeline = str(value or _PIPELINE_PARQUET_DUCKDB).strip().lower()
    if pipeline != _PIPELINE_PARQUET_DUCKDB:
        _record_invalid_config("pipeline", path=path, value=pipeline)
        raise ValueError(
            f"Invalid pipeline at {path}: {pipeline!r}. Expected {_PIPELINE_PARQUET_DUCKDB!r}."
        )
    return pipeline


def _normalize_parallel_profile(value: object, *, path: str) -> str:
    profile = str(value or _PRODUCTION_PARALLEL_PROFILE_DEFAULT).strip().lower()
    if profile not in VALID_PARALLEL_PROFILES:
        _record_invalid_config("parallel_profile", path=path, value=profile)
        choices = ", ".join(sorted(VALID_PARALLEL_PROFILES))
        raise ValueError(f"Invalid parallel_profile at {path}: {profile!r}. Expected one of: {choices}")
    return profile


def _normalize_ordered(value: object, *, path: str) -> str:
    try:
        return normalize_order_mode(
            value,
            default_order_mode=_PRODUCTION_ORDERED_DEFAULT,
            valid_order_modes=VALID_ORDER_MODES,
        )
    except Exception as exc:
        _record_invalid_config("ordered", path=path, value=value)
        choices = ", ".join(sorted(VALID_ORDER_MODES))
        raise ValueError(f"Invalid ordered mode at {path}: {value!r}. Expected one of: {choices}") from exc


def _workers_to_profile_name(workers: int) -> str:
    if int(workers) <= _DEFAULT_WORKERS_TIER:
        return PRODUCTION_PROFILE_ELT_DEFAULT
    if int(workers) <= _SERVER_LIMITED_WORKERS_TIER:
        return PRODUCTION_PROFILE_ELT_SERVER_LIMITED
    return PRODUCTION_PROFILE_ELT_FULL


@dataclass(frozen=True)
class _ProductionProfileConfig:
    workers: int
    parallel_profile: str
    ordered: str
    large_query_mode: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "workflow": _WORKFLOW_ELT,
            "workers": int(self.workers),
            "pipeline": _PIPELINE_PARQUET_DUCKDB,
            "parallel_profile": str(self.parallel_profile),
            "ordered": str(self.ordered),
            "large_query_mode": bool(self.large_query_mode),
        }


@dataclass(frozen=True)
class _MineProfileConfig:
    display_name: str
    host_patterns: tuple[str, ...]
    path_prefixes: tuple[str, ...]
    host_patterns_normalized: tuple[str, ...]
    path_prefixes_normalized: tuple[str, ...]
    default_workers: int
    large_query_threshold_rows: int
    workers_above_threshold: int
    workers_when_size_unknown: int
    production_profile_switch_rows: int
    production_profile: str
    resource_profile: str
    user_agent: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "display_name": self.display_name,
            "host_patterns": list(self.host_patterns),
            "path_prefixes": list(self.path_prefixes),
            "host_patterns_normalized": self.host_patterns_normalized,
            "path_prefixes_normalized": self.path_prefixes_normalized,
            "default_workers": int(self.default_workers),
            "large_query_threshold_rows": int(self.large_query_threshold_rows),
            "workers_above_threshold": int(self.workers_above_threshold),
            "workers_when_size_unknown": int(self.workers_when_size_unknown),
            "production_profile_switch_rows": int(self.production_profile_switch_rows),
            "production_profile": self.production_profile,
            "resource_profile": self.resource_profile,
            "user_agent": self.user_agent,
        }


def _default_production_profile(workers: int) -> dict[str, object]:
    return _ProductionProfileConfig(
        workers=int(workers),
        parallel_profile=_PRODUCTION_PARALLEL_PROFILE_DEFAULT,
        ordered=_PRODUCTION_ORDERED_DEFAULT,
        large_query_mode=True,
    ).as_dict()


_PRODUCTION_PROFILES_BASE: dict[str, dict[str, object]] = {
    PRODUCTION_PROFILE_ELT_DEFAULT: _default_production_profile(_DEFAULT_WORKERS_TIER),
    PRODUCTION_PROFILE_ELT_SERVER_LIMITED: _default_production_profile(_SERVER_LIMITED_WORKERS_TIER),
    PRODUCTION_PROFILE_ELT_FULL: _default_production_profile(_FULL_WORKERS_TIER),
}


def _standard_mine_profile(
    *,
    display_name: str,
    host_patterns: list[str],
    path_prefixes: list[str],
    default_workers: int = _FULL_WORKERS_TIER,
    workers_above_threshold: int | None = None,
    workers_when_size_unknown: int | None = None,
    production_profile_switch_rows: int = _DEFAULT_PROFILE_SWITCH_ROWS,
    production_profile: str | None = None,
    resource_profile: str = _RESOURCE_PROFILE_DEFAULT,
    user_agent: str | None = None,
) -> dict[str, object]:
    default_workers = int(default_workers)
    if workers_above_threshold is None:
        workers_above_threshold = default_workers
    if workers_when_size_unknown is None:
        workers_when_size_unknown = workers_above_threshold
    profile_name = production_profile or _workers_to_profile_name(default_workers)
    return {
        "display_name": str(display_name),
        "host_patterns": list(host_patterns),
        "path_prefixes": list(path_prefixes),
        "default_workers": int(default_workers),
        "large_query_threshold_rows": int(production_profile_switch_rows),
        "workers_above_threshold": int(workers_above_threshold),
        "workers_when_size_unknown": int(workers_when_size_unknown),
        "production_profile_switch_rows": int(production_profile_switch_rows),
        "production_profile": str(profile_name),
        "resource_profile": str(resource_profile),
        "user_agent": user_agent,
    }


_REGISTRY_MINES_BASE: dict[str, dict[str, object]] = {
    "legumemine": _standard_mine_profile(
        display_name="LegumeMine",
        host_patterns=["mines.legumeinfo.org"],
        path_prefixes=["/legumemine"],
        default_workers=_DEFAULT_WORKERS_TIER,
        workers_above_threshold=_DEFAULT_WORKERS_TIER,
        workers_when_size_unknown=_DEFAULT_WORKERS_TIER,
        production_profile=PRODUCTION_PROFILE_ELT_DEFAULT,
    ),
    "maizemine": _standard_mine_profile(
        display_name="MaizeMine",
        host_patterns=["maizemine.rnet.missouri.edu"],
        path_prefixes=["/maizemine"],
        default_workers=_SERVER_LIMITED_WORKERS_TIER,
        workers_above_threshold=_SERVER_LIMITED_WORKERS_TIER,
        workers_when_size_unknown=_SERVER_LIMITED_WORKERS_TIER,
        production_profile=PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
    ),
    "thalemine": _standard_mine_profile(
        display_name="ThaleMine",
        host_patterns=["bar.utoronto.ca"],
        path_prefixes=["/thalemine"],
        user_agent=THALEMINE_BROWSER_USER_AGENT,
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


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _overlay_named_profiles(
    base_profiles: Mapping[str, Mapping[str, object]],
    raw_profiles: object,
) -> dict[str, Mapping[str, object]]:
    merged = dict(base_profiles)
    for name, profile in _as_mapping(raw_profiles).items():
        if not isinstance(profile, Mapping):
            continue
        base = merged.get(str(name))
        if isinstance(base, Mapping):
            merged[str(name)] = {**base, **dict(profile)}
        else:
            merged[str(name)] = dict(profile)
    return merged


def _merge_mines(
    base_registry: Mapping[str, Mapping[str, object]],
    loaded: object,
) -> dict[str, Mapping[str, object]]:
    root = _as_mapping(loaded)
    defaults_block = _as_mapping(root.get("defaults"))
    mine_defaults = _as_mapping(defaults_block.get("mine"))
    raw_mines = _as_mapping(root.get("mines"))
    if not raw_mines and not mine_defaults:
        return dict(base_registry)

    merged: dict[str, Mapping[str, object]] = dict(base_registry)
    for name, profile in raw_mines.items():
        if not isinstance(profile, Mapping):
            continue
        base = _as_mapping(merged.get(str(name), {}))
        merged[str(name)] = {**base, **mine_defaults, **dict(profile)}

    if mine_defaults:
        for name, profile in base_registry.items():
            if str(name) not in raw_mines:
                merged[str(name)] = {**profile, **mine_defaults}
    return merged


def _normalize_production_profile_entry(profile_name: str, profile: Mapping[str, object]) -> dict[str, object]:
    base_path = f"production_profiles.{profile_name}"
    workers_default = int(_PRODUCTION_PROFILES_BASE.get(profile_name, {}).get("workers", _DEFAULT_WORKERS_TIER))
    cfg = _ProductionProfileConfig(
        workers=_decode_positive_int(profile.get("workers"), path=f"{base_path}.workers", default=workers_default),
        parallel_profile=_normalize_parallel_profile(
            profile.get("parallel_profile"),
            path=f"{base_path}.parallel_profile",
        ),
        ordered=_normalize_ordered(profile.get("ordered"), path=f"{base_path}.ordered"),
        large_query_mode=bool(profile.get("large_query_mode", True)),
    )
    _normalize_workflow(profile.get("workflow", _WORKFLOW_ELT))
    _normalize_pipeline(profile.get("pipeline", _PIPELINE_PARQUET_DUCKDB), path=f"{base_path}.pipeline")
    return cfg.as_dict()


def _normalize_production_profiles(profiles: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name, profile in profiles.items():
        if not isinstance(profile, Mapping):
            continue
        out[str(name)] = _normalize_production_profile_entry(str(name), profile)
    return out


def _normalize_mine_profile_entry(profile_name: str, profile: Mapping[str, object]) -> dict[str, object]:
    base_path = f"mines.{profile_name}"
    host_patterns = _decode_string_tuple(profile.get("host_patterns"), path=f"{base_path}.host_patterns")
    path_prefixes = _decode_string_tuple(profile.get("path_prefixes"), path=f"{base_path}.path_prefixes")

    default_workers = _decode_positive_int(
        profile.get("default_workers"),
        path=f"{base_path}.default_workers",
        default=_DEFAULT_PARALLEL_WORKERS,
    )
    workers_above_threshold = _decode_positive_int(
        profile.get("workers_above_threshold"),
        path=f"{base_path}.workers_above_threshold",
        default=default_workers,
    )
    workers_when_size_unknown = _decode_positive_int(
        profile.get("workers_when_size_unknown"),
        path=f"{base_path}.workers_when_size_unknown",
        default=workers_above_threshold,
    )
    large_query_threshold_rows = _decode_positive_int(
        profile.get("large_query_threshold_rows"),
        path=f"{base_path}.large_query_threshold_rows",
        default=_DEFAULT_PROFILE_SWITCH_ROWS,
    )
    production_profile_switch_rows = _decode_positive_int(
        profile.get("production_profile_switch_rows"),
        path=f"{base_path}.production_profile_switch_rows",
        default=large_query_threshold_rows,
    )

    profile_default = _workers_to_profile_name(default_workers)
    production_profile = _decode_string(
        profile.get("production_profile"),
        path=f"{base_path}.production_profile",
        default=profile_default,
    )

    resource_profile = _decode_string(
        profile.get("resource_profile"),
        path=f"{base_path}.resource_profile",
        default=_RESOURCE_PROFILE_DEFAULT,
    )

    cfg = _MineProfileConfig(
        display_name=_decode_string(profile.get("display_name"), path=f"{base_path}.display_name", default=profile_name),
        host_patterns=host_patterns,
        path_prefixes=path_prefixes,
        host_patterns_normalized=_normalize_match_values(host_patterns),
        path_prefixes_normalized=_normalize_match_values(path_prefixes),
        default_workers=default_workers,
        large_query_threshold_rows=large_query_threshold_rows,
        workers_above_threshold=workers_above_threshold,
        workers_when_size_unknown=workers_when_size_unknown,
        production_profile_switch_rows=production_profile_switch_rows,
        production_profile=production_profile,
        resource_profile=resource_profile,
        user_agent=_decode_optional_string(profile.get("user_agent"), path=f"{base_path}.user_agent"),
    )
    return cfg.as_dict()


def _normalize_mines_for_matching(mines: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name, profile in mines.items():
        if not isinstance(profile, Mapping):
            continue
        out[str(name)] = _normalize_mine_profile_entry(str(name), profile)
    return out


def _value_distribution(mapping: Mapping[str, Mapping[str, object]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for profile in mapping.values():
        value = profile.get(field)
        token = str(value)
        counts[token] = int(counts.get(token, 0)) + 1
    return counts


def _load_registry() -> dict[str, object]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    merged_mines: dict[str, Mapping[str, object]] = dict(_REGISTRY_MINES_BASE)
    merged_profiles: dict[str, Mapping[str, object]] = dict(_PRODUCTION_PROFILES_BASE)
    config_source = "defaults_fallback"
    config_path = None

    packaged = load_packaged_mine_parallel_preferences_detailed()
    packaged_payload = packaged.get("payload")
    packaged_path = str(packaged.get("path")) if packaged.get("path") else None
    if bool(packaged.get("ok")) and isinstance(packaged_payload, Mapping):
        merged_mines = _merge_mines(_REGISTRY_MINES_BASE, packaged_payload)
        merged_profiles = _overlay_named_profiles(_PRODUCTION_PROFILES_BASE, packaged_payload.get("production_profiles"))
        config_source = "packaged_mine_parallel_preferences_toml"
        config_path = packaged_path

    loaded = load_mine_parallel_preferences()
    try:
        active_config_path = str(resolve_mine_parallel_preferences_path())
    except Exception:
        active_config_path = None

    if isinstance(loaded, Mapping) and active_config_path and active_config_path != packaged_path:
        merged_mines = _merge_mines(merged_mines, loaded)
        merged_profiles = _overlay_named_profiles(merged_profiles, loaded.get("production_profiles"))
        config_source = f"{config_source}+override_toml"
        config_path = active_config_path

    production_profiles = _normalize_production_profiles(merged_profiles)
    mines = _normalize_mines_for_matching(merged_mines)
    metrics = {
        "pipeline_distribution": _value_distribution(production_profiles, "pipeline"),
        "parallel_profile_distribution": _value_distribution(production_profiles, "parallel_profile"),
        "production_workers_distribution": _value_distribution(production_profiles, "workers"),
        "mine_default_workers_distribution": _value_distribution(mines, "default_workers"),
        "mine_threshold_rows_distribution": _value_distribution(mines, "production_profile_switch_rows"),
    }

    _CACHE = {
        "production_profiles": production_profiles,
        "mines": mines,
        "_config_source": config_source,
        "_config_path": config_path,
        "_normalization_metrics": metrics,
    }
    _log_event(
        "registry_preferences_cache_build",
        cache_populated=True,
        mine_count=len(mines),
        production_profile_count=len(production_profiles),
        config_source=config_source,
        config_path=config_path,
        invalid_config_attempts=int(_INVALID_CONFIG_ATTEMPTS),
    )
    return _CACHE


def _normalize_service_root(service_root: object) -> tuple[str, str]:
    if not service_root:
        return "", ""
    parsed = urlparse(str(service_root))
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/service"):
        path = path[: -len("/service")]
    return host, path


def _matches_profile(profile: Mapping[str, object], host: str, path: str) -> bool:
    host_patterns = profile.get("host_patterns_normalized")
    if not isinstance(host_patterns, tuple):
        host_patterns = tuple()
    if host_patterns and host not in host_patterns:
        return False

    prefixes = profile.get("path_prefixes_normalized")
    if not isinstance(prefixes, tuple):
        prefixes = tuple()
    if not prefixes:
        return True

    path_lower = path.lower()
    return any(path_lower.startswith(prefix) for prefix in prefixes)


def _match_mine_profile(service_root: object, *, mines: Mapping[str, Mapping[str, object]] | None = None):
    host, path = _normalize_service_root(service_root)
    if not host:
        return None
    mine_profiles = mines if isinstance(mines, Mapping) else _load_registry().get("mines", {})
    if not isinstance(mine_profiles, Mapping):
        return None
    for profile in mine_profiles.values():
        if isinstance(profile, Mapping) and _matches_profile(profile, host, path):
            return profile
    return None


def _select_mine_workers(mine_profile: Mapping[str, object], size: int | None) -> int:
    threshold = int(mine_profile.get("production_profile_switch_rows", _DEFAULT_PROFILE_SWITCH_ROWS))
    if size is None:
        return int(mine_profile.get("workers_when_size_unknown", mine_profile.get("workers_above_threshold", _DEFAULT_PARALLEL_WORKERS)))
    if int(size) > threshold:
        return int(mine_profile.get("workers_above_threshold", mine_profile.get("default_workers", _DEFAULT_PARALLEL_WORKERS)))
    return int(mine_profile.get("default_workers", _DEFAULT_PARALLEL_WORKERS))


def _normalize_profile_name(name: str, profiles: Mapping[str, Mapping[str, object]]) -> str:
    if name in profiles:
        return name
    fallback = PRODUCTION_PROFILE_ELT_DEFAULT
    if fallback in profiles:
        return fallback
    return next(iter(profiles))


def _resolve_profile_name(
    *,
    explicit_profile: str,
    mine_profile: Mapping[str, object] | None,
    size: int | None,
    profiles: Mapping[str, Mapping[str, object]],
) -> str:
    token = str(explicit_profile or "").strip().lower()
    if token and token != "auto":
        return _normalize_profile_name(token, profiles)

    if not isinstance(mine_profile, Mapping):
        return _normalize_profile_name(PRODUCTION_PROFILE_ELT_DEFAULT, profiles)

    configured_profile = str(mine_profile.get("production_profile", "")).strip()
    if configured_profile:
        return _normalize_profile_name(configured_profile, profiles)

    workers = _select_mine_workers(mine_profile, size)
    return _normalize_profile_name(_workers_to_profile_name(workers), profiles)


def _production_profile_to_plan(name: str, profile_data: Mapping[str, object]) -> dict[str, object]:
    return {
        "name": str(name),
        "workflow": _WORKFLOW_ELT,
        "workers": int(profile_data.get("workers", _DEFAULT_WORKERS_TIER)),
        "pipeline": _PIPELINE_PARQUET_DUCKDB,
        "parallel_profile": _normalize_parallel_profile(
            profile_data.get("parallel_profile"),
            path=f"production_profiles.{name}.parallel_profile",
        ),
        "ordered": _normalize_ordered(profile_data.get("ordered"), path=f"production_profiles.{name}.ordered"),
        "large_query_mode": bool(profile_data.get("large_query_mode", True)),
    }


def resolve_production_plan(
    service_root,
    size,
    *,
    workflow=_WORKFLOW_ELT,
    production_profile="auto",
):
    _normalize_workflow(workflow)
    size_value = _decode_size(size, path="resolve_production_plan.size")
    registry = _load_registry()
    profiles = registry.get("production_profiles", {})
    if not isinstance(profiles, Mapping) or not profiles:
        profiles = _PRODUCTION_PROFILES_BASE

    mine_profile = _match_mine_profile(service_root, mines=registry.get("mines", {}))
    profile_name = _resolve_profile_name(
        explicit_profile=str(production_profile),
        mine_profile=mine_profile,
        size=size_value,
        profiles=profiles,
    )
    profile_data = profiles.get(profile_name)
    if not isinstance(profile_data, Mapping):
        profile_name = _normalize_profile_name(PRODUCTION_PROFILE_ELT_DEFAULT, profiles)
        profile_data = profiles[profile_name]

    plan = _production_profile_to_plan(profile_name, profile_data)
    if isinstance(mine_profile, Mapping):
        plan["resource_profile"] = str(mine_profile.get("resource_profile", _RESOURCE_PROFILE_DEFAULT))
    else:
        plan["resource_profile"] = _RESOURCE_PROFILE_DEFAULT
    return plan


def resolve_production_resource_profile(
    service_root,
    size,
    *,
    workflow=_WORKFLOW_ELT,
    production_profile="auto",
    fallback_resource_profile=_RESOURCE_PROFILE_DEFAULT,
):
    _normalize_workflow(workflow)
    _decode_size(size, path="resolve_production_resource_profile.size")
    plan = resolve_production_plan(
        service_root,
        size,
        workflow=workflow,
        production_profile=production_profile,
    )
    resolved = str(plan.get("resource_profile", fallback_resource_profile) or fallback_resource_profile).strip()
    if not resolved:
        return str(fallback_resource_profile)
    return resolved


def resolve_preferred_workers(service_root, size, fallback_workers):
    size_value = _decode_size(size, path="resolve_preferred_workers.size")
    mine_profile = _match_mine_profile(service_root, mines=_load_registry().get("mines", {}))
    if not isinstance(mine_profile, Mapping):
        return fallback_workers
    try:
        return int(_select_mine_workers(mine_profile, size_value))
    except Exception:
        return fallback_workers


def resolve_mine_transport_policy(service_root):
    mine_profile = _match_mine_profile(service_root, mines=_load_registry().get("mines", {}))
    if not isinstance(mine_profile, Mapping):
        return {}
    user_agent = mine_profile.get("user_agent")
    if not isinstance(user_agent, str):
        return {}
    text = user_agent.strip()
    if not text:
        return {}
    return {"user_agent": text}


def resolve_mine_user_agent(service_root):
    value = resolve_mine_transport_policy(service_root).get("user_agent")
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text


def registry_preferences_metrics():
    registry = _load_registry()
    metrics = dict(registry.get("_normalization_metrics", {}))
    metrics["invalid_config_attempts"] = int(_INVALID_CONFIG_ATTEMPTS)
    metrics["invalid_config_attempts_by_field"] = dict(_INVALID_CONFIG_ATTEMPTS_BY_FIELD)
    metrics["config_source"] = registry.get("_config_source")
    return metrics
