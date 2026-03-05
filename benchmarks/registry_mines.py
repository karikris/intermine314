from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from urllib.parse import urlparse

from benchmarks.policy_loader import (
    load_mine_parallel_preferences_document,
    load_mine_profiles,
    resolve_parallel_policy,
)

_WORKFLOW_ELT = "elt"
_PIPELINE_ELT = "parquet_duckdb"
_RESOURCE_PROFILE_DEFAULT = "default"


def _to_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _to_lower_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _normalize_service_root(service_root: object) -> tuple[str, str]:
    if not service_root:
        return "", ""
    parsed = urlparse(str(service_root))
    host = (parsed.hostname or "").strip().lower()
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/service"):
        path = path[: -len("/service")]
    return host, path.lower()


def _matches_profile(profile: Mapping[str, object], host: str, path: str) -> bool:
    host_patterns = profile.get("host_patterns_normalized")
    if not isinstance(host_patterns, tuple):
        host_patterns = _to_lower_tuple(profile.get("host_patterns"))
    if host_patterns and host not in host_patterns:
        return False

    path_prefixes = profile.get("path_prefixes_normalized")
    if not isinstance(path_prefixes, tuple):
        path_prefixes = _to_lower_tuple(profile.get("path_prefixes"))
    if not path_prefixes:
        return True
    return any(path.startswith(prefix) for prefix in path_prefixes)


def _size_or_none(size: object) -> int | None:
    if size is None:
        return None
    return int(size)


@lru_cache(maxsize=1)
def _registry_data() -> dict[str, object]:
    doc = load_mine_parallel_preferences_document()
    mines_raw = load_mine_profiles()

    mines: dict[str, dict[str, object]] = {}
    for name, raw_profile in mines_raw.items():
        profile = _to_mapping(raw_profile)
        if not profile:
            continue
        normalized = dict(profile)
        normalized["host_patterns_normalized"] = _to_lower_tuple(normalized.get("host_patterns"))
        normalized["path_prefixes_normalized"] = _to_lower_tuple(normalized.get("path_prefixes"))
        mines[str(name).strip().lower()] = normalized

    return {
        "mines": mines,
        "config_source": doc.source,
        "config_path": doc.path,
        "config_ok": bool(doc.ok),
    }


def clear_registry_cache() -> None:
    _registry_data.cache_clear()


def _match_mine_profile(service_root: object, *, mines: Mapping[str, Mapping[str, object]] | None = None):
    host, path = _normalize_service_root(service_root)
    if not host:
        return None
    profiles = mines if isinstance(mines, Mapping) else _registry_data().get("mines", {})
    if not isinstance(profiles, Mapping):
        return None
    for profile in profiles.values():
        if isinstance(profile, Mapping) and _matches_profile(profile, host, path):
            return profile
    return None


def resolve_production_plan(
    service_root,
    size,
    *,
    workflow=_WORKFLOW_ELT,
    production_profile="auto",
):
    policy = resolve_parallel_policy(
        service_root,
        _size_or_none(size),
        {
            "workflow": workflow,
            "production_profile": production_profile,
        },
    )
    return {
        "name": policy.production_profile,
        "workflow": policy.workflow,
        "workers": int(policy.workers),
        "pipeline": policy.pipeline,
        "parallel_profile": policy.profile,
        "ordered": policy.ordered,
        "large_query_mode": bool(policy.large_query_mode),
        "resource_profile": policy.resource_profile,
    }


def resolve_production_resource_profile(
    service_root,
    size,
    *,
    workflow=_WORKFLOW_ELT,
    production_profile="auto",
    fallback_resource_profile=_RESOURCE_PROFILE_DEFAULT,
):
    policy = resolve_parallel_policy(
        service_root,
        _size_or_none(size),
        {
            "workflow": workflow,
            "production_profile": production_profile,
        },
    )
    resolved = str(policy.resource_profile or fallback_resource_profile).strip()
    if not resolved:
        return str(fallback_resource_profile)
    return resolved


def resolve_preferred_workers(service_root, size, fallback_workers):
    try:
        policy = resolve_parallel_policy(service_root, _size_or_none(size), None)
        return int(policy.workers)
    except Exception:
        return fallback_workers


def resolve_mine_transport_policy(service_root):
    profile = _match_mine_profile(service_root, mines=_registry_data().get("mines", {}))
    if not isinstance(profile, Mapping):
        return {}
    user_agent = profile.get("user_agent")
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
    data = _registry_data()
    mines = data.get("mines", {})
    mine_profiles = mines.values() if isinstance(mines, Mapping) else ()
    profile_distribution: dict[str, int] = {}
    for profile in mine_profiles:
        if not isinstance(profile, Mapping):
            continue
        token = str(profile.get("production_profile") or "").strip().lower()
        if not token:
            continue
        profile_distribution[token] = int(profile_distribution.get(token, 0)) + 1
    return {
        "config_source": data.get("config_source"),
        "config_path": data.get("config_path"),
        "config_ok": bool(data.get("config_ok", False)),
        "mine_count": len(mines) if isinstance(mines, Mapping) else 0,
        "pipeline_distribution": {_PIPELINE_ELT: int(len(profile_distribution))},
        "mine_production_profile_distribution": profile_distribution,
        "parallel_profile_distribution": {},
        "production_workers_distribution": {},
        "invalid_config_attempts": 0,
        "invalid_config_attempts_by_field": {},
    }


__all__ = [
    "clear_registry_cache",
    "registry_preferences_metrics",
    "resolve_mine_transport_policy",
    "resolve_mine_user_agent",
    "resolve_preferred_workers",
    "resolve_production_plan",
    "resolve_production_resource_profile",
]
