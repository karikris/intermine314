from __future__ import annotations

import atexit
from copy import deepcopy
from dataclasses import dataclass
import os
import tempfile
from functools import lru_cache
from importlib import resources as importlib_resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlparse

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None

_RESOURCE_PACKAGE = "intermine314.config"
_RUNTIME_DEFAULTS_FILE = "runtime-defaults.toml"
_RUNTIME_DEFAULTS_LEGACY_FILE = "defaults.toml"
_MINE_PARALLEL_PREFERENCES_FILE = "mine-parallel-preferences.toml"
_PARALLEL_PROFILES_FILE = "parallel-profiles.toml"

_MAX_CONFIG_FILE_BYTES = 1_048_576
_RESOURCE_PATH_CACHE: dict[str, Path] = {}
_RESOURCE_TMPDIR: tempfile.TemporaryDirectory | None = None
_RESOURCE_TMPDIR_CLEANUP_REGISTERED = False

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

_THALEMINE_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)


def _cleanup_resource_tmpdir() -> None:
    global _RESOURCE_TMPDIR
    _RESOURCE_PATH_CACHE.clear()
    tmpdir = _RESOURCE_TMPDIR
    _RESOURCE_TMPDIR = None
    if tmpdir is None:
        return
    try:
        tmpdir.cleanup()
    except Exception:
        return


def _ensure_resource_tmpdir_cleanup_registered() -> None:
    global _RESOURCE_TMPDIR_CLEANUP_REGISTERED
    if _RESOURCE_TMPDIR_CLEANUP_REGISTERED:
        return
    atexit.register(_cleanup_resource_tmpdir)
    _RESOURCE_TMPDIR_CLEANUP_REGISTERED = True


def _pkg_config_path(filename: str) -> Path:
    resource = importlib_resources.files(_RESOURCE_PACKAGE).joinpath(filename)
    try:
        candidate = Path(resource)
        if candidate.exists():
            return candidate
    except Exception:
        pass

    global _RESOURCE_TMPDIR
    if _RESOURCE_TMPDIR is None:
        _RESOURCE_TMPDIR = tempfile.TemporaryDirectory(prefix="intermine314-config-")
        _ensure_resource_tmpdir_cleanup_registered()
    cached = _RESOURCE_PATH_CACHE.get(filename)
    if cached is not None and cached.exists():
        return cached
    materialized = Path(_RESOURCE_TMPDIR.name) / filename
    materialized.write_bytes(resource.read_bytes())
    _RESOURCE_PATH_CACHE[filename] = materialized
    return materialized


def _resolve_packaged_runtime_defaults_path() -> Path:
    try:
        return _pkg_config_path(_RUNTIME_DEFAULTS_FILE)
    except Exception:
        return _pkg_config_path(_RUNTIME_DEFAULTS_LEGACY_FILE)


def resolve_runtime_defaults_path() -> Path:
    override = os.getenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", "").strip()
    if override:
        return Path(override)
    return _resolve_packaged_runtime_defaults_path()


def resolve_mine_parallel_preferences_path() -> Path:
    override = os.getenv("INTERMINE314_MINE_PARALLEL_PREFERENCES_PATH", "").strip()
    if override:
        return Path(override)
    return _pkg_config_path(_MINE_PARALLEL_PREFERENCES_FILE)


def resolve_parallel_profiles_path() -> Path:
    override = os.getenv("INTERMINE314_PARALLEL_PROFILES_PATH", "").strip()
    if override:
        return Path(override)
    return _pkg_config_path(_PARALLEL_PROFILES_FILE)


def _path_cache_key(path: Path) -> tuple[str, int, int]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


def _immutable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        frozen = {str(key): _immutable_payload(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_immutable_payload(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_immutable_payload(item) for item in value)
    return value


def _clone_payload(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        return value


def _prepare_payload(payload: dict[str, Any], *, read_only: bool) -> Any:
    if read_only:
        return _immutable_payload(payload)
    return _clone_payload(payload)


@lru_cache(maxsize=64)
def _load_toml_cached(path_str: str, mtime_ns: int, size_bytes: int) -> dict[str, Any]:
    _ = (mtime_ns, size_bytes)
    path = Path(path_str)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_toml_detailed(path: Path, *, read_only: bool = False) -> dict[str, Any]:
    path_str = str(path)
    if tomllib is None:
        return {
            "ok": False,
            "payload": {},
            "path": path_str,
            "error_kind": "tomllib_unavailable",
        }
    try:
        cache_key = _path_cache_key(path)
    except FileNotFoundError:
        return {
            "ok": False,
            "payload": {},
            "path": path_str,
            "error_kind": "missing",
        }
    except Exception:
        return {
            "ok": False,
            "payload": {},
            "path": path_str,
            "error_kind": "unreadable",
        }
    if cache_key[2] > _MAX_CONFIG_FILE_BYTES:
        return {
            "ok": False,
            "payload": {},
            "path": cache_key[0],
            "error_kind": "oversized",
            "size_bytes": int(cache_key[2]),
        }
    try:
        loaded = _load_toml_cached(*cache_key)
    except Exception:
        return {
            "ok": False,
            "payload": {},
            "path": cache_key[0],
            "error_kind": "invalid_toml",
        }
    if not isinstance(loaded, dict):
        return {
            "ok": False,
            "payload": {},
            "path": cache_key[0],
            "error_kind": "invalid_shape",
        }
    return {
        "ok": True,
        "payload": _prepare_payload(loaded, read_only=bool(read_only)),
        "path": cache_key[0],
        "error_kind": None,
        "size_bytes": int(cache_key[2]),
    }


def load_toml(path: Path, *, read_only: bool = False) -> Mapping[str, Any]:
    result = load_toml_detailed(path, read_only=read_only)
    payload = result.get("payload")
    if read_only:
        return payload if isinstance(payload, MappingProxyType) else MappingProxyType({})
    return payload if isinstance(payload, Mapping) else {}


@dataclass(frozen=True)
class ConfigDocument:
    payload: Mapping[str, Any]
    ok: bool
    source: str
    path: str | None
    error_kind: str | None


@dataclass(frozen=True)
class ConfigBundle:
    runtime_defaults_packaged: ConfigDocument
    runtime_defaults_override: ConfigDocument | None
    mine_parallel_preferences_packaged: ConfigDocument
    mine_parallel_preferences_override: ConfigDocument | None
    parallel_profiles_packaged: ConfigDocument
    parallel_profiles_override: ConfigDocument | None

    def effective_runtime_defaults(self) -> ConfigDocument:
        if _document_usable(self.runtime_defaults_override):
            return self.runtime_defaults_override  # type: ignore[return-value]
        return self.runtime_defaults_packaged

    def effective_mine_parallel_preferences(self) -> ConfigDocument:
        if _document_usable(self.mine_parallel_preferences_override):
            return self.mine_parallel_preferences_override  # type: ignore[return-value]
        return self.mine_parallel_preferences_packaged

    def effective_parallel_profiles(self) -> ConfigDocument:
        if _document_usable(self.parallel_profiles_override):
            return self.parallel_profiles_override  # type: ignore[return-value]
        return self.parallel_profiles_packaged


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


def _document_usable(doc: ConfigDocument | None) -> bool:
    if doc is None:
        return False
    return bool(doc.ok) and isinstance(doc.payload, Mapping)


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _to_document(result: Mapping[str, Any], *, source: str) -> ConfigDocument:
    payload = result.get("payload")
    if isinstance(payload, MappingProxyType):
        normalized_payload: Mapping[str, Any] = payload
    elif isinstance(payload, Mapping):
        normalized_payload = MappingProxyType(dict(payload))
    else:
        normalized_payload = MappingProxyType({})
    return ConfigDocument(
        payload=normalized_payload,
        ok=bool(result.get("ok")),
        source=source,
        path=str(result.get("path")) if result.get("path") else None,
        error_kind=str(result.get("error_kind")) if result.get("error_kind") else None,
    )


def _load_document(path: Path, *, source: str) -> ConfigDocument:
    result = load_toml_detailed(path, read_only=True)
    return _to_document(result, source=source)


def _load_override_document(env_var: str) -> ConfigDocument | None:
    override = os.getenv(env_var, "").strip()
    if not override:
        return None
    return _load_document(Path(override), source="override_toml")


@lru_cache(maxsize=1)
def load_config() -> ConfigBundle:
    return ConfigBundle(
        runtime_defaults_packaged=_load_document(_resolve_packaged_runtime_defaults_path(), source="packaged_toml"),
        runtime_defaults_override=_load_override_document("INTERMINE314_RUNTIME_DEFAULTS_PATH"),
        mine_parallel_preferences_packaged=_load_document(
            _pkg_config_path(_MINE_PARALLEL_PREFERENCES_FILE),
            source="packaged_toml",
        ),
        mine_parallel_preferences_override=_load_override_document("INTERMINE314_MINE_PARALLEL_PREFERENCES_PATH"),
        parallel_profiles_packaged=_load_document(_pkg_config_path(_PARALLEL_PROFILES_FILE), source="packaged_toml"),
        parallel_profiles_override=_load_override_document("INTERMINE314_PARALLEL_PROFILES_PATH"),
    )


def clear_config_cache() -> None:
    load_config.cache_clear()


def _parse_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    if parsed <= 0:
        return int(default)
    return int(parsed)


def _normalize_mode(value: Any, *, default: str, valid_modes: set[str] | frozenset[str]) -> str:
    if isinstance(value, bool):
        return "ordered" if value else "unordered"
    text = str(value or "").strip().lower()
    if not text:
        return str(default)
    if text not in valid_modes:
        return str(default)
    return text


def _default_production_profiles(*, workers_default: int, workers_server_limited: int, workers_full: int) -> dict[str, dict[str, Any]]:
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
            "resource_profile": "default",
        },
        "maizemine": {
            "display_name": "MaizeMine",
            "host_patterns": ["maizemine.rnet.missouri.edu"],
            "path_prefixes": ["/maizemine"],
            "production_profile": _PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
            "resource_profile": "default",
        },
        "thalemine": {
            "display_name": "ThaleMine",
            "host_patterns": ["bar.utoronto.ca"],
            "path_prefixes": ["/thalemine"],
            "production_profile": _PRODUCTION_PROFILE_ELT_FULL,
            "resource_profile": "default",
            "user_agent": _THALEMINE_BROWSER_USER_AGENT,
        },
        "oakmine": {
            "display_name": "OakMine",
            "host_patterns": ["urgi.versailles.inra.fr", "urgi.versailles.inrae.fr"],
            "path_prefixes": ["/OakMine_PM1N"],
            "production_profile": _PRODUCTION_PROFILE_ELT_FULL,
            "resource_profile": "default",
        },
        "wheatmine": {
            "display_name": "WheatMine",
            "host_patterns": ["urgi.versailles.inra.fr", "urgi.versailles.inrae.fr"],
            "path_prefixes": ["/WheatMine"],
            "production_profile": _PRODUCTION_PROFILE_ELT_FULL,
            "resource_profile": "default",
        },
    }


def _normalize_service_root(service_root: object) -> tuple[str, str]:
    if not service_root:
        return "", ""
    parsed = urlparse(str(service_root))
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/service"):
        path = path[: -len("/service")]
    return host, path.lower()


def _mine_matches(profile: Mapping[str, Any], host: str, path: str) -> bool:
    host_patterns_raw = profile.get("host_patterns")
    host_patterns = tuple(str(item).lower() for item in host_patterns_raw) if isinstance(host_patterns_raw, (list, tuple)) else ()
    if host_patterns and host not in host_patterns:
        return False
    prefixes_raw = profile.get("path_prefixes")
    prefixes = tuple(str(item).lower() for item in prefixes_raw) if isinstance(prefixes_raw, (list, tuple)) else ()
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


def resolve_parallel_policy(mine_url: object, size: object, user_overrides: Mapping[str, Any] | None = None) -> ResolvedParallelPolicy:
    from intermine314.parallel.policy import (
        VALID_ORDER_MODES,
        VALID_PARALLEL_PROFILES,
    )

    if size is not None:
        try:
            int(size)
        except Exception as exc:
            raise ValueError(f"Invalid size: {size!r}") from exc
    overrides = _to_mapping(user_overrides)

    workflow = str(overrides.get("workflow", _WORKFLOW_ELT) or _WORKFLOW_ELT).strip().lower()
    if workflow != _WORKFLOW_ELT:
        raise ValueError("workflow must be 'elt'")

    bundle = load_config()
    runtime_root = _to_mapping(bundle.effective_runtime_defaults().payload)
    query_defaults = _to_mapping(runtime_root.get("query_defaults"))
    registry_defaults = _to_mapping(runtime_root.get("registry_defaults"))

    workers_default = _parse_positive_int(registry_defaults.get("default_workers_tier"), _DEFAULT_WORKERS_TIER)
    workers_server_limited = _parse_positive_int(
        registry_defaults.get("server_limited_workers_tier"),
        _SERVER_LIMITED_WORKERS_TIER,
    )
    workers_full = _parse_positive_int(registry_defaults.get("full_workers_tier"), _FULL_WORKERS_TIER)

    default_parallel_profile = str(query_defaults.get("default_parallel_profile", "default") or "default").strip().lower()
    if default_parallel_profile not in VALID_PARALLEL_PROFILES:
        default_parallel_profile = "default"
    default_ordered_mode = _normalize_mode(
        query_defaults.get("default_parallel_ordered_mode"),
        default="ordered",
        valid_modes=VALID_ORDER_MODES,
    )
    default_large_query_mode = bool(query_defaults.get("default_large_query_mode", False))

    production_profiles = _default_production_profiles(
        workers_default=workers_default,
        workers_server_limited=workers_server_limited,
        workers_full=workers_full,
    )
    mines = _default_mines()

    mine_root = _to_mapping(bundle.effective_mine_parallel_preferences().payload)
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
        mines[key] = {**base, **mine_defaults, **dict(mine_profile)}

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

    parallel_profiles_root = _to_mapping(bundle.effective_parallel_profiles().payload)
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
    ordered = _normalize_mode(
        ordered_value,
        default=default_ordered_mode,
        valid_modes=VALID_ORDER_MODES,
    )

    if "large_query_mode" in overrides:
        large_query_mode = bool(overrides.get("large_query_mode"))
    elif "large_query_mode" in production_profile:
        large_query_mode = bool(production_profile.get("large_query_mode"))
    elif "large_query_mode" in parallel_profile_defaults:
        large_query_mode = bool(parallel_profile_defaults.get("large_query_mode"))
    else:
        large_query_mode = bool(default_large_query_mode)

    mine_resource_profile = None
    if isinstance(matched_mine, Mapping):
        mine_resource_profile = matched_mine.get("resource_profile")
    if "resource_profile" in overrides:
        resource_profile = str(overrides.get("resource_profile") or "default").strip().lower() or "default"
    elif mine_resource_profile is not None:
        resource_profile = str(mine_resource_profile or "default").strip().lower() or "default"
    else:
        resource_profile = str(mine_defaults.get("resource_profile", "default") or "default").strip().lower() or "default"

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


__all__ = [
    "ConfigDocument",
    "ConfigBundle",
    "ResolvedParallelPolicy",
    "resolve_runtime_defaults_path",
    "resolve_mine_parallel_preferences_path",
    "resolve_parallel_profiles_path",
    "load_toml_detailed",
    "load_toml",
    "load_config",
    "clear_config_cache",
    "resolve_parallel_policy",
]
