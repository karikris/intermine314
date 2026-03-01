from __future__ import annotations

from copy import deepcopy
import os
import tempfile
from functools import lru_cache
from importlib import resources as importlib_resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None

_RESOURCE_PACKAGE = "intermine314.config"
_RUNTIME_DEFAULTS_FILE = "defaults.toml"
_MINE_PARALLEL_PREFERENCES_FILE = "mine-parallel-preferences.toml"
_MAX_CONFIG_FILE_BYTES = 1_048_576
_RESOURCE_PATH_CACHE: dict[str, Path] = {}
_RESOURCE_TMPDIR: tempfile.TemporaryDirectory | None = None


def _pkg_config_path(filename: str) -> Path:
    resource = importlib_resources.files(_RESOURCE_PACKAGE).joinpath(filename)
    try:
        return Path(resource)
    except Exception:
        global _RESOURCE_TMPDIR
        if _RESOURCE_TMPDIR is None:
            _RESOURCE_TMPDIR = tempfile.TemporaryDirectory(prefix="intermine314-config-")
        cached = _RESOURCE_PATH_CACHE.get(filename)
        if cached is not None and cached.exists():
            return cached
        materialized = Path(_RESOURCE_TMPDIR.name) / filename
        materialized.write_bytes(resource.read_bytes())
        _RESOURCE_PATH_CACHE[filename] = materialized
        return materialized


def resolve_runtime_defaults_path() -> Path:
    override = os.getenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", "").strip()
    if override:
        return Path(override)
    return _pkg_config_path(_RUNTIME_DEFAULTS_FILE)


def resolve_mine_parallel_preferences_path() -> Path:
    override = os.getenv("INTERMINE314_MINE_PARALLEL_PREFERENCES_PATH", "").strip()
    if override:
        return Path(override)
    return _pkg_config_path(_MINE_PARALLEL_PREFERENCES_FILE)


def _path_cache_key(path: Path):
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


def _immutable_payload(value: Any):
    if isinstance(value, dict):
        frozen = {str(key): _immutable_payload(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_immutable_payload(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_immutable_payload(item) for item in value)
    return value


def _clone_payload(value: Any):
    try:
        return deepcopy(value)
    except Exception:
        return value


def _prepare_payload(payload: dict[str, Any], *, read_only: bool):
    if read_only:
        return _immutable_payload(payload)
    return _clone_payload(payload)


@lru_cache(maxsize=64)
def _load_toml_cached(path_str: str, mtime_ns: int, size_bytes: int):
    _ = (mtime_ns, size_bytes)
    path = Path(path_str)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_toml_detailed(path: Path, *, read_only: bool = False) -> dict:
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


def load_toml(path: Path, *, read_only: bool = False):
    result = load_toml_detailed(path, read_only=read_only)
    payload = result.get("payload")
    if read_only:
        return payload if isinstance(payload, MappingProxyType) else MappingProxyType({})
    return payload if isinstance(payload, dict) else {}


def _load_toml_with_override(env_var: str, filename: str, *, read_only: bool = False):
    override = os.getenv(env_var, "").strip()
    if override:
        return load_toml(Path(override), read_only=read_only)
    return load_toml(_pkg_config_path(filename), read_only=read_only)


def load_runtime_defaults_detailed(*, read_only: bool = False) -> dict:
    override = os.getenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", "").strip()
    if override:
        result = load_toml_detailed(Path(override), read_only=read_only)
        result["source"] = "override_toml"
        return result
    return load_packaged_runtime_defaults_detailed(read_only=read_only)


def load_packaged_runtime_defaults_detailed(*, read_only: bool = False) -> dict:
    result = load_toml_detailed(_pkg_config_path(_RUNTIME_DEFAULTS_FILE), read_only=read_only)
    result["source"] = "packaged_toml"
    return result


def load_runtime_defaults_override_detailed(*, read_only: bool = False) -> dict | None:
    override = os.getenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", "").strip()
    if not override:
        return None
    result = load_toml_detailed(Path(override), read_only=read_only)
    result["source"] = "override_toml"
    return result


def load_runtime_defaults(*, read_only: bool = False):
    result = load_runtime_defaults_detailed(read_only=read_only)
    payload = result.get("payload")
    if read_only:
        return payload if isinstance(payload, MappingProxyType) else MappingProxyType({})
    return payload if isinstance(payload, dict) else {}


def load_mine_parallel_preferences(*, read_only: bool = False):
    return _load_toml_with_override(
        "INTERMINE314_MINE_PARALLEL_PREFERENCES_PATH",
        _MINE_PARALLEL_PREFERENCES_FILE,
        read_only=read_only,
    )
