from __future__ import annotations

import atexit
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources as importlib_resources
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None

_RESOURCE_PACKAGE = "intermine314.config"
_RUNTIME_DEFAULTS_FILE = "runtime-defaults.toml"
_RUNTIME_DEFAULTS_LEGACY_FILE = "defaults.toml"
_MAX_CONFIG_FILE_BYTES = 1_048_576

_RESOURCE_PATH_CACHE: dict[str, Path] = {}
_RESOURCE_TMPDIR: tempfile.TemporaryDirectory | None = None
_RESOURCE_TMPDIR_CLEANUP_REGISTERED = False


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

    def effective_runtime_defaults(self) -> ConfigDocument:
        if _document_usable(self.runtime_defaults_override):
            return self.runtime_defaults_override  # type: ignore[return-value]
        return self.runtime_defaults_packaged


def _document_usable(doc: ConfigDocument | None) -> bool:
    if doc is None:
        return False
    return bool(doc.ok) and isinstance(doc.payload, Mapping)


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
    )


def clear_config_cache() -> None:
    load_config.cache_clear()
    _load_toml_cached.cache_clear()


__all__ = [
    "ConfigDocument",
    "ConfigBundle",
    "resolve_runtime_defaults_path",
    "load_toml_detailed",
    "load_toml",
    "load_config",
    "clear_config_cache",
]
