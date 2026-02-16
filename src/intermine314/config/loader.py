from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from importlib import resources as importlib_resources
from pathlib import Path

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


@lru_cache(maxsize=64)
def _load_toml_cached(path_str: str, mtime_ns: int, size_bytes: int) -> dict:
    _ = (mtime_ns, size_bytes)
    path = Path(path_str)
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def load_toml(path: Path) -> dict:
    if tomllib is None:
        return {}
    try:
        cache_key = _path_cache_key(path)
        if cache_key[2] > _MAX_CONFIG_FILE_BYTES:
            return {}
    except Exception:
        return {}
    try:
        loaded = _load_toml_cached(*cache_key)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_toml_with_override(env_var: str, filename: str) -> dict:
    override = os.getenv(env_var, "").strip()
    if override:
        return load_toml(Path(override))
    return load_toml(_pkg_config_path(filename))


def load_runtime_defaults() -> dict:
    return _load_toml_with_override("INTERMINE314_RUNTIME_DEFAULTS_PATH", _RUNTIME_DEFAULTS_FILE)


def load_mine_parallel_preferences() -> dict:
    return _load_toml_with_override(
        "INTERMINE314_MINE_PARALLEL_PREFERENCES_PATH",
        _MINE_PARALLEL_PREFERENCES_FILE,
    )
