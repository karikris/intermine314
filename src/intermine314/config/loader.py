from __future__ import annotations

import os
import tempfile
from importlib import resources as importlib_resources
from pathlib import Path

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None

_RESOURCE_PACKAGE = "intermine314.config"
_RUNTIME_DEFAULTS_FILE = "defaults.toml"
_MINE_PARALLEL_PREFERENCES_FILE = "mine-parallel-preferences.toml"
_PARALLEL_PROFILES_FILE = "parallel-profiles.toml"
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
        materialized.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")
        _RESOURCE_PATH_CACHE[filename] = materialized
        return materialized


def _read_pkg_config_text(filename: str) -> str:
    try:
        return importlib_resources.files(_RESOURCE_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
    except Exception:
        fallback = Path(__file__).resolve().parent / filename
        return fallback.read_text(encoding="utf-8")


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


def resolve_parallel_profiles_path() -> Path:
    override = os.getenv("INTERMINE314_PARALLEL_PROFILES_PATH", "").strip()
    if override:
        return Path(override)
    return _pkg_config_path(_PARALLEL_PROFILES_FILE)


def load_toml(path: Path) -> dict:
    if tomllib is None or not path.exists():
        return {}
    try:
        if path.stat().st_size > _MAX_CONFIG_FILE_BYTES:
            return {}
    except Exception:
        return {}
    try:
        loaded = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_packaged_toml(filename: str) -> dict:
    if tomllib is None:
        return {}
    try:
        text = _read_pkg_config_text(filename)
    except Exception:
        return {}
    if len(text) > _MAX_CONFIG_FILE_BYTES:
        return {}
    try:
        loaded = tomllib.loads(text)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_toml_with_override(env_var: str, filename: str) -> dict:
    override = os.getenv(env_var, "").strip()
    if override:
        return load_toml(Path(override))
    return _load_packaged_toml(filename)


def load_runtime_defaults() -> dict:
    return _load_toml_with_override("INTERMINE314_RUNTIME_DEFAULTS_PATH", _RUNTIME_DEFAULTS_FILE)


def load_mine_parallel_preferences() -> dict:
    return _load_toml_with_override(
        "INTERMINE314_MINE_PARALLEL_PREFERENCES_PATH",
        _MINE_PARALLEL_PREFERENCES_FILE,
    )


def load_parallel_profiles() -> dict:
    return _load_toml_with_override("INTERMINE314_PARALLEL_PROFILES_PATH", _PARALLEL_PROFILES_FILE)
