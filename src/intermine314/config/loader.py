from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None


def _pkg_config_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def resolve_runtime_defaults_path() -> Path:
    override = os.getenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", "").strip()
    if override:
        return Path(override)
    return _pkg_config_path("defaults.toml")


def resolve_mine_parallel_preferences_path() -> Path:
    override = os.getenv("INTERMINE314_MINE_PARALLEL_PREFERENCES_PATH", "").strip()
    if override:
        return Path(override)
    return _pkg_config_path("mine-parallel-preferences.toml")


def resolve_parallel_profiles_path() -> Path:
    override = os.getenv("INTERMINE314_PARALLEL_PROFILES_PATH", "").strip()
    if override:
        return Path(override)
    return _pkg_config_path("parallel-profiles.toml")


def load_toml(path: Path) -> dict:
    if tomllib is None or not path.exists():
        return {}
    try:
        loaded = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_runtime_defaults() -> dict:
    return load_toml(resolve_runtime_defaults_path())


def load_mine_parallel_preferences() -> dict:
    return load_toml(resolve_mine_parallel_preferences_path())


def load_parallel_profiles() -> dict:
    return load_toml(resolve_parallel_profiles_path())
