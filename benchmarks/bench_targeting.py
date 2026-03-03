from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except Exception:  # pragma: no cover - Python 3.14 includes tomllib
    tomllib = None

from benchmarks.bench_utils import merge_shallow_dict, normalize_string_list


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_CONFIG_PATH = REPO_ROOT / "benchmarks" / "profiles" / "benchmark-targets.toml"
TARGET_CONFIG_CACHE: dict[str, Any] | None = None


def get_target_defaults(target_config: dict[str, Any]) -> dict[str, Any]:
    defaults = target_config.get("defaults", {}) if isinstance(target_config, dict) else {}
    if not isinstance(defaults, dict):
        return {}
    target_defaults = defaults.get("target", {})
    if not isinstance(target_defaults, dict):
        return {}
    return target_defaults


def _resolve_named_profile(
    profile_name: str | None,
    profile_map: dict[str, Any],
) -> dict[str, Any]:
    if not profile_name:
        return {}
    profile = profile_map.get(str(profile_name).strip())
    if isinstance(profile, dict):
        return profile
    return {}


def load_target_config() -> dict[str, Any]:
    global TARGET_CONFIG_CACHE
    if TARGET_CONFIG_CACHE is not None:
        return TARGET_CONFIG_CACHE
    out: dict[str, Any] = {}
    if tomllib is not None and TARGET_CONFIG_PATH.exists():
        try:
            parsed = tomllib.loads(TARGET_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                out = parsed
        except Exception:
            out = {}
    TARGET_CONFIG_CACHE = out
    return out


def _load_target_presets(target_config: dict[str, Any]) -> dict[str, Any]:
    targets = target_config.get("targets", {})
    return targets if isinstance(targets, dict) else {}


def normalize_target_settings(
    target_name: str,
    target_config: dict[str, Any],
    target_defaults: dict[str, Any],
) -> dict[str, Any] | None:
    if target_name == "auto":
        return None
    target = _load_target_presets(target_config).get(target_name)
    if not isinstance(target, dict):
        return None

    settings: dict[str, Any] = {}
    if target_defaults:
        settings.update(target_defaults)
    settings = merge_shallow_dict(settings, target)

    query_profiles = target_defaults.get("query_profiles", {})
    if not isinstance(query_profiles, dict):
        query_profiles = {}
    query_profile_base = _resolve_named_profile(settings.get("query_profile"), query_profiles)
    if query_profile_base:
        settings = merge_shallow_dict(query_profile_base, settings)
    settings.pop("query_profile", None)

    settings["views"] = normalize_string_list(settings.get("views"))
    settings["joins"] = normalize_string_list(settings.get("joins"))
    settings.pop("query_profiles", None)
    return settings


def resolve_benchmark_profile(base_profile: str, target_settings: dict[str, Any] | None) -> str:
    token = str(base_profile or "").strip().lower()
    if token and token != "auto":
        return token
    if not target_settings:
        return "auto"
    configured = str(target_settings.get("benchmark_profile", "")).strip().lower()
    if configured:
        return configured
    if "server_restricted" in target_settings:
        return "server_restricted" if bool(target_settings.get("server_restricted")) else "non_restricted"
    return "auto"
