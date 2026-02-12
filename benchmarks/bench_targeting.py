from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except Exception:  # pragma: no cover - Python 3.14 includes tomllib
    tomllib = None

from intermine314.service import Service as NewService
from benchmarks.bench_utils import merge_shallow_dict, normalize_string_list


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_CONFIG_PATH = REPO_ROOT / "benchmarks" / "profiles" / "benchmark-targets.toml"
TARGETED_EXPORT_SETTINGS_KEYS = (
    "enabled",
    "id_path",
    "list_type",
    "list_chunk_size",
    "list_name_prefix",
    "list_description",
    "list_tags",
    "keyset_batch_size",
    "template_keywords",
    "template_limit",
    "tables",
)

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


def _normalize_targeted_tables(
    targeted: dict[str, Any],
    target_defaults: dict[str, Any],
) -> dict[str, Any]:
    table_profiles = target_defaults.get("table_profiles", {})
    if not isinstance(table_profiles, dict):
        table_profiles = {}
    raw_tables = targeted.get("tables", [])
    if not isinstance(raw_tables, list):
        return targeted

    resolved_tables: list[dict[str, Any]] = []
    for table in raw_tables:
        if not isinstance(table, dict):
            continue
        base = _resolve_named_profile(table.get("table_profile"), table_profiles)
        merged = merge_shallow_dict(base, table)
        merged.pop("table_profile", None)
        merged["views"] = normalize_string_list(merged.get("views"))
        merged["joins"] = normalize_string_list(merged.get("joins"))
        resolved_tables.append(merged)
    targeted["tables"] = resolved_tables
    return targeted


def normalize_targeted_settings(
    target_settings: dict[str, Any] | None,
    target_defaults: dict[str, Any],
) -> dict[str, Any]:
    if not target_settings:
        return {}
    targeted = target_settings.get("targeted_exports")
    if not isinstance(targeted, dict):
        return {}
    output: dict[str, Any] = {}
    for key in TARGETED_EXPORT_SETTINGS_KEYS:
        if key in targeted:
            output[key] = targeted[key]
    return _normalize_targeted_tables(output, target_defaults)


def resolve_reachable_mine_url(
    primary_url: str,
    target_settings: dict[str, Any] | None,
    request_timeout: int | float | None = None,
) -> tuple[str, list[dict[str, str]]]:
    candidates: list[str] = []
    seen = set()

    def add_candidate(url: str | None) -> None:
        if not url:
            return
        value = str(url).strip()
        if not value or value in seen:
            return
        seen.add(value)
        candidates.append(value)

    add_candidate(primary_url)
    if target_settings:
        add_candidate(target_settings.get("endpoint"))
        for fallback in target_settings.get("endpoint_fallbacks", []):
            add_candidate(fallback)

    probe_errors: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            if request_timeout is None:
                service = NewService(candidate)
            else:
                service = NewService(candidate, request_timeout=request_timeout)
            _ = service.version
            return candidate, probe_errors
        except Exception as exc:
            probe_errors.append({"url": candidate, "error": str(exc)})
    if probe_errors:
        details = "; ".join(f"{item['url']} => {item['error']}" for item in probe_errors)
        raise RuntimeError(f"No reachable endpoint from configured candidates: {details}")
    return primary_url, probe_errors


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

    defaults = target_config.get("defaults", {}) if isinstance(target_config, dict) else {}
    if isinstance(defaults, dict):
        targeted_defaults = defaults.get("targeted_exports", {})
        if isinstance(targeted_defaults, dict):
            target_targeted = settings.get("targeted_exports", {})
            if isinstance(target_targeted, dict):
                settings["targeted_exports"] = merge_shallow_dict(targeted_defaults, target_targeted)
            elif target_targeted:
                settings["targeted_exports"] = target_targeted
            else:
                settings["targeted_exports"] = dict(targeted_defaults)

    settings["views"] = normalize_string_list(settings.get("views"))
    settings["joins"] = normalize_string_list(settings.get("joins"))
    settings.pop("query_profiles", None)
    settings.pop("table_profiles", None)
    return settings


def profile_for_rows(base_profile: str, target_settings: dict[str, Any] | None, rows_target: int) -> str:
    if base_profile != "auto":
        return base_profile
    if not target_settings:
        return "auto"
    switch = target_settings.get("profile_switch_rows")
    small_profile = target_settings.get("profile_small")
    large_profile = target_settings.get("profile_large")
    if switch is None or not small_profile or not large_profile:
        return "auto"
    try:
        threshold = int(switch)
    except Exception:
        return "auto"
    if rows_target <= threshold:
        return str(small_profile)
    return str(large_profile)
