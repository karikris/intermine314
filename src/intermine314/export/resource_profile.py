from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import shutil
from typing import Any


RESOURCE_PROFILE_ENV_VAR = "INTERMINE314_RESOURCE_PROFILE"
TEMP_DIR_ENV_VAR = "INTERMINE314_TEMP_DIR"

_TOR_LOW_MEM_MAX_INFLIGHT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ResourceProfile:
    """Resource envelope for bounded parallel exports."""

    name: str = "default"
    max_workers: int | None = None
    ordered: bool | str | None = None
    prefetch: int | None = None
    inflight_limit: int | None = None
    max_inflight_bytes_estimate: int | None = None
    temp_dir: str | Path | None = None
    temp_dir_min_free_bytes: int | None = None


_NAMED_RESOURCE_PROFILES: dict[str, ResourceProfile] = {
    "default": ResourceProfile(name="default"),
    "tor_low_mem": ResourceProfile(
        name="tor_low_mem",
        max_workers=2,
        ordered="window",
        prefetch=2,
        inflight_limit=2,
        max_inflight_bytes_estimate=_TOR_LOW_MEM_MAX_INFLIGHT_BYTES,
        temp_dir="/tmp",
    ),
}


def _optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _optional_non_negative_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _normalize_temp_dir(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).expanduser())


def _normalize_profile_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text:
        return text
    return "default"


def normalize_resource_profile(profile: ResourceProfile) -> ResourceProfile:
    normalized = ResourceProfile(
        name=_normalize_profile_name(profile.name),
        max_workers=_optional_positive_int("max_workers", profile.max_workers),
        ordered=profile.ordered,
        prefetch=_optional_positive_int("prefetch", profile.prefetch),
        inflight_limit=_optional_positive_int("inflight_limit", profile.inflight_limit),
        max_inflight_bytes_estimate=_optional_positive_int(
            "max_inflight_bytes_estimate",
            profile.max_inflight_bytes_estimate,
        ),
        temp_dir=_normalize_temp_dir(profile.temp_dir),
        temp_dir_min_free_bytes=_optional_non_negative_int(
            "temp_dir_min_free_bytes",
            profile.temp_dir_min_free_bytes,
        ),
    )
    return normalized


def named_resource_profiles() -> dict[str, ResourceProfile]:
    return dict(_NAMED_RESOURCE_PROFILES)


def resolve_resource_profile(resource_profile: ResourceProfile | str | None) -> ResourceProfile:
    resolved = resource_profile
    if resolved is None:
        env_name = str(os.getenv(RESOURCE_PROFILE_ENV_VAR, "")).strip().lower()
        if env_name:
            resolved = env_name

    if resolved is None:
        profile = _NAMED_RESOURCE_PROFILES["default"]
    elif isinstance(resolved, ResourceProfile):
        profile = resolved
    else:
        key = str(resolved).strip().lower()
        if not key:
            key = "default"
        if key not in _NAMED_RESOURCE_PROFILES:
            choices = ", ".join(sorted(_NAMED_RESOURCE_PROFILES))
            raise ValueError(f"Unknown resource profile: {resolved!r}. Expected one of: {choices}")
        profile = _NAMED_RESOURCE_PROFILES[key]

    env_temp_dir = _normalize_temp_dir(os.getenv(TEMP_DIR_ENV_VAR))
    if env_temp_dir is not None and profile.temp_dir is None:
        profile = replace(profile, temp_dir=env_temp_dir)

    return normalize_resource_profile(profile)


def resolve_temp_dir(temp_dir: str | Path | None) -> Path | None:
    normalized = _normalize_temp_dir(temp_dir)
    if normalized is None:
        return None
    path = Path(normalized)
    path.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"temp_dir must be a directory: {path}")
    return path


def validate_temp_dir_constraints(
    *,
    temp_dir: str | Path,
    min_free_bytes: int | None,
    context: str,
) -> dict[str, int | str]:
    path = resolve_temp_dir(temp_dir)
    if path is None:  # pragma: no cover - defensive guard; caller passes temp_dir
        raise ValueError(f"{context} requires a concrete temp_dir path")

    min_free = _optional_non_negative_int("temp_dir_min_free_bytes", min_free_bytes)
    usage = shutil.disk_usage(path)
    if min_free is not None and int(usage.free) < int(min_free):
        raise ValueError(
            f"{context} requires at least {int(min_free)} free bytes in temp_dir={path} "
            f"(available={int(usage.free)})"
        )
    return {
        "temp_dir": str(path),
        "temp_dir_free_bytes": int(usage.free),
        "temp_dir_total_bytes": int(usage.total),
    }

