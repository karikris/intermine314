from __future__ import annotations

from typing import Any

from intermine314.registry.mines import resolve_preferred_workers


BENCHMARK_PROFILE_1 = "benchmark_profile_1"
BENCHMARK_PROFILE_2 = "benchmark_profile_2"
BENCHMARK_PROFILE_3 = "benchmark_profile_3"
BENCHMARK_PROFILE_4 = "benchmark_profile_4"

_BENCHMARK_PROFILES: dict[str, dict[str, Any]] = {
    BENCHMARK_PROFILE_1: {
        "include_legacy_baseline": False,
        "workers": [4, 8, 12, 16],
    },
    BENCHMARK_PROFILE_2: {
        "include_legacy_baseline": False,
        "workers": [4, 6, 8],
    },
    BENCHMARK_PROFILE_3: {
        "include_legacy_baseline": True,
        "workers": [4, 8, 12, 16],
    },
    BENCHMARK_PROFILE_4: {
        "include_legacy_baseline": True,
        "workers": [4, 6, 8],
    },
}
_BENCHMARK_FALLBACK_PROFILE = BENCHMARK_PROFILE_3
_BENCHMARK_SWITCH_THRESHOLD_ROWS = 50_000
_SERVER_LIMITED_WORKER_CEILING = 8
_DEFAULT_FALLBACK_WORKERS = 4


def _to_int_list(values: list[int] | tuple[int, ...]) -> list[int]:
    parsed: list[int] = []
    for value in values:
        try:
            number = int(value)
        except Exception:
            continue
        if number <= 0:
            continue
        parsed.append(number)
    return parsed


def _is_server_restricted(mine_url: str, rows_target: int) -> bool:
    preferred = resolve_preferred_workers(mine_url, rows_target, _DEFAULT_FALLBACK_WORKERS)
    try:
        return int(preferred) <= _SERVER_LIMITED_WORKER_CEILING
    except Exception:
        return False


def _auto_profile_for_rows(*, mine_url: str, rows_target: int) -> str:
    server_restricted = _is_server_restricted(mine_url, rows_target)
    small_rows = int(rows_target) <= _BENCHMARK_SWITCH_THRESHOLD_ROWS
    if server_restricted:
        return BENCHMARK_PROFILE_4 if small_rows else BENCHMARK_PROFILE_2
    return BENCHMARK_PROFILE_3 if small_rows else BENCHMARK_PROFILE_1


def _normalize_profile_name(profile_name: str) -> str:
    value = str(profile_name or "").strip().lower()
    if value in _BENCHMARK_PROFILES:
        return value
    return _BENCHMARK_FALLBACK_PROFILE


def resolve_execution_plan(
    *,
    mine_url: str,
    rows_target: int,
    explicit_workers: list[int],
    benchmark_profile: str,
    phase_default_include_legacy: bool,
) -> dict[str, Any]:
    workers = _to_int_list(explicit_workers or [])
    if workers:
        return {
            "name": "workers_override",
            "workers": workers,
            "include_legacy_baseline": bool(phase_default_include_legacy),
        }

    profile_token = str(benchmark_profile or "").strip().lower()
    if profile_token and profile_token != "auto":
        profile_name = _normalize_profile_name(profile_token)
    else:
        profile_name = _auto_profile_for_rows(mine_url=mine_url, rows_target=rows_target)
    profile_data = _BENCHMARK_PROFILES.get(profile_name, _BENCHMARK_PROFILES[_BENCHMARK_FALLBACK_PROFILE])

    return {
        "name": profile_name,
        "workers": list(profile_data.get("workers", [])),
        "include_legacy_baseline": bool(phase_default_include_legacy)
        and bool(profile_data.get("include_legacy_baseline", False)),
    }

