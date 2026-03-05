from __future__ import annotations

from typing import Any

from benchmarks.registry_mines import resolve_preferred_workers


BENCHMARK_PROFILE_SERVER_RESTRICTED = "server_restricted"
BENCHMARK_PROFILE_NON_RESTRICTED = "non_restricted"

_BENCHMARK_PROFILES: dict[str, dict[str, Any]] = {
    BENCHMARK_PROFILE_SERVER_RESTRICTED: {
        "workers": [3, 6, 9],
        "include_legacy_baseline": True,
    },
    BENCHMARK_PROFILE_NON_RESTRICTED: {
        "workers": [4, 8, 12, 16],
        "include_legacy_baseline": True,
    },
}
_BENCHMARK_FALLBACK_PROFILE = BENCHMARK_PROFILE_NON_RESTRICTED
_SERVER_RESTRICTED_WORKER_CEILING = 9
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
        return int(preferred) <= _SERVER_RESTRICTED_WORKER_CEILING
    except Exception:
        return False


def _auto_profile(*, mine_url: str, rows_target: int) -> str:
    if _is_server_restricted(mine_url, rows_target):
        return BENCHMARK_PROFILE_SERVER_RESTRICTED
    return BENCHMARK_PROFILE_NON_RESTRICTED


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
    phase_default_include_legacy: bool,  # kept for compatibility with call sites
    server_restricted: bool | None = None,
) -> dict[str, Any]:
    del phase_default_include_legacy
    workers = _to_int_list(explicit_workers or [])
    if workers:
        return {
            "name": "workers_override",
            "workers": workers,
            "include_legacy_baseline": True,
        }

    profile_token = str(benchmark_profile or "").strip().lower()
    if profile_token and profile_token != "auto":
        profile_name = _normalize_profile_name(profile_token)
    else:
        if server_restricted is None:
            profile_name = _auto_profile(mine_url=mine_url, rows_target=rows_target)
        else:
            profile_name = (
                BENCHMARK_PROFILE_SERVER_RESTRICTED
                if bool(server_restricted)
                else BENCHMARK_PROFILE_NON_RESTRICTED
            )
    profile_data = _BENCHMARK_PROFILES.get(profile_name, _BENCHMARK_PROFILES[_BENCHMARK_FALLBACK_PROFILE])

    return {
        "name": profile_name,
        "workers": list(profile_data.get("workers", [])),
        "include_legacy_baseline": bool(profile_data.get("include_legacy_baseline", True)),
    }
