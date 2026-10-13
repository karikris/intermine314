from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from intermine314.parallel.policy import require_non_negative_int, require_positive_int

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
    cpu_budget: int = 16
    memory_budget_bytes: int | None = None
    arrow_cpu_threads: int | None = None
    arrow_io_threads: int | None = None
    duckdb_threads: int | None = None
    polars_threads: int | None = None
    per_host_limit: int | None = None
    stage: str = "public_http"

    @property
    def effective_cpu_budget(self):
        return min(self.cpu_budget, available_cpus())

    def workers_for(self, stage):
        """Stage allocations share a budget; these are not additive pools."""
        counts = self._worker_counts(self.effective_cpu_budget)
        if stage not in counts:
            raise ValueError(f"Unknown resource stage: {stage!r}")
        return counts[stage]

    def _worker_counts(self, budget):
        small = min(4, (budget + 1) // 2)
        counts = {
            "public_http": min(4, budget), "low_latency_http": min(2, budget),
            "high_latency_http": min(8, budget), "arrow_cpu": small,
            "arrow_io": 1 if budget <= 2 else 2, "ordered_copy": small,
            "unordered_copy": min(8, budget), "duckdb_sql": budget,
            "polars": min(12, budget), "python_processes": small,
            "metadata": min(4, budget), "mutable_owner": 1,
        }
        if self.name == "tor_low_mem":
            counts['ordered_copy'] = 1
        return counts

    @property
    def memory_envelope(self):
        return self.memory_budget_bytes or default_memory_budget()

    @property
    def queue_bytes(self):
        return self.max_inflight_bytes_estimate or max(1, min(64 * 1024**2, self.memory_envelope // 8))


def available_cpus():
    """Respect process affinity and discoverable Linux cgroup quotas."""
    counts = [os.process_cpu_count() or 1]
    try:
        counts.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            counts.append(max(1, (int(quota) + int(period) - 1) // int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        try:
            quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
            period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
            if quota > 0 and period > 0:
                counts.append(max(1, (quota + period - 1) // period))
        except (OSError, ValueError):
            pass
    return max(1, min(counts))


def default_memory_budget():
    available = []
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                available.append(int(line.split()[1]) * 1024)
    except (OSError, ValueError):
        pass
    for limit_path, used_path in [
        ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes'),
    ]:
        try:
            limit = Path(limit_path).read_text().strip()
            if limit != 'max':
                available.append(max(1, int(limit) - int(Path(used_path).read_text())))
        except (OSError, ValueError):
            pass
    return max(1, min(4 * 1024**3, min(available) // 4)) if available else 512 * 1024**2


_NAMED_RESOURCE_PROFILES: dict[str, ResourceProfile] = {
    "default": ResourceProfile(name="cpu16"),
    **{f"cpu{budget}": ResourceProfile(name=f"cpu{budget}", cpu_budget=budget)
       for budget in (16, 8, 6, 4, 2)},
    "tor_low_mem": ResourceProfile(
        name="tor_low_mem",
        max_workers=2,
        ordered="unordered",
        prefetch=2,
        inflight_limit=2,
        max_inflight_bytes_estimate=_TOR_LOW_MEM_MAX_INFLIGHT_BYTES,
        temp_dir="/tmp",
        cpu_budget=2,
    ),
}


def _optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return require_positive_int(name, value)


def _optional_non_negative_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return require_non_negative_int(name, value)


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
        cpu_budget=require_positive_int("cpu_budget", profile.cpu_budget),
        memory_budget_bytes=_optional_positive_int("memory_budget_bytes", profile.memory_budget_bytes),
        arrow_cpu_threads=_optional_positive_int("arrow_cpu_threads", profile.arrow_cpu_threads),
        arrow_io_threads=_optional_positive_int("arrow_io_threads", profile.arrow_io_threads),
        duckdb_threads=_optional_positive_int("duckdb_threads", profile.duckdb_threads),
        polars_threads=_optional_positive_int("polars_threads", profile.polars_threads),
        per_host_limit=_optional_positive_int("per_host_limit", profile.per_host_limit),
        stage=str(profile.stage),
    )
    if normalized.stage not in {"public_http", "low_latency_http", "high_latency_http"}:
        raise ValueError("stage must be public_http, low_latency_http or high_latency_http")
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
