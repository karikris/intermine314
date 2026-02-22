from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from urllib.parse import urlparse

from intermine314.config.constants import (
    DEFAULT_PARALLEL_WORKERS,
    DEFAULT_PRODUCTION_PROFILE_SWITCH_ROWS,
    DEFAULT_WORKERS_TIER,
    FULL_WORKERS_TIER,
    PRODUCTION_PROFILE_ELT_DEFAULT,
    PRODUCTION_PROFILE_ELT_FULL,
    PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
    PRODUCTION_PROFILE_ETL_DEFAULT,
    PRODUCTION_PROFILE_ETL_FULL,
    PRODUCTION_PROFILE_ETL_SERVER_LIMITED,
    PRODUCTION_WORKFLOW_ELT,
    PRODUCTION_WORKFLOW_ETL,
    PRODUCTION_WORKFLOWS,
    SERVER_LIMITED_WORKERS_TIER,
)
from intermine314.config.loader import load_mine_parallel_preferences, resolve_mine_parallel_preferences_path
from intermine314.parallel.policy import (
    VALID_ORDER_MODES,
    VALID_PARALLEL_PROFILES,
    normalize_order_mode,
)
from intermine314.util.logging import log_structured_event


DEFAULT_BENCHMARK_SMALL_PROFILE = "benchmark_profile_3"
DEFAULT_BENCHMARK_LARGE_PROFILE = "benchmark_profile_1"
DEFAULT_BENCHMARK_FALLBACK_PROFILE = DEFAULT_BENCHMARK_SMALL_PROFILE
DEFAULT_PROFILE_SWITCH_ROWS = DEFAULT_PRODUCTION_PROFILE_SWITCH_ROWS
PIPELINE_PARQUET_DUCKDB = "parquet_duckdb"
PIPELINE_POLARS_DUCKDB = "polars_duckdb"
PIPELINE_BY_WORKFLOW = {
    PRODUCTION_WORKFLOW_ELT: PIPELINE_PARQUET_DUCKDB,
    PRODUCTION_WORKFLOW_ETL: PIPELINE_POLARS_DUCKDB,
}
VALID_PIPELINES = frozenset(PIPELINE_BY_WORKFLOW.values())
PRODUCTION_PARALLEL_PROFILE_DEFAULT = "large_query"
PRODUCTION_ORDERED_DEFAULT = "unordered"
PRODUCTION_LARGE_QUERY_MODE_DEFAULT = True


@dataclass(frozen=True)
class _BenchmarkProfileConfig:
    include_legacy_baseline: bool
    workers: tuple[int, ...]

    def as_dict(self):
        return {
            "include_legacy_baseline": bool(self.include_legacy_baseline),
            "workers": list(self.workers),
        }


@dataclass(frozen=True)
class _ProductionProfileConfig:
    workflow: str
    workers: int
    pipeline: str
    parallel_profile: str
    ordered: str
    large_query_mode: bool
    prefetch: int | None
    inflight_limit: int | None

    def as_dict(self):
        return {
            "workflow": self.workflow,
            "workers": int(self.workers),
            "pipeline": self.pipeline,
            "parallel_profile": self.parallel_profile,
            "ordered": self.ordered,
            "large_query_mode": bool(self.large_query_mode),
            "prefetch": self.prefetch,
            "inflight_limit": self.inflight_limit,
        }


@dataclass(frozen=True)
class _MineProfileConfig:
    display_name: str
    host_patterns: tuple[str, ...]
    path_prefixes: tuple[str, ...]
    host_patterns_normalized: tuple[str, ...]
    path_prefixes_normalized: tuple[str, ...]
    default_workers: int
    large_query_threshold_rows: int
    workers_above_threshold: int
    workers_when_size_unknown: int
    production_profile_switch_rows: int
    production_elt_small_profile: str
    production_elt_large_profile: str
    production_etl_small_profile: str
    production_etl_large_profile: str
    benchmark_profile: str
    benchmark_small_profile: str
    benchmark_switch_threshold_rows: int
    benchmark_large_profile: str
    benchmark_small_workers: tuple[int, ...]
    benchmark_small_include_legacy_baseline: bool

    def as_dict(self):
        return {
            "display_name": self.display_name,
            "host_patterns": list(self.host_patterns),
            "path_prefixes": list(self.path_prefixes),
            "host_patterns_normalized": self.host_patterns_normalized,
            "path_prefixes_normalized": self.path_prefixes_normalized,
            "default_workers": int(self.default_workers),
            "large_query_threshold_rows": int(self.large_query_threshold_rows),
            "workers_above_threshold": int(self.workers_above_threshold),
            "workers_when_size_unknown": int(self.workers_when_size_unknown),
            "production_profile_switch_rows": int(self.production_profile_switch_rows),
            "production_elt_small_profile": self.production_elt_small_profile,
            "production_elt_large_profile": self.production_elt_large_profile,
            "production_etl_small_profile": self.production_etl_small_profile,
            "production_etl_large_profile": self.production_etl_large_profile,
            "benchmark_profile": self.benchmark_profile,
            "benchmark_small_profile": self.benchmark_small_profile,
            "benchmark_switch_threshold_rows": int(self.benchmark_switch_threshold_rows),
            "benchmark_large_profile": self.benchmark_large_profile,
            "benchmark_small_workers": list(self.benchmark_small_workers),
            "benchmark_small_include_legacy_baseline": bool(self.benchmark_small_include_legacy_baseline),
        }

DEFAULT_BENCHMARK_PROFILES = {
    DEFAULT_BENCHMARK_LARGE_PROFILE: {
        "include_legacy_baseline": False,
        "workers": [4, 8, 12, 16],
    },
    "benchmark_profile_2": {
        "include_legacy_baseline": False,
        "workers": [4, 6, 8],
    },
    "benchmark_profile_3": {
        "include_legacy_baseline": True,
        "workers": [4, 8, 12, 16],
    },
    "benchmark_profile_4": {
        "include_legacy_baseline": True,
        "workers": [4, 6, 8],
    },
}


def _default_production_profile(*, workflow, workers):
    return {
        "workflow": workflow,
        "workers": int(workers),
        "pipeline": PIPELINE_BY_WORKFLOW[workflow],
        "parallel_profile": PRODUCTION_PARALLEL_PROFILE_DEFAULT,
        "ordered": PRODUCTION_ORDERED_DEFAULT,
        "large_query_mode": PRODUCTION_LARGE_QUERY_MODE_DEFAULT,
    }


DEFAULT_PRODUCTION_PROFILES = {
    PRODUCTION_PROFILE_ELT_DEFAULT: _default_production_profile(
        workflow=PRODUCTION_WORKFLOW_ELT,
        workers=DEFAULT_WORKERS_TIER,
    ),
    PRODUCTION_PROFILE_ELT_SERVER_LIMITED: _default_production_profile(
        workflow=PRODUCTION_WORKFLOW_ELT,
        workers=SERVER_LIMITED_WORKERS_TIER,
    ),
    PRODUCTION_PROFILE_ELT_FULL: _default_production_profile(
        workflow=PRODUCTION_WORKFLOW_ELT,
        workers=FULL_WORKERS_TIER,
    ),
    PRODUCTION_PROFILE_ETL_DEFAULT: _default_production_profile(
        workflow=PRODUCTION_WORKFLOW_ETL,
        workers=DEFAULT_WORKERS_TIER,
    ),
    PRODUCTION_PROFILE_ETL_SERVER_LIMITED: _default_production_profile(
        workflow=PRODUCTION_WORKFLOW_ETL,
        workers=SERVER_LIMITED_WORKERS_TIER,
    ),
    PRODUCTION_PROFILE_ETL_FULL: _default_production_profile(
        workflow=PRODUCTION_WORKFLOW_ETL,
        workers=FULL_WORKERS_TIER,
    ),
}

DEFAULT_PRODUCTION_PROFILE_BY_WORKFLOW = {
    PRODUCTION_WORKFLOW_ELT: PRODUCTION_PROFILE_ELT_DEFAULT,
    PRODUCTION_WORKFLOW_ETL: PRODUCTION_PROFILE_ETL_DEFAULT,
}

FULL_PRODUCTION_PROFILE_BY_WORKFLOW = {
    PRODUCTION_WORKFLOW_ELT: PRODUCTION_PROFILE_ELT_FULL,
    PRODUCTION_WORKFLOW_ETL: PRODUCTION_PROFILE_ETL_FULL,
}

SERVER_LIMITED_PROFILE_BY_WORKFLOW = {
    PRODUCTION_WORKFLOW_ELT: PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
    PRODUCTION_WORKFLOW_ETL: PRODUCTION_PROFILE_ETL_SERVER_LIMITED,
}


def _to_int_list(values):
    result = []
    for value in values:
        try:
            ivalue = int(value)
        except Exception:
            continue
        if ivalue <= 0:
            continue
        result.append(ivalue)
    return result


def _decode_positive_int(value, *, path, default=None):
    raw = default if value is None else value
    try:
        parsed = int(raw)
    except Exception:
        _record_invalid_config("positive_int", path=path, value=raw)
        raise ValueError(f"Invalid integer at {path}: {raw!r}")
    if parsed <= 0:
        _record_invalid_config("positive_int", path=path, value=raw)
        raise ValueError(f"Expected positive integer at {path}: {raw!r}")
    return parsed


def _decode_optional_positive_int(value, *, path):
    if value is None:
        return None
    return _decode_positive_int(value, path=path)


def _decode_size(size, *, path):
    if size is None:
        return None
    try:
        return int(size)
    except Exception:
        raise ValueError(f"Invalid size at {path}: {size!r}")


def _decode_string(value, *, path, default=None):
    raw = default if value is None else value
    text = str(raw or "").strip()
    if not text:
        _record_invalid_config("string", path=path, value=raw)
        raise ValueError(f"Expected non-empty string at {path}")
    return text


def _decode_string_tuple(values, *, path):
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        _record_invalid_config("string_list", path=path, value=values)
        raise ValueError(f"Expected iterable of strings at {path}")
    try:
        iterator = iter(values)
    except Exception:
        _record_invalid_config("string_list", path=path, value=values)
        raise ValueError(f"Expected iterable of strings at {path}")
    return tuple(str(value) for value in iterator)


def _decode_workers_tuple(values, *, path):
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        _record_invalid_config("workers", path=path, value=values)
        raise ValueError(f"Expected iterable of positive integers at {path}")
    try:
        iterator = iter(values)
    except Exception:
        _record_invalid_config("workers", path=path, value=values)
        raise ValueError(f"Expected iterable of positive integers at {path}")
    workers = []
    for idx, value in enumerate(iterator):
        workers.append(_decode_positive_int(value, path=f"{path}[{idx}]"))
    return tuple(workers)


def _normalize_workflow(workflow):
    value = str(workflow or PRODUCTION_WORKFLOW_ELT).strip().lower()
    if value not in PRODUCTION_WORKFLOWS:
        choices = ", ".join(PRODUCTION_WORKFLOWS)
        raise ValueError(f"workflow must be one of: {choices}")
    return value


def _workers_to_production_profile(workflow, workers):
    workflow = _normalize_workflow(workflow)
    if int(workers) <= DEFAULT_WORKERS_TIER:
        return DEFAULT_PRODUCTION_PROFILE_BY_WORKFLOW[workflow]
    if int(workers) <= SERVER_LIMITED_WORKERS_TIER:
        return SERVER_LIMITED_PROFILE_BY_WORKFLOW[workflow]
    return FULL_PRODUCTION_PROFILE_BY_WORKFLOW[workflow]


def _standard_mine_profile(
    *,
    display_name,
    host_patterns,
    path_prefixes,
    default_workers=FULL_WORKERS_TIER,
    production_large_workers=FULL_WORKERS_TIER,
    production_switch_rows=DEFAULT_PROFILE_SWITCH_ROWS,
    benchmark_small_profile=DEFAULT_BENCHMARK_SMALL_PROFILE,
    benchmark_large_profile=DEFAULT_BENCHMARK_LARGE_PROFILE,
    benchmark_switch_rows=DEFAULT_PROFILE_SWITCH_ROWS,
    production_elt_small_profile=None,
    production_elt_large_profile=None,
    production_etl_small_profile=None,
    production_etl_large_profile=None,
):
    default_workers = int(default_workers)
    production_large_workers = int(production_large_workers)
    if production_elt_small_profile is None:
        production_elt_small_profile = _workers_to_production_profile(PRODUCTION_WORKFLOW_ELT, default_workers)
    if production_elt_large_profile is None:
        production_elt_large_profile = _workers_to_production_profile(PRODUCTION_WORKFLOW_ELT, production_large_workers)
    if production_etl_small_profile is None:
        production_etl_small_profile = _workers_to_production_profile(PRODUCTION_WORKFLOW_ETL, default_workers)
    if production_etl_large_profile is None:
        production_etl_large_profile = _workers_to_production_profile(PRODUCTION_WORKFLOW_ETL, production_large_workers)

    return {
        "display_name": display_name,
        "host_patterns": host_patterns,
        "path_prefixes": path_prefixes,
        # Legacy worker-resolution fields still used by existing query defaults.
        "default_workers": default_workers,
        "large_query_threshold_rows": int(production_switch_rows),
        "workers_above_threshold": production_large_workers,
        "workers_when_size_unknown": production_large_workers,
        # Production profile mapping (ELT + ETL).
        "production_profile_switch_rows": int(production_switch_rows),
        "production_elt_small_profile": str(production_elt_small_profile),
        "production_elt_large_profile": str(production_elt_large_profile),
        "production_etl_small_profile": str(production_etl_small_profile),
        "production_etl_large_profile": str(production_etl_large_profile),
        # Benchmark profile mapping.
        "benchmark_profile": benchmark_small_profile,
        "benchmark_small_profile": benchmark_small_profile,
        "benchmark_switch_threshold_rows": int(benchmark_switch_rows),
        "benchmark_large_profile": benchmark_large_profile,
    }


DEFAULT_REGISTRY = {
    "legumemine": {
        "display_name": "LegumeMine",
        "host_patterns": ["mines.legumeinfo.org"],
        "path_prefixes": ["/legumemine"],
        "default_workers": DEFAULT_WORKERS_TIER,
        "large_query_threshold_rows": DEFAULT_PROFILE_SWITCH_ROWS,
        "workers_above_threshold": DEFAULT_WORKERS_TIER,
        "workers_when_size_unknown": DEFAULT_WORKERS_TIER,
        "production_profile_switch_rows": DEFAULT_PROFILE_SWITCH_ROWS,
        "production_elt_small_profile": PRODUCTION_PROFILE_ELT_DEFAULT,
        "production_elt_large_profile": PRODUCTION_PROFILE_ELT_DEFAULT,
        "production_etl_small_profile": PRODUCTION_PROFILE_ETL_DEFAULT,
        "production_etl_large_profile": PRODUCTION_PROFILE_ETL_DEFAULT,
        "benchmark_profile": DEFAULT_BENCHMARK_SMALL_PROFILE,
        "benchmark_small_profile": DEFAULT_BENCHMARK_SMALL_PROFILE,
        "benchmark_switch_threshold_rows": DEFAULT_PROFILE_SWITCH_ROWS,
        "benchmark_small_workers": [],
        "benchmark_small_include_legacy_baseline": False,
        "benchmark_large_profile": "benchmark_profile_2",
    },
    "maizemine": _standard_mine_profile(
        display_name="MaizeMine",
        host_patterns=["maizemine.rnet.missouri.edu"],
        path_prefixes=["/maizemine"],
        default_workers=SERVER_LIMITED_WORKERS_TIER,
        production_large_workers=SERVER_LIMITED_WORKERS_TIER,
        benchmark_small_profile="benchmark_profile_4",
        benchmark_large_profile="benchmark_profile_2",
        production_elt_small_profile=PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
        production_elt_large_profile=PRODUCTION_PROFILE_ELT_SERVER_LIMITED,
        production_etl_small_profile=PRODUCTION_PROFILE_ETL_SERVER_LIMITED,
        production_etl_large_profile=PRODUCTION_PROFILE_ETL_SERVER_LIMITED,
    ),
    "thalemine": _standard_mine_profile(
        display_name="ThaleMine",
        host_patterns=["bar.utoronto.ca"],
        path_prefixes=["/thalemine"],
    ),
    "oakmine": _standard_mine_profile(
        display_name="OakMine",
        host_patterns=["urgi.versailles.inra.fr", "urgi.versailles.inrae.fr"],
        path_prefixes=["/OakMine_PM1N"],
    ),
    "wheatmine": _standard_mine_profile(
        display_name="WheatMine",
        host_patterns=["urgi.versailles.inra.fr", "urgi.versailles.inrae.fr"],
        path_prefixes=["/WheatMine"],
    ),
}

_CACHE = None
_REGISTRY_MINES_LOG = logging.getLogger("intermine314.registry.mines")
_INVALID_CONFIG_ATTEMPTS = 0
_INVALID_CONFIG_ATTEMPTS_BY_FIELD = {}


def _log_registry_mines_event(event, **fields):
    if not _REGISTRY_MINES_LOG.isEnabledFor(logging.DEBUG):
        return
    log_structured_event(_REGISTRY_MINES_LOG, logging.DEBUG, event, **fields)


def _log_registry_mines_warning(event, **fields):
    if not _REGISTRY_MINES_LOG.isEnabledFor(logging.WARNING):
        return
    log_structured_event(_REGISTRY_MINES_LOG, logging.WARNING, event, **fields)


def _record_invalid_config(kind, *, profile_name=None, path=None, value=None):
    global _INVALID_CONFIG_ATTEMPTS
    _INVALID_CONFIG_ATTEMPTS += 1
    field_key = str(path or kind)
    _INVALID_CONFIG_ATTEMPTS_BY_FIELD[field_key] = int(_INVALID_CONFIG_ATTEMPTS_BY_FIELD.get(field_key, 0)) + 1
    _log_registry_mines_warning(
        "registry_preferences_invalid_config",
        kind=str(kind),
        profile_name=str(profile_name),
        path=path,
        value=str(value),
        invalid_config_attempts=int(_INVALID_CONFIG_ATTEMPTS),
        invalid_config_attempts_by_field=dict(_INVALID_CONFIG_ATTEMPTS_BY_FIELD),
    )


def _as_mapping(value):
    if isinstance(value, Mapping):
        return value
    return {}


def _overlay_named_profiles(base_profiles, raw_profiles):
    raw = _as_mapping(raw_profiles)
    merged = None
    for name, profile in raw.items():
        if not isinstance(profile, Mapping):
            continue
        # Copy-on-write: keep base mapping untouched unless an override is present.
        base_profile = base_profiles.get(name)
        if merged is None:
            merged = dict(base_profiles)
        if isinstance(base_profile, Mapping):
            merged[name] = {**base_profile, **profile}
        else:
            merged[name] = dict(profile)
    return base_profiles if merged is None else merged


def _merge_mines(loaded):
    root = _as_mapping(loaded)
    defaults_block = _as_mapping(root.get("defaults"))
    mine_defaults = _as_mapping(defaults_block.get("mine"))
    raw_mines = _as_mapping(root.get("mines"))

    if not raw_mines and not mine_defaults:
        return DEFAULT_REGISTRY

    merged = dict(DEFAULT_REGISTRY)

    for name, profile in raw_mines.items():
        if not isinstance(profile, Mapping):
            continue
        base = DEFAULT_REGISTRY.get(name, {})
        if not isinstance(base, Mapping):
            base = {}
        if mine_defaults:
            merged[name] = {**base, **mine_defaults, **profile}
        else:
            merged[name] = {**base, **profile}

    if mine_defaults:
        for name, profile in DEFAULT_REGISTRY.items():
            if name not in raw_mines:
                merged[name] = {**profile, **mine_defaults}
    return merged


def _merge_benchmark_profiles(loaded):
    root = _as_mapping(loaded)
    return _overlay_named_profiles(DEFAULT_BENCHMARK_PROFILES, root.get("benchmark_profiles"))


def _default_pipeline_for_workflow(workflow):
    workflow_key = _normalize_workflow(workflow)
    return PIPELINE_BY_WORKFLOW[workflow_key]


def _normalize_pipeline_value(value, *, workflow, profile_name, path):
    default_pipeline = _default_pipeline_for_workflow(workflow)
    pipeline = str(value or default_pipeline).strip().lower()
    if pipeline not in VALID_PIPELINES:
        _record_invalid_config("pipeline", profile_name=profile_name, path=path, value=pipeline)
        choices = ", ".join(sorted(VALID_PIPELINES))
        raise ValueError(
            f"Invalid pipeline at {path}: {pipeline!r}. "
            f"Expected one of: {choices}"
        )
    if pipeline != default_pipeline:
        _record_invalid_config("pipeline_workflow_mismatch", profile_name=profile_name, path=path, value=pipeline)
        raise ValueError(
            f"Invalid pipeline at {path}: {pipeline!r} for workflow {workflow!r}. "
            f"Expected {default_pipeline!r}."
        )
    return pipeline


def _normalize_parallel_profile_value(value, *, profile_name, path):
    parallel_profile = str(value or PRODUCTION_PARALLEL_PROFILE_DEFAULT).strip().lower()
    if parallel_profile not in VALID_PARALLEL_PROFILES:
        _record_invalid_config("parallel_profile", profile_name=profile_name, path=path, value=parallel_profile)
        choices = ", ".join(sorted(VALID_PARALLEL_PROFILES))
        raise ValueError(
            f"Invalid parallel_profile at {path}: {parallel_profile!r}. "
            f"Expected one of: {choices}"
        )
    return parallel_profile


def _normalize_ordered_value(value, *, profile_name, path):
    try:
        return normalize_order_mode(
            value,
            default_order_mode=PRODUCTION_ORDERED_DEFAULT,
            valid_order_modes=VALID_ORDER_MODES,
        )
    except Exception:
        _record_invalid_config("ordered", profile_name=profile_name, path=path, value=value)
        choices = ", ".join(sorted(VALID_ORDER_MODES))
        raise ValueError(
            f"Invalid ordered mode at {path}: {value!r}. "
            f"Expected bool or one of: {choices}"
        )


def _normalize_production_profile_entry(profile_name, profile):
    base_path = f"production_profiles.{profile_name}"
    workflow = _normalize_workflow(profile.get("workflow", PRODUCTION_WORKFLOW_ELT))
    workers_default = DEFAULT_PRODUCTION_PROFILES.get(profile_name, {}).get("workers", DEFAULT_WORKERS_TIER)
    cfg = _ProductionProfileConfig(
        workflow=workflow,
        workers=_decode_positive_int(
            profile.get("workers"),
            path=f"{base_path}.workers",
            default=workers_default,
        ),
        pipeline=_normalize_pipeline_value(
            profile.get("pipeline"),
            workflow=workflow,
            profile_name=profile_name,
            path=f"{base_path}.pipeline",
        ),
        parallel_profile=_normalize_parallel_profile_value(
            profile.get("parallel_profile"),
            profile_name=profile_name,
            path=f"{base_path}.parallel_profile",
        ),
        ordered=_normalize_ordered_value(
            profile.get("ordered"),
            profile_name=profile_name,
            path=f"{base_path}.ordered",
        ),
        large_query_mode=bool(profile.get("large_query_mode", PRODUCTION_LARGE_QUERY_MODE_DEFAULT)),
        prefetch=_decode_optional_positive_int(profile.get("prefetch"), path=f"{base_path}.prefetch"),
        inflight_limit=_decode_optional_positive_int(
            profile.get("inflight_limit"),
            path=f"{base_path}.inflight_limit",
        ),
    )
    return cfg.as_dict()


def _value_distribution(mapping, field):
    counts = {}
    for profile in mapping.values():
        value = profile.get(field)
        key = str(value)
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _normalize_production_profiles(profiles):
    normalized = {}
    for name, profile in profiles.items():
        if not isinstance(profile, Mapping):
            continue
        normalized[name] = _normalize_production_profile_entry(str(name), profile)
    return normalized


def _normalize_benchmark_profile_entry(profile_name, profile):
    base_path = f"benchmark_profiles.{profile_name}"
    workers_default = DEFAULT_BENCHMARK_PROFILES.get(
        profile_name,
        DEFAULT_BENCHMARK_PROFILES[DEFAULT_BENCHMARK_FALLBACK_PROFILE],
    ).get("workers", ())
    workers_value = profile.get("workers", workers_default)
    workers = _decode_workers_tuple(workers_value, path=f"{base_path}.workers")
    cfg = _BenchmarkProfileConfig(
        include_legacy_baseline=bool(profile.get("include_legacy_baseline", False)),
        workers=workers,
    )
    return cfg.as_dict()


def _normalize_benchmark_profiles(profiles):
    normalized = {}
    for name, profile in profiles.items():
        if not isinstance(profile, Mapping):
            continue
        normalized[name] = _normalize_benchmark_profile_entry(str(name), profile)
    return normalized


def _normalize_mine_profile_entry(profile_name, profile):
    base_path = f"mines.{profile_name}"
    host_patterns = _decode_string_tuple(profile.get("host_patterns"), path=f"{base_path}.host_patterns")
    path_prefixes = _decode_string_tuple(profile.get("path_prefixes"), path=f"{base_path}.path_prefixes")
    default_workers = _decode_positive_int(
        profile.get("default_workers"),
        path=f"{base_path}.default_workers",
        default=DEFAULT_PARALLEL_WORKERS,
    )
    workers_above_threshold = _decode_positive_int(
        profile.get("workers_above_threshold"),
        path=f"{base_path}.workers_above_threshold",
        default=default_workers,
    )
    workers_when_size_unknown = _decode_positive_int(
        profile.get("workers_when_size_unknown"),
        path=f"{base_path}.workers_when_size_unknown",
        default=workers_above_threshold,
    )
    large_query_threshold_rows = _decode_positive_int(
        profile.get("large_query_threshold_rows"),
        path=f"{base_path}.large_query_threshold_rows",
        default=DEFAULT_PROFILE_SWITCH_ROWS,
    )
    production_profile_switch_rows = _decode_positive_int(
        profile.get("production_profile_switch_rows"),
        path=f"{base_path}.production_profile_switch_rows",
        default=large_query_threshold_rows,
    )
    benchmark_switch_threshold_rows = _decode_positive_int(
        profile.get("benchmark_switch_threshold_rows"),
        path=f"{base_path}.benchmark_switch_threshold_rows",
        default=DEFAULT_PROFILE_SWITCH_ROWS,
    )
    cfg = _MineProfileConfig(
        display_name=_decode_string(profile.get("display_name"), path=f"{base_path}.display_name", default=profile_name),
        host_patterns=host_patterns,
        path_prefixes=path_prefixes,
        host_patterns_normalized=_normalize_match_values(host_patterns),
        path_prefixes_normalized=_normalize_match_values(path_prefixes),
        default_workers=default_workers,
        large_query_threshold_rows=large_query_threshold_rows,
        workers_above_threshold=workers_above_threshold,
        workers_when_size_unknown=workers_when_size_unknown,
        production_profile_switch_rows=production_profile_switch_rows,
        production_elt_small_profile=_decode_string(
            profile.get("production_elt_small_profile"),
            path=f"{base_path}.production_elt_small_profile",
            default=_workers_to_production_profile(PRODUCTION_WORKFLOW_ELT, default_workers),
        ),
        production_elt_large_profile=_decode_string(
            profile.get("production_elt_large_profile"),
            path=f"{base_path}.production_elt_large_profile",
            default=_workers_to_production_profile(PRODUCTION_WORKFLOW_ELT, workers_above_threshold),
        ),
        production_etl_small_profile=_decode_string(
            profile.get("production_etl_small_profile"),
            path=f"{base_path}.production_etl_small_profile",
            default=_workers_to_production_profile(PRODUCTION_WORKFLOW_ETL, default_workers),
        ),
        production_etl_large_profile=_decode_string(
            profile.get("production_etl_large_profile"),
            path=f"{base_path}.production_etl_large_profile",
            default=_workers_to_production_profile(PRODUCTION_WORKFLOW_ETL, workers_above_threshold),
        ),
        benchmark_profile=_decode_string(
            profile.get("benchmark_profile"),
            path=f"{base_path}.benchmark_profile",
            default=DEFAULT_BENCHMARK_FALLBACK_PROFILE,
        ),
        benchmark_small_profile=_decode_string(
            profile.get("benchmark_small_profile"),
            path=f"{base_path}.benchmark_small_profile",
            default=profile.get("benchmark_profile", DEFAULT_BENCHMARK_FALLBACK_PROFILE),
        ),
        benchmark_switch_threshold_rows=benchmark_switch_threshold_rows,
        benchmark_large_profile=_decode_string(
            profile.get("benchmark_large_profile"),
            path=f"{base_path}.benchmark_large_profile",
            default=DEFAULT_BENCHMARK_LARGE_PROFILE,
        ),
        benchmark_small_workers=_decode_workers_tuple(
            profile.get("benchmark_small_workers", ()),
            path=f"{base_path}.benchmark_small_workers",
        ),
        benchmark_small_include_legacy_baseline=bool(
            profile.get("benchmark_small_include_legacy_baseline", False)
        ),
    )
    return cfg.as_dict()


def _merge_production_profiles(loaded):
    root = _as_mapping(loaded)
    return _overlay_named_profiles(DEFAULT_PRODUCTION_PROFILES, root.get("production_profiles"))


def _normalize_match_values(values):
    try:
        iterator = iter(values)
    except Exception:
        return ()
    return tuple(str(value).lower() for value in iterator)


def _normalize_mines_for_matching(mines):
    normalized = {}
    for name, profile in mines.items():
        if not isinstance(profile, Mapping):
            continue
        normalized[name] = _normalize_mine_profile_entry(str(name), profile)
    return normalized


def _load_registry():
    global _CACHE
    if _CACHE is not None:
        normalization_metrics = _CACHE.get("_normalization_metrics", {})
        _log_registry_mines_event(
            "registry_preferences_cache_hit",
            cache_populated=True,
            mine_count=len(_CACHE.get("mines", {})),
            benchmark_profile_count=len(_CACHE.get("benchmark_profiles", {})),
            production_profile_count=len(_CACHE.get("production_profiles", {})),
            config_source=_CACHE.get("_config_source"),
            config_path=_CACHE.get("_config_path"),
            invalid_config_attempts=int(_INVALID_CONFIG_ATTEMPTS),
            invalid_config_attempts_by_field=dict(_INVALID_CONFIG_ATTEMPTS_BY_FIELD),
            pipeline_distribution=normalization_metrics.get("pipeline_distribution"),
            parallel_profile_distribution=normalization_metrics.get("parallel_profile_distribution"),
            production_workers_distribution=normalization_metrics.get("production_workers_distribution"),
            mine_default_workers_distribution=normalization_metrics.get("mine_default_workers_distribution"),
            mine_threshold_rows_distribution=normalization_metrics.get("mine_threshold_rows_distribution"),
        )
        return _CACHE

    merged_mines = DEFAULT_REGISTRY
    merged_benchmark_profiles = DEFAULT_BENCHMARK_PROFILES
    merged_production_profiles = DEFAULT_PRODUCTION_PROFILES
    config_source = "defaults"
    config_path = None
    loaded = load_mine_parallel_preferences()
    try:
        config_path = str(resolve_mine_parallel_preferences_path())
    except Exception:
        config_path = None
    if isinstance(loaded, Mapping):
        config_source = "mine_parallel_preferences_toml"
        merged_mines = _merge_mines(loaded)
        merged_benchmark_profiles = _merge_benchmark_profiles(loaded)
        merged_production_profiles = _merge_production_profiles(loaded)
    data = {
        "benchmark_profiles": _normalize_benchmark_profiles(merged_benchmark_profiles),
        "production_profiles": _normalize_production_profiles(merged_production_profiles),
        "mines": _normalize_mines_for_matching(merged_mines),
    }
    data["_normalization_metrics"] = {
        "pipeline_distribution": _value_distribution(data.get("production_profiles", {}), "pipeline"),
        "parallel_profile_distribution": _value_distribution(data.get("production_profiles", {}), "parallel_profile"),
        "production_workers_distribution": _value_distribution(data.get("production_profiles", {}), "workers"),
        "mine_default_workers_distribution": _value_distribution(data.get("mines", {}), "default_workers"),
        "mine_threshold_rows_distribution": _value_distribution(
            data.get("mines", {}),
            "production_profile_switch_rows",
        ),
    }
    data["_config_source"] = config_source
    data["_config_path"] = config_path
    _CACHE = data
    _log_registry_mines_event(
        "registry_preferences_cache_build",
        cache_populated=True,
        mine_count=len(data.get("mines", {})),
        benchmark_profile_count=len(data.get("benchmark_profiles", {})),
        production_profile_count=len(data.get("production_profiles", {})),
        config_source=config_source,
        config_path=config_path,
        invalid_config_attempts=int(_INVALID_CONFIG_ATTEMPTS),
        invalid_config_attempts_by_field=dict(_INVALID_CONFIG_ATTEMPTS_BY_FIELD),
        pipeline_distribution=data["_normalization_metrics"]["pipeline_distribution"],
        parallel_profile_distribution=data["_normalization_metrics"]["parallel_profile_distribution"],
        production_workers_distribution=data["_normalization_metrics"]["production_workers_distribution"],
        mine_default_workers_distribution=data["_normalization_metrics"]["mine_default_workers_distribution"],
        mine_threshold_rows_distribution=data["_normalization_metrics"]["mine_threshold_rows_distribution"],
    )
    return _CACHE


def _normalize_service_root(service_root):
    if not service_root:
        return "", ""
    parsed = urlparse(service_root)
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/service"):
        path = path[: -len("/service")]
    return host, path


def _matches_profile(profile, host, path):
    host_patterns = profile.get("host_patterns_normalized")
    if host_patterns is None:
        host_patterns = _normalize_match_values(profile.get("host_patterns", []))
    if host_patterns and host not in host_patterns:
        return False
    prefixes = profile.get("path_prefixes_normalized")
    if prefixes is None:
        prefixes = _normalize_match_values(profile.get("path_prefixes", []))
    if not prefixes:
        return True
    path_lower = path.lower()
    return any(path_lower.startswith(prefix) for prefix in prefixes)


def _match_mine_profile(service_root, *, mines=None):
    host, path = _normalize_service_root(service_root)
    if not host:
        return None

    mine_profiles = mines
    if not isinstance(mine_profiles, Mapping):
        mine_profiles = _load_registry().get("mines", {})
    for profile in mine_profiles.values():
        if _matches_profile(profile, host, path):
            return profile
    return None


def _normalize_benchmark_profile(name, profiles, fallback_name):
    if not profiles:
        return fallback_name
    if name in profiles:
        return name
    if fallback_name in profiles:
        return fallback_name
    return next(iter(profiles))


def _profile_to_plan(profile_name, profile_data):
    workers = list(profile_data.get("workers", ()))
    if not workers:
        workers = list(DEFAULT_BENCHMARK_PROFILES[DEFAULT_BENCHMARK_FALLBACK_PROFILE]["workers"])
    include_legacy = bool(profile_data.get("include_legacy_baseline", False))
    return {
        "name": profile_name,
        "workers": workers,
        "include_legacy_baseline": include_legacy,
    }


def _normalize_production_profile(name, profiles, workflow):
    workflow = _normalize_workflow(workflow)
    fallback = DEFAULT_PRODUCTION_PROFILE_BY_WORKFLOW[workflow]
    if not profiles:
        return fallback
    if name in profiles:
        return name
    if fallback in profiles:
        return fallback
    return next(iter(profiles))


def _production_profile_to_plan(profile_name, profile_data, workflow):
    workflow = _normalize_workflow(workflow)
    default_workers = DEFAULT_WORKERS_TIER
    if profile_name in DEFAULT_PRODUCTION_PROFILES:
        default_workers = int(DEFAULT_PRODUCTION_PROFILES[profile_name].get("workers", DEFAULT_WORKERS_TIER))

    workers = int(profile_data.get("workers", default_workers))

    plan_workflow = _normalize_workflow(profile_data.get("workflow", workflow))
    if plan_workflow != workflow:
        plan_workflow = workflow
    default_pipeline = _default_pipeline_for_workflow(plan_workflow)
    pipeline = _normalize_pipeline_value(
        profile_data.get("pipeline", default_pipeline),
        workflow=plan_workflow,
        profile_name=profile_name,
        path=f"production_profiles.{profile_name}.pipeline",
    )
    parallel_profile = _normalize_parallel_profile_value(
        profile_data.get("parallel_profile", PRODUCTION_PARALLEL_PROFILE_DEFAULT),
        profile_name=profile_name,
        path=f"production_profiles.{profile_name}.parallel_profile",
    )
    ordered = _normalize_ordered_value(
        profile_data.get("ordered", PRODUCTION_ORDERED_DEFAULT),
        profile_name=profile_name,
        path=f"production_profiles.{profile_name}.ordered",
    )

    return {
        "name": profile_name,
        "workflow": plan_workflow,
        "workers": workers,
        "pipeline": pipeline,
        "parallel_profile": parallel_profile,
        "ordered": ordered,
        "large_query_mode": bool(profile_data.get("large_query_mode", PRODUCTION_LARGE_QUERY_MODE_DEFAULT)),
        "prefetch": profile_data.get("prefetch"),
        "inflight_limit": profile_data.get("inflight_limit"),
    }


def _mine_profile_name_for_workflow(mine_profile, *, size, workflow, fallback_profile):
    workflow = _normalize_workflow(workflow)
    threshold = int(
        mine_profile.get(
            "production_profile_switch_rows",
            mine_profile.get("large_query_threshold_rows", DEFAULT_PROFILE_SWITCH_ROWS),
        )
    )

    small_key = f"production_{workflow}_small_profile"
    large_key = f"production_{workflow}_large_profile"
    small_profile = mine_profile.get(small_key)
    large_profile = mine_profile.get(large_key)

    if not small_profile:
        small_profile = _workers_to_production_profile(workflow, mine_profile.get("default_workers", DEFAULT_PARALLEL_WORKERS))
    if not large_profile:
        large_workers = mine_profile.get(
            "workers_above_threshold",
            mine_profile.get("default_workers", DEFAULT_PARALLEL_WORKERS),
        )
        large_profile = _workers_to_production_profile(workflow, large_workers)

    if size is None:
        unknown_workers = mine_profile.get(
            "workers_when_size_unknown",
            mine_profile.get(
                "workers_above_threshold",
                mine_profile.get("default_workers", DEFAULT_PARALLEL_WORKERS),
            ),
        )
        return _workers_to_production_profile(workflow, unknown_workers)

    if size > threshold:
        return str(large_profile or fallback_profile)
    return str(small_profile or fallback_profile)


def _resolve_production_plan_from_context(
    *,
    profiles,
    size,
    workflow,
    production_profile,
    mine_profile,
):
    if production_profile is not None and str(production_profile).strip().lower() != "auto":
        resolved_name = _normalize_production_profile(str(production_profile), profiles, workflow)
        plan = _production_profile_to_plan(resolved_name, profiles[resolved_name], workflow)
        _log_registry_mines_event(
            "registry_production_plan_resolved",
            workflow=workflow,
            mode="explicit_profile",
            matched_mine=False,
            size=size,
            resolved_profile=resolved_name,
            workers=plan.get("workers"),
        )
        return plan

    fallback_name = DEFAULT_PRODUCTION_PROFILE_BY_WORKFLOW[workflow]
    if mine_profile is None:
        resolved_name = _normalize_production_profile(fallback_name, profiles, workflow)
        plan = _production_profile_to_plan(resolved_name, profiles[resolved_name], workflow)
        _log_registry_mines_event(
            "registry_production_plan_resolved",
            workflow=workflow,
            mode="workflow_fallback",
            matched_mine=False,
            size=size,
            resolved_profile=resolved_name,
            workers=plan.get("workers"),
        )
        return plan

    profile_name = _mine_profile_name_for_workflow(
        mine_profile,
        size=size,
        workflow=workflow,
        fallback_profile=fallback_name,
    )
    resolved_name = _normalize_production_profile(profile_name, profiles, workflow)
    plan = _production_profile_to_plan(resolved_name, profiles[resolved_name], workflow)
    _log_registry_mines_event(
        "registry_production_plan_resolved",
        workflow=workflow,
        mode="mine_auto",
        matched_mine=True,
        size=size,
        resolved_profile=resolved_name,
        workers=plan.get("workers"),
    )
    return plan


def resolve_production_plan(
    service_root,
    size,
    *,
    workflow=PRODUCTION_WORKFLOW_ELT,
    production_profile="auto",
):
    workflow = _normalize_workflow(workflow)
    size = _decode_size(size, path="resolve_production_plan.size")
    registry = _load_registry()
    profiles = registry.get("production_profiles", {}) or DEFAULT_PRODUCTION_PROFILES
    mine_profile = _match_mine_profile(service_root, mines=registry.get("mines", {}))
    return _resolve_production_plan_from_context(
        profiles=profiles,
        size=size,
        workflow=workflow,
        production_profile=production_profile,
        mine_profile=mine_profile,
    )


def resolve_preferred_workers(service_root, size, fallback_workers):
    size = _decode_size(size, path="resolve_preferred_workers.size")
    registry = _load_registry()
    mine_profile = _match_mine_profile(service_root, mines=registry.get("mines", {}))
    if mine_profile is None:
        return fallback_workers
    profiles = registry.get("production_profiles", {}) or DEFAULT_PRODUCTION_PROFILES
    plan = _resolve_production_plan_from_context(
        profiles=profiles,
        size=size,
        workflow=PRODUCTION_WORKFLOW_ELT,
        production_profile="auto",
        mine_profile=mine_profile,
    )
    try:
        return int(plan["workers"])
    except Exception:
        return fallback_workers


def _resolve_benchmark_plan_from_context(
    *,
    profiles,
    size,
    fallback_profile,
    mine_profile,
):
    fallback_name = _normalize_benchmark_profile(fallback_profile, profiles, DEFAULT_BENCHMARK_FALLBACK_PROFILE)
    if mine_profile is None:
        plan = _profile_to_plan(fallback_name, profiles[fallback_name])
        _log_registry_mines_event(
            "registry_benchmark_plan_resolved",
            mode="profile_fallback",
            matched_mine=False,
            size=size,
            resolved_profile=plan.get("name"),
            workers=plan.get("workers"),
        )
        return plan

    threshold = mine_profile.get("benchmark_switch_threshold_rows")
    if (
        threshold is not None
        and size is not None
        and size <= threshold
        and isinstance(mine_profile.get("benchmark_small_workers"), list)
    ):
        workers = list(mine_profile.get("benchmark_small_workers", ()))
        if workers:
            plan = {
                "name": "benchmark_small_workers_override",
                "workers": workers,
                "include_legacy_baseline": bool(mine_profile.get("benchmark_small_include_legacy_baseline", False)),
            }
            _log_registry_mines_event(
                "registry_benchmark_plan_resolved",
                mode="small_workers_override",
                matched_mine=True,
                size=size,
                resolved_profile=plan.get("name"),
                workers=plan.get("workers"),
            )
            return plan

    profile_name = mine_profile.get("benchmark_small_profile", mine_profile.get("benchmark_profile", fallback_name))
    if threshold is not None and size is not None and size > threshold:
        profile_name = mine_profile.get("benchmark_large_profile", profile_name)
    profile_name = _normalize_benchmark_profile(str(profile_name), profiles, fallback_name)
    plan = _profile_to_plan(profile_name, profiles[profile_name])
    _log_registry_mines_event(
        "registry_benchmark_plan_resolved",
        mode="mine_profile",
        matched_mine=True,
        size=size,
        resolved_profile=plan.get("name"),
        workers=plan.get("workers"),
    )
    return plan


def resolve_benchmark_plan(service_root, size, fallback_profile=DEFAULT_BENCHMARK_FALLBACK_PROFILE):
    size = _decode_size(size, path="resolve_benchmark_plan.size")
    registry = _load_registry()
    profiles = registry.get("benchmark_profiles", {}) or DEFAULT_BENCHMARK_PROFILES
    mine_profile = _match_mine_profile(service_root, mines=registry.get("mines", {}))
    return _resolve_benchmark_plan_from_context(
        profiles=profiles,
        size=size,
        fallback_profile=fallback_profile,
        mine_profile=mine_profile,
    )


def resolve_named_benchmark_profile(profile_name, fallback_profile=DEFAULT_BENCHMARK_FALLBACK_PROFILE):
    registry = _load_registry()
    profiles = registry.get("benchmark_profiles", {}) or DEFAULT_BENCHMARK_PROFILES
    fallback_name = _normalize_benchmark_profile(fallback_profile, profiles, DEFAULT_BENCHMARK_FALLBACK_PROFILE)
    resolved_name = _normalize_benchmark_profile(str(profile_name), profiles, fallback_name)
    return _profile_to_plan(resolved_name, profiles[resolved_name])


def registry_preferences_metrics():
    registry = _load_registry()
    metrics = dict(registry.get("_normalization_metrics", {}))
    metrics["invalid_config_attempts"] = int(_INVALID_CONFIG_ATTEMPTS)
    metrics["invalid_config_attempts_by_field"] = dict(_INVALID_CONFIG_ATTEMPTS_BY_FIELD)
    metrics["config_source"] = registry.get("_config_source")
    return metrics


def resolve_execution_plan(
    *,
    service_root,
    rows_target,
    explicit_workers,
    benchmark_profile,
    include_legacy_baseline,
):
    workers = _to_int_list(explicit_workers or [])
    if workers:
        return {
            "name": "workers_override",
            "workers": workers,
            "include_legacy_baseline": bool(include_legacy_baseline),
        }

    if benchmark_profile is not None and str(benchmark_profile).strip().lower() != "auto":
        profile_plan = resolve_named_benchmark_profile(str(benchmark_profile), DEFAULT_BENCHMARK_FALLBACK_PROFILE)
    else:
        profile_plan = resolve_benchmark_plan(service_root, rows_target, DEFAULT_BENCHMARK_FALLBACK_PROFILE)

    return {
        "name": profile_plan["name"],
        "workers": list(profile_plan["workers"]),
        "include_legacy_baseline": bool(include_legacy_baseline) and bool(profile_plan["include_legacy_baseline"]),
    }


def resolve_benchmark_phase_plan(
    *,
    service_root,
    rows_target,
    explicit_workers,
    benchmark_profile,
    include_legacy_baseline,
):
    # Backward-compatible alias for older benchmark tooling imports.
    return resolve_execution_plan(
        service_root=service_root,
        rows_target=rows_target,
        explicit_workers=explicit_workers,
        benchmark_profile=benchmark_profile,
        include_legacy_baseline=include_legacy_baseline,
    )
