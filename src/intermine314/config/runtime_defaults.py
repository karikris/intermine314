from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from intermine314.config.loader import load_runtime_defaults

_MAX_CONFIG_STRING_LENGTH = 128
_MAX_CONFIG_INT = 10_000_000
_MAX_CONFIG_LIST_ITEMS = 64
_VALID_PARALLEL_PAGINATION = frozenset({"auto", "offset", "keyset"})
_VALID_PARALLEL_PROFILES = frozenset({"default", "large_query", "unordered", "mostly_ordered"})
_VALID_ORDER_MODES = frozenset({"ordered", "unordered", "window", "mostly_ordered"})


@dataclass(frozen=True)
class QueryDefaults:
    default_parallel_workers: int = 16
    default_parallel_page_size: int = 1000
    default_parallel_pagination: str = "auto"
    default_parallel_profile: str = "default"
    default_parallel_ordered_mode: str = "ordered"
    default_large_query_mode: bool = False
    default_parallel_prefetch: int | None = None
    default_parallel_inflight_limit: int | None = None
    default_order_window_pages: int = 10
    default_keyset_batch_size: int = 2000
    keyset_auto_min_size: int = 50_000
    default_parallel_max_buffered_rows: int = 100_000
    default_batch_size: int = 5000
    default_export_batch_size: int = 10_000
    default_query_thread_name_prefix: str = "intermine314-query"


@dataclass(frozen=True)
class ListDefaults:
    default_list_chunk_size: int = 10_000
    default_list_entries_batch_size: int = 5000


@dataclass(frozen=True)
class TargetedExportDefaults:
    default_targeted_export_page_size: int = 5_000
    default_targeted_list_name_prefix: str = "intermine314_targeted_chunk"
    default_targeted_list_description: str = "Temporary chunk list for targeted benchmark export"
    default_targeted_list_tags: tuple[str, ...] = ("intermine314", "benchmark", "targeted")


@dataclass(frozen=True)
class ServiceDefaults:
    default_connect_timeout_seconds: int = 10
    default_request_timeout_seconds: int = 60
    default_id_resolution_max_backoff_seconds: int = 60


@dataclass(frozen=True)
class RegistryDefaults:
    default_workers_tier: int = 4
    server_limited_workers_tier: int = 8
    full_workers_tier: int = 16
    default_production_profile_switch_rows: int = 50_000


@dataclass(frozen=True)
class RuntimeDefaults:
    query_defaults: QueryDefaults
    list_defaults: ListDefaults
    targeted_export_defaults: TargetedExportDefaults
    service_defaults: ServiceDefaults
    registry_defaults: RegistryDefaults


_BUILTIN_QUERY_DEFAULTS = QueryDefaults()
_BUILTIN_LIST_DEFAULTS = ListDefaults()
_BUILTIN_TARGETED_EXPORT_DEFAULTS = TargetedExportDefaults()
_BUILTIN_SERVICE_DEFAULTS = ServiceDefaults()
_BUILTIN_REGISTRY_DEFAULTS = RegistryDefaults()
_BUILTIN_RUNTIME_DEFAULTS = RuntimeDefaults(
    query_defaults=_BUILTIN_QUERY_DEFAULTS,
    list_defaults=_BUILTIN_LIST_DEFAULTS,
    targeted_export_defaults=_BUILTIN_TARGETED_EXPORT_DEFAULTS,
    service_defaults=_BUILTIN_SERVICE_DEFAULTS,
    registry_defaults=_BUILTIN_REGISTRY_DEFAULTS,
)


def _parse_positive_int(raw: Any, default: int) -> int:
    try:
        parsed = int(raw)
    except Exception:
        return int(default)
    if parsed <= 0:
        return int(default)
    return min(parsed, _MAX_CONFIG_INT)


def _parse_choice(raw: Any, default: str, valid_values: set[str] | frozenset[str]) -> str:
    if raw is None:
        return str(default)
    value = str(raw).strip().lower()
    if len(value) > _MAX_CONFIG_STRING_LENGTH:
        return str(default)
    return value if value in valid_values else str(default)


def _parse_optional_positive_int(raw: Any, default: int | None) -> int | None:
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if len(value) > _MAX_CONFIG_STRING_LENGTH:
        return default
    if value in {"", "none", "null", "auto"}:
        return None
    try:
        parsed = int(value)
    except Exception:
        return default
    if parsed <= 0:
        return default
    return min(parsed, _MAX_CONFIG_INT)


def _parse_bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if len(value) > _MAX_CONFIG_STRING_LENGTH:
            return bool(default)
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _parse_small_string(raw: Any, default: str) -> str:
    if raw is None:
        return str(default)
    value = str(raw).strip()
    if not value:
        return str(default)
    if len(value) > _MAX_CONFIG_STRING_LENGTH:
        return str(default)
    return value


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _parse_small_string_list(raw: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return tuple(default)
    result: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if len(result) >= _MAX_CONFIG_LIST_ITEMS:
            break
        item = str(value).strip()
        if not item:
            continue
        if len(item) > _MAX_CONFIG_STRING_LENGTH:
            continue
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    if not result:
        return tuple(default)
    return tuple(result)


def parse_runtime_defaults(payload: Mapping[str, Any] | None) -> RuntimeDefaults:
    root = _to_mapping(payload)
    query_raw = _to_mapping(root.get("query_defaults"))
    query_builtin = _BUILTIN_QUERY_DEFAULTS
    query_defaults = QueryDefaults(
        default_parallel_workers=_parse_positive_int(
            query_raw.get("default_parallel_workers"),
            query_builtin.default_parallel_workers,
        ),
        default_parallel_page_size=_parse_positive_int(
            query_raw.get("default_parallel_page_size"),
            query_builtin.default_parallel_page_size,
        ),
        default_parallel_pagination=_parse_choice(
            query_raw.get("default_parallel_pagination"),
            query_builtin.default_parallel_pagination,
            _VALID_PARALLEL_PAGINATION,
        ),
        default_parallel_profile=_parse_choice(
            query_raw.get("default_parallel_profile"),
            query_builtin.default_parallel_profile,
            _VALID_PARALLEL_PROFILES,
        ),
        default_parallel_ordered_mode=_parse_choice(
            query_raw.get("default_parallel_ordered_mode"),
            query_builtin.default_parallel_ordered_mode,
            _VALID_ORDER_MODES,
        ),
        default_large_query_mode=_parse_bool(
            query_raw.get("default_large_query_mode"),
            query_builtin.default_large_query_mode,
        ),
        default_parallel_prefetch=_parse_optional_positive_int(
            query_raw.get("default_parallel_prefetch"),
            query_builtin.default_parallel_prefetch,
        ),
        default_parallel_inflight_limit=_parse_optional_positive_int(
            query_raw.get("default_parallel_inflight_limit"),
            query_builtin.default_parallel_inflight_limit,
        ),
        default_order_window_pages=_parse_positive_int(
            query_raw.get("default_order_window_pages"),
            query_builtin.default_order_window_pages,
        ),
        default_keyset_batch_size=_parse_positive_int(
            query_raw.get("default_keyset_batch_size"),
            query_builtin.default_keyset_batch_size,
        ),
        keyset_auto_min_size=_parse_positive_int(
            query_raw.get("keyset_auto_min_size"),
            query_builtin.keyset_auto_min_size,
        ),
        default_parallel_max_buffered_rows=_parse_positive_int(
            query_raw.get("default_parallel_max_buffered_rows"),
            query_builtin.default_parallel_max_buffered_rows,
        ),
        default_batch_size=_parse_positive_int(
            query_raw.get("default_batch_size"),
            query_builtin.default_batch_size,
        ),
        default_export_batch_size=_parse_positive_int(
            query_raw.get("default_export_batch_size"),
            query_builtin.default_export_batch_size,
        ),
        default_query_thread_name_prefix=_parse_small_string(
            query_raw.get("default_query_thread_name_prefix"),
            query_builtin.default_query_thread_name_prefix,
        ),
    )
    list_raw = _to_mapping(root.get("list_defaults"))
    list_builtin = _BUILTIN_LIST_DEFAULTS
    list_defaults = ListDefaults(
        default_list_chunk_size=_parse_positive_int(
            list_raw.get("default_list_chunk_size"),
            list_builtin.default_list_chunk_size,
        ),
        default_list_entries_batch_size=_parse_positive_int(
            list_raw.get("default_list_entries_batch_size"),
            query_defaults.default_batch_size,
        ),
    )

    targeted_raw = _to_mapping(root.get("targeted_export_defaults"))
    targeted_builtin = _BUILTIN_TARGETED_EXPORT_DEFAULTS
    targeted_defaults = TargetedExportDefaults(
        default_targeted_export_page_size=_parse_positive_int(
            targeted_raw.get("default_targeted_export_page_size"),
            targeted_builtin.default_targeted_export_page_size,
        ),
        default_targeted_list_name_prefix=_parse_small_string(
            targeted_raw.get("default_targeted_list_name_prefix"),
            targeted_builtin.default_targeted_list_name_prefix,
        ),
        default_targeted_list_description=_parse_small_string(
            targeted_raw.get("default_targeted_list_description"),
            targeted_builtin.default_targeted_list_description,
        ),
        default_targeted_list_tags=_parse_small_string_list(
            targeted_raw.get("default_targeted_list_tags"),
            targeted_builtin.default_targeted_list_tags,
        ),
    )

    service_raw = _to_mapping(root.get("service_defaults"))
    service_builtin = _BUILTIN_SERVICE_DEFAULTS
    service_defaults = ServiceDefaults(
        default_connect_timeout_seconds=_parse_positive_int(
            service_raw.get("default_connect_timeout_seconds"),
            service_builtin.default_connect_timeout_seconds,
        ),
        default_request_timeout_seconds=_parse_positive_int(
            service_raw.get("default_request_timeout_seconds"),
            service_builtin.default_request_timeout_seconds,
        ),
        default_id_resolution_max_backoff_seconds=_parse_positive_int(
            service_raw.get("default_id_resolution_max_backoff_seconds"),
            service_builtin.default_id_resolution_max_backoff_seconds,
        ),
    )

    registry_raw = _to_mapping(root.get("registry_defaults"))
    registry_builtin = _BUILTIN_REGISTRY_DEFAULTS
    registry_defaults = RegistryDefaults(
        default_workers_tier=_parse_positive_int(
            registry_raw.get("default_workers_tier"),
            registry_builtin.default_workers_tier,
        ),
        server_limited_workers_tier=_parse_positive_int(
            registry_raw.get("server_limited_workers_tier"),
            registry_builtin.server_limited_workers_tier,
        ),
        full_workers_tier=_parse_positive_int(
            registry_raw.get("full_workers_tier"),
            registry_builtin.full_workers_tier,
        ),
        default_production_profile_switch_rows=_parse_positive_int(
            registry_raw.get("default_production_profile_switch_rows"),
            registry_builtin.default_production_profile_switch_rows,
        ),
    )

    return RuntimeDefaults(
        query_defaults=query_defaults,
        list_defaults=list_defaults,
        targeted_export_defaults=targeted_defaults,
        service_defaults=service_defaults,
        registry_defaults=registry_defaults,
    )


@lru_cache(maxsize=1)
def get_runtime_defaults() -> RuntimeDefaults:
    loaded = load_runtime_defaults()
    if not isinstance(loaded, Mapping):
        return _BUILTIN_RUNTIME_DEFAULTS
    return parse_runtime_defaults(loaded)


def clear_runtime_defaults_cache() -> None:
    get_runtime_defaults.cache_clear()
