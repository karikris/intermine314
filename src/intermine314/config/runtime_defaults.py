from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Any, Mapping

from intermine314.config.loader import (
    load_packaged_runtime_defaults_detailed,
    load_runtime_defaults_override_detailed,
)
from intermine314.parallel.policy import (
    VALID_ORDER_MODES as _VALID_ORDER_MODES,
    VALID_PARALLEL_PAGINATION as _VALID_PARALLEL_PAGINATION,
    VALID_PARALLEL_PROFILES as _VALID_PARALLEL_PROFILES,
)
from intermine314.util.logging import log_structured_event

_MAX_CONFIG_STRING_LENGTH = 128
_MAX_CONFIG_INT = 10_000_000
_MAX_CONFIG_LIST_ITEMS = 64
_VALID_TARGETED_REPORT_MODES = frozenset({"summary", "full"})
_VALID_PROXY_SCHEMES = frozenset({"socks5", "socks5h"})
RUNTIME_DEFAULTS_SCHEMA_VERSION = 1
_RUNTIME_DEFAULTS_LOG = logging.getLogger("intermine314.config.runtime_defaults")
_RUNTIME_DEFAULTS_LOAD_TELEMETRY = {
    "source": "unknown",
    "fallback_activations": 0,
    "error_kind": None,
    "schema_status": "unknown",
}


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
    default_targeted_report_mode: str = "summary"
    default_targeted_report_sample_size: int = 20


@dataclass(frozen=True)
class ServiceDefaults:
    default_connect_timeout_seconds: int = 10
    default_request_timeout_seconds: int = 60
    default_id_resolution_max_backoff_seconds: int = 60
    default_registry_instances_url: str = "https://registry.intermine.org/service/instances"
    default_tor_socks_host: str = "127.0.0.1"
    default_tor_socks_port: int = 9050
    default_tor_proxy_scheme: str = "socks5h"


@dataclass(frozen=True)
class RegistryDefaults:
    default_workers_tier: int = 4
    server_limited_workers_tier: int = 8
    full_workers_tier: int = 16
    default_production_profile_switch_rows: int = 50_000
    default_registry_service_cache_size: int = 32


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
_MINIMAL_FALLBACK_RUNTIME_DEFAULTS = _BUILTIN_RUNTIME_DEFAULTS


def _runtime_defaults_telemetry(
    *,
    source: str,
    fallback_activations: int,
    error_kind: str | None,
    schema_status: str,
) -> dict[str, object]:
    return {
        "source": source,
        "fallback_activations": int(fallback_activations),
        "error_kind": error_kind,
        "schema_status": schema_status,
    }


def _set_runtime_defaults_telemetry(
    *,
    source: str,
    error_kind: str | None,
    schema_status: str,
    used_fallback: bool,
) -> None:
    if used_fallback:
        _RUNTIME_DEFAULTS_LOAD_TELEMETRY["fallback_activations"] = int(
            _RUNTIME_DEFAULTS_LOAD_TELEMETRY["fallback_activations"]
        ) + 1
    _RUNTIME_DEFAULTS_LOAD_TELEMETRY["source"] = source
    _RUNTIME_DEFAULTS_LOAD_TELEMETRY["error_kind"] = error_kind
    _RUNTIME_DEFAULTS_LOAD_TELEMETRY["schema_status"] = schema_status


def runtime_defaults_load_telemetry() -> dict[str, object]:
    return _runtime_defaults_telemetry(
        source=str(_RUNTIME_DEFAULTS_LOAD_TELEMETRY["source"]),
        fallback_activations=int(_RUNTIME_DEFAULTS_LOAD_TELEMETRY["fallback_activations"]),
        error_kind=_RUNTIME_DEFAULTS_LOAD_TELEMETRY["error_kind"],
        schema_status=str(_RUNTIME_DEFAULTS_LOAD_TELEMETRY["schema_status"]),
    )


def reset_runtime_defaults_load_telemetry() -> None:
    _RUNTIME_DEFAULTS_LOAD_TELEMETRY["source"] = "unknown"
    _RUNTIME_DEFAULTS_LOAD_TELEMETRY["fallback_activations"] = 0
    _RUNTIME_DEFAULTS_LOAD_TELEMETRY["error_kind"] = None
    _RUNTIME_DEFAULTS_LOAD_TELEMETRY["schema_status"] = "unknown"


def _log_runtime_defaults_source(source: str, *, error_kind: str | None, schema_status: str, used_fallback: bool) -> None:
    level = logging.WARNING if used_fallback else logging.DEBUG
    log_structured_event(
        _RUNTIME_DEFAULTS_LOG,
        level,
        "runtime_defaults_source",
        source=source,
        schema_status=schema_status,
        error_kind=error_kind,
        used_fallback=bool(used_fallback),
        fallback_activations=int(_RUNTIME_DEFAULTS_LOAD_TELEMETRY["fallback_activations"]),
    )


def _schema_version(payload: Mapping[str, Any]) -> int | None:
    meta = _to_mapping(payload.get("meta"))
    raw = meta.get("schema_version")
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _schema_status(payload: Mapping[str, Any], *, require_schema: bool) -> tuple[bool, str]:
    version = _schema_version(payload)
    if version is None:
        if require_schema:
            return False, "missing"
        return True, "absent"
    if version != int(RUNTIME_DEFAULTS_SCHEMA_VERSION):
        return False, "mismatch"
    return True, "ok"


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


def _parse_non_negative_int(raw: Any, default: int) -> int:
    try:
        parsed = int(raw)
    except Exception:
        return int(default)
    if parsed < 0:
        return int(default)
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


def parse_runtime_defaults(payload: Mapping[str, Any] | None, *, base: RuntimeDefaults | None = None) -> RuntimeDefaults:
    root = _to_mapping(payload)
    query_raw = _to_mapping(root.get("query_defaults"))
    runtime_base = _BUILTIN_RUNTIME_DEFAULTS if base is None else base
    query_builtin = runtime_base.query_defaults
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
    list_builtin = runtime_base.list_defaults
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
    targeted_builtin = runtime_base.targeted_export_defaults
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
        default_targeted_report_mode=_parse_choice(
            targeted_raw.get("default_targeted_report_mode"),
            targeted_builtin.default_targeted_report_mode,
            _VALID_TARGETED_REPORT_MODES,
        ),
        default_targeted_report_sample_size=_parse_non_negative_int(
            targeted_raw.get("default_targeted_report_sample_size"),
            targeted_builtin.default_targeted_report_sample_size,
        ),
    )

    service_raw = _to_mapping(root.get("service_defaults"))
    service_builtin = runtime_base.service_defaults
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
        default_registry_instances_url=_parse_small_string(
            service_raw.get("default_registry_instances_url"),
            service_builtin.default_registry_instances_url,
        ),
        default_tor_socks_host=_parse_small_string(
            service_raw.get("default_tor_socks_host"),
            service_builtin.default_tor_socks_host,
        ),
        default_tor_socks_port=_parse_positive_int(
            service_raw.get("default_tor_socks_port"),
            service_builtin.default_tor_socks_port,
        ),
        default_tor_proxy_scheme=_parse_choice(
            service_raw.get("default_tor_proxy_scheme"),
            service_builtin.default_tor_proxy_scheme,
            _VALID_PROXY_SCHEMES,
        ),
    )

    registry_raw = _to_mapping(root.get("registry_defaults"))
    registry_builtin = runtime_base.registry_defaults
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
        default_registry_service_cache_size=_parse_positive_int(
            registry_raw.get("default_registry_service_cache_size"),
            registry_builtin.default_registry_service_cache_size,
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
    packaged = load_packaged_runtime_defaults_detailed()
    packaged_payload = packaged.get("payload")
    packaged_error = packaged.get("error_kind")
    packaged_ok = bool(packaged.get("ok", True))
    if not packaged_ok:
        fallback_reason = f"packaged_{packaged_error}" if isinstance(packaged_error, str) else "packaged_load_error"
        _set_runtime_defaults_telemetry(
            source="minimal_fallback",
            error_kind=fallback_reason,
            schema_status="missing",
            used_fallback=True,
        )
        _log_runtime_defaults_source(
            "minimal_fallback",
            error_kind=fallback_reason,
            schema_status="missing",
            used_fallback=True,
        )
        return _MINIMAL_FALLBACK_RUNTIME_DEFAULTS
    if not isinstance(packaged_payload, Mapping):
        _set_runtime_defaults_telemetry(
            source="minimal_fallback",
            error_kind="invalid_packaged_shape",
            schema_status="missing",
            used_fallback=True,
        )
        _log_runtime_defaults_source(
            "minimal_fallback",
            error_kind="invalid_packaged_shape",
            schema_status="missing",
            used_fallback=True,
        )
        return _MINIMAL_FALLBACK_RUNTIME_DEFAULTS

    schema_ok, schema_state = _schema_status(packaged_payload, require_schema=True)
    if not schema_ok:
        fallback_reason = (
            "missing_packaged_schema" if schema_state == "missing" else "packaged_schema_mismatch"
        )
        _set_runtime_defaults_telemetry(
            source="minimal_fallback",
            error_kind=fallback_reason,
            schema_status=schema_state,
            used_fallback=True,
        )
        _log_runtime_defaults_source(
            "minimal_fallback",
            error_kind=fallback_reason,
            schema_status=schema_state,
            used_fallback=True,
        )
        return _MINIMAL_FALLBACK_RUNTIME_DEFAULTS

    parsed = parse_runtime_defaults(packaged_payload, base=_MINIMAL_FALLBACK_RUNTIME_DEFAULTS)
    source = "packaged_toml"
    error_kind = packaged_error if isinstance(packaged_error, str) else None

    override = load_runtime_defaults_override_detailed()
    if override is not None:
        override_payload = override.get("payload")
        override_error = override.get("error_kind")
        if isinstance(override_payload, Mapping):
            override_schema_ok, override_schema_state = _schema_status(override_payload, require_schema=False)
            if override_schema_ok:
                parsed = parse_runtime_defaults(override_payload, base=parsed)
                source = "override_toml"
                error_kind = override_error if isinstance(override_error, str) else None
                schema_state = override_schema_state
            else:
                source = "packaged_toml"
                error_kind = "override_schema_mismatch"
                schema_state = override_schema_state
        else:
            source = "packaged_toml"
            if isinstance(override_error, str):
                error_kind = f"override_{override_error}"
            else:
                error_kind = "override_invalid_shape"
            schema_state = "ok"
    _set_runtime_defaults_telemetry(
        source=source,
        error_kind=error_kind,
        schema_status=schema_state,
        used_fallback=False,
    )
    _log_runtime_defaults_source(
        source,
        error_kind=error_kind,
        schema_status=schema_state,
        used_fallback=False,
    )
    return parsed


def clear_runtime_defaults_cache() -> None:
    get_runtime_defaults.cache_clear()
