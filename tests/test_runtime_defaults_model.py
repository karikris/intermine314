from intermine314.config.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_EXPORT_BATCH_SIZE,
    DEFAULT_ID_RESOLUTION_MAX_BACKOFF_SECONDS,
    DEFAULT_KEYSET_BATCH_SIZE,
    DEFAULT_KEYSET_AUTO_MIN_SIZE,
    DEFAULT_LARGE_QUERY_MODE,
    DEFAULT_LIST_CHUNK_SIZE,
    DEFAULT_LIST_ENTRIES_BATCH_SIZE,
    DEFAULT_ORDER_WINDOW_PAGES,
    DEFAULT_PARALLEL_INFLIGHT_LIMIT,
    DEFAULT_PARALLEL_MAX_BUFFERED_ROWS,
    DEFAULT_PARALLEL_ORDERED_MODE,
    DEFAULT_PARALLEL_PAGE_SIZE,
    DEFAULT_PARALLEL_PAGINATION,
    DEFAULT_PARALLEL_PREFETCH,
    DEFAULT_PARALLEL_PROFILE,
    DEFAULT_PARALLEL_WORKERS,
    DEFAULT_PRODUCTION_PROFILE_SWITCH_ROWS,
    DEFAULT_QUERY_THREAD_NAME_PREFIX,
    DEFAULT_REGISTRY_INSTANCES_URL,
    DEFAULT_REGISTRY_SERVICE_CACHE_SIZE,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TARGETED_EXPORT_PAGE_SIZE,
    DEFAULT_TARGETED_LIST_DESCRIPTION,
    DEFAULT_TARGETED_LIST_NAME_PREFIX,
    DEFAULT_TARGETED_REPORT_MODE,
    DEFAULT_TARGETED_REPORT_SAMPLE_SIZE,
    DEFAULT_TARGETED_LIST_TAGS,
    DEFAULT_TOR_PROXY_SCHEME,
    DEFAULT_TOR_SOCKS_HOST,
    DEFAULT_TOR_SOCKS_PORT,
    DEFAULT_WORKERS_TIER,
    FULL_WORKERS_TIER,
    SERVER_LIMITED_WORKERS_TIER,
)
from intermine314.config.runtime_defaults import clear_runtime_defaults_cache, get_runtime_defaults


def test_runtime_defaults_model_honors_override_and_validates(tmp_path, monkeypatch):
    override = tmp_path / "runtime-defaults-validated.toml"
    override.write_text(
        "\n".join(
            [
                "[query_defaults]",
                "default_parallel_workers = -4",
                "default_parallel_page_size = 777",
                'default_parallel_pagination = "keyset"',
                'default_parallel_profile = "invalid_profile"',
                'default_parallel_ordered_mode = "window"',
                "default_large_query_mode = true",
                "default_parallel_prefetch = 9",
                'default_parallel_inflight_limit = "auto"',
                "default_order_window_pages = 6",
                "default_keyset_batch_size = 333",
                "keyset_auto_min_size = 1000",
                "default_parallel_max_buffered_rows = 9000",
                "default_batch_size = 222",
                "default_export_batch_size = 444",
                'default_query_thread_name_prefix = "my-prefix"',
                "[list_defaults]",
                "default_list_chunk_size = 6543",
                "default_list_entries_batch_size = -1",
                "[targeted_export_defaults]",
                "default_targeted_export_page_size = 876",
                'default_targeted_list_name_prefix = "prefix-x"',
                'default_targeted_list_description = "desc-x"',
                'default_targeted_list_tags = ["x", "y", "z"]',
                'default_targeted_report_mode = "full"',
                "default_targeted_report_sample_size = 0",
                "[service_defaults]",
                "default_connect_timeout_seconds = 11",
                "default_request_timeout_seconds = 71",
                "default_id_resolution_max_backoff_seconds = 91",
                'default_registry_instances_url = "https://registry.example.test/service/instances"',
                'default_tor_socks_host = "tor.local"',
                "default_tor_socks_port = 9150",
                'default_tor_proxy_scheme = "socks5"',
                "[registry_defaults]",
                "default_workers_tier = 5",
                "server_limited_workers_tier = 9",
                "full_workers_tier = 17",
                "default_production_profile_switch_rows = 12345",
                "default_registry_service_cache_size = 7",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", str(override))
    clear_runtime_defaults_cache()
    defaults = get_runtime_defaults()
    query_defaults = defaults.query_defaults
    list_defaults = defaults.list_defaults
    targeted_defaults = defaults.targeted_export_defaults
    service_defaults = defaults.service_defaults
    registry_defaults = defaults.registry_defaults

    # Invalid values should fall back; valid values should be preserved.
    assert query_defaults.default_parallel_workers == 16
    assert query_defaults.default_parallel_page_size == 777
    assert query_defaults.default_parallel_pagination == "keyset"
    assert query_defaults.default_parallel_profile == "default"
    assert query_defaults.default_parallel_ordered_mode == "window"
    assert query_defaults.default_large_query_mode is True
    assert query_defaults.default_parallel_prefetch == 9
    assert query_defaults.default_parallel_inflight_limit is None
    assert query_defaults.default_order_window_pages == 6
    assert query_defaults.default_keyset_batch_size == 333
    assert query_defaults.keyset_auto_min_size == 1000
    assert query_defaults.default_parallel_max_buffered_rows == 9000
    assert query_defaults.default_batch_size == 222
    assert query_defaults.default_export_batch_size == 444
    assert query_defaults.default_query_thread_name_prefix == "my-prefix"

    # List defaults inherit and validate against query defaults.
    assert list_defaults.default_list_chunk_size == 6543
    assert list_defaults.default_list_entries_batch_size == 222

    assert targeted_defaults.default_targeted_export_page_size == 876
    assert targeted_defaults.default_targeted_list_name_prefix == "prefix-x"
    assert targeted_defaults.default_targeted_list_description == "desc-x"
    assert targeted_defaults.default_targeted_list_tags == ("x", "y", "z")
    assert targeted_defaults.default_targeted_report_mode == "full"
    assert targeted_defaults.default_targeted_report_sample_size == 0

    assert service_defaults.default_connect_timeout_seconds == 11
    assert service_defaults.default_request_timeout_seconds == 71
    assert service_defaults.default_id_resolution_max_backoff_seconds == 91
    assert service_defaults.default_registry_instances_url == "https://registry.example.test/service/instances"
    assert service_defaults.default_tor_socks_host == "tor.local"
    assert service_defaults.default_tor_socks_port == 9150
    assert service_defaults.default_tor_proxy_scheme == "socks5"

    assert registry_defaults.default_workers_tier == 5
    assert registry_defaults.server_limited_workers_tier == 9
    assert registry_defaults.full_workers_tier == 17
    assert registry_defaults.default_production_profile_switch_rows == 12345
    assert registry_defaults.default_registry_service_cache_size == 7
    clear_runtime_defaults_cache()


def test_constants_are_compatible_aliases_to_runtime_defaults():
    defaults = get_runtime_defaults()
    query_defaults = defaults.query_defaults
    list_defaults = defaults.list_defaults
    targeted_defaults = defaults.targeted_export_defaults
    service_defaults = defaults.service_defaults
    registry_defaults = defaults.registry_defaults

    assert DEFAULT_PARALLEL_WORKERS == query_defaults.default_parallel_workers
    assert DEFAULT_PARALLEL_PAGE_SIZE == query_defaults.default_parallel_page_size
    assert DEFAULT_PARALLEL_PAGINATION == query_defaults.default_parallel_pagination
    assert DEFAULT_PARALLEL_PROFILE == query_defaults.default_parallel_profile
    assert DEFAULT_PARALLEL_ORDERED_MODE == query_defaults.default_parallel_ordered_mode
    assert DEFAULT_LARGE_QUERY_MODE == query_defaults.default_large_query_mode
    assert DEFAULT_PARALLEL_PREFETCH == query_defaults.default_parallel_prefetch
    assert DEFAULT_PARALLEL_INFLIGHT_LIMIT == query_defaults.default_parallel_inflight_limit
    assert DEFAULT_ORDER_WINDOW_PAGES == query_defaults.default_order_window_pages
    assert DEFAULT_KEYSET_BATCH_SIZE == query_defaults.default_keyset_batch_size
    assert DEFAULT_KEYSET_AUTO_MIN_SIZE == query_defaults.keyset_auto_min_size
    assert DEFAULT_PARALLEL_MAX_BUFFERED_ROWS == query_defaults.default_parallel_max_buffered_rows
    assert DEFAULT_BATCH_SIZE == query_defaults.default_batch_size
    assert DEFAULT_EXPORT_BATCH_SIZE == query_defaults.default_export_batch_size
    assert DEFAULT_QUERY_THREAD_NAME_PREFIX == query_defaults.default_query_thread_name_prefix

    assert DEFAULT_LIST_CHUNK_SIZE == list_defaults.default_list_chunk_size
    assert DEFAULT_LIST_ENTRIES_BATCH_SIZE == list_defaults.default_list_entries_batch_size

    assert DEFAULT_TARGETED_EXPORT_PAGE_SIZE == targeted_defaults.default_targeted_export_page_size
    assert DEFAULT_TARGETED_LIST_NAME_PREFIX == targeted_defaults.default_targeted_list_name_prefix
    assert DEFAULT_TARGETED_LIST_DESCRIPTION == targeted_defaults.default_targeted_list_description
    assert DEFAULT_TARGETED_LIST_TAGS == targeted_defaults.default_targeted_list_tags
    assert DEFAULT_TARGETED_REPORT_MODE == targeted_defaults.default_targeted_report_mode
    assert DEFAULT_TARGETED_REPORT_SAMPLE_SIZE == targeted_defaults.default_targeted_report_sample_size

    assert DEFAULT_CONNECT_TIMEOUT_SECONDS == service_defaults.default_connect_timeout_seconds
    assert DEFAULT_REQUEST_TIMEOUT_SECONDS == service_defaults.default_request_timeout_seconds
    assert DEFAULT_ID_RESOLUTION_MAX_BACKOFF_SECONDS == service_defaults.default_id_resolution_max_backoff_seconds
    assert DEFAULT_REGISTRY_INSTANCES_URL == service_defaults.default_registry_instances_url
    assert DEFAULT_TOR_SOCKS_HOST == service_defaults.default_tor_socks_host
    assert DEFAULT_TOR_SOCKS_PORT == service_defaults.default_tor_socks_port
    assert DEFAULT_TOR_PROXY_SCHEME == service_defaults.default_tor_proxy_scheme

    assert DEFAULT_WORKERS_TIER == registry_defaults.default_workers_tier
    assert SERVER_LIMITED_WORKERS_TIER == registry_defaults.server_limited_workers_tier
    assert FULL_WORKERS_TIER == registry_defaults.full_workers_tier
    assert DEFAULT_PRODUCTION_PROFILE_SWITCH_ROWS == registry_defaults.default_production_profile_switch_rows
    assert DEFAULT_REGISTRY_SERVICE_CACHE_SIZE == registry_defaults.default_registry_service_cache_size
