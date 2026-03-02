from __future__ import annotations

from dataclasses import replace

from intermine314.config import runtime_defaults as runtime_defaults_mod


def _parse(payload: dict[str, object], *, base=None):
    return runtime_defaults_mod.parse_runtime_defaults(payload, base=base)


def test_runtime_defaults_positive_int_validation():
    parsed = _parse(
        {
            "query_defaults": {
                "default_parallel_workers": -4,
                "default_parallel_page_size": 777,
            }
        }
    )
    assert parsed.query_defaults.default_parallel_workers == runtime_defaults_mod.QueryDefaults().default_parallel_workers
    assert parsed.query_defaults.default_parallel_page_size == 777


def test_runtime_defaults_optional_int_validation_for_auto_none_and_invalid():
    base = _parse({})
    custom_base = replace(
        base,
        query_defaults=replace(base.query_defaults, default_parallel_prefetch=11, default_parallel_inflight_limit=13),
    )
    parsed = _parse(
        {
            "query_defaults": {
                "default_parallel_prefetch": "auto",
                "default_parallel_inflight_limit": "invalid",
            }
        },
        base=custom_base,
    )
    assert parsed.query_defaults.default_parallel_prefetch is None
    assert parsed.query_defaults.default_parallel_inflight_limit == 13

    parsed_explicit_null = _parse(
        {"query_defaults": {"default_parallel_prefetch": "17", "default_parallel_inflight_limit": "none"}},
        base=custom_base,
    )
    assert parsed_explicit_null.query_defaults.default_parallel_prefetch == 17
    assert parsed_explicit_null.query_defaults.default_parallel_inflight_limit is None


def test_runtime_defaults_enum_choice_validation():
    base = _parse({})
    parsed = _parse(
        {
            "query_defaults": {
                "default_parallel_profile": "invalid_profile",
                "default_parallel_ordered_mode": "window",
            },
            "storage_defaults": {"default_parquet_compression": "SnApPy"},
        },
        base=base,
    )
    assert parsed.query_defaults.default_parallel_profile == base.query_defaults.default_parallel_profile
    assert parsed.query_defaults.default_parallel_ordered_mode == "window"
    assert parsed.storage_defaults.default_parquet_compression == "snappy"


def test_runtime_defaults_bool_validation():
    parsed_true = _parse({"query_defaults": {"default_large_query_mode": "yes"}})
    assert parsed_true.query_defaults.default_large_query_mode is True

    parsed_false = _parse({"query_defaults": {"default_large_query_mode": "off"}})
    assert parsed_false.query_defaults.default_large_query_mode is False


def test_runtime_defaults_bounded_string_validation():
    base = _parse({})
    parsed = _parse(
        {
            "query_defaults": {
                "default_query_thread_name_prefix": "  worker-prefix  ",
            }
        },
        base=base,
    )
    assert parsed.query_defaults.default_query_thread_name_prefix == "worker-prefix"

    oversized = _parse(
        {"query_defaults": {"default_query_thread_name_prefix": "x" * 1024}},
        base=base,
    )
    assert oversized.query_defaults.default_query_thread_name_prefix == base.query_defaults.default_query_thread_name_prefix


def test_runtime_defaults_list_validation_deduplicates_and_applies_bounds():
    tags_payload = ["alpha", "alpha", "", "x" * 1024] + [f"tag{i}" for i in range(80)]
    parsed = _parse(
        {
            "targeted_export_defaults": {
                "default_targeted_list_tags": tags_payload,
            },
            "transport_defaults": {
                "default_http_retry_status_codes": [429, 429, -1, "503", 123456789],
                "default_http_retry_methods": ["get", "GET", "post", "TRACE", "DELETE", ""],
            },
        }
    )
    tags = parsed.targeted_export_defaults.default_targeted_list_tags
    assert tags[0] == "alpha"
    assert len(tags) == 64
    assert "x" * 1024 not in tags
    assert tags.count("alpha") == 1

    assert parsed.transport_defaults.default_http_retry_status_codes == (429, 503, 10_000_000)
    assert parsed.transport_defaults.default_http_retry_methods == ("GET", "POST", "DELETE")
