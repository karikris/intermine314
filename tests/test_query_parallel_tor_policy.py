from types import SimpleNamespace
from unittest.mock import patch

from intermine314.query import builder as query_builder

_object_sentinel = object()


class _TorPolicyHarness:
    def __init__(self, service=_object_sentinel):
        if service is not _object_sentinel:
            self.service = service

    def _apply_parallel_profile(self, profile, ordered, large_query_mode):
        return query_builder.Query._apply_parallel_profile(self, profile, ordered, large_query_mode)

    def _resolve_effective_workers(self, max_workers, size):
        _ = size
        return int(max_workers) if max_workers is not None else 4

    def _normalize_order_mode(self, ordered):
        return query_builder.Query._normalize_order_mode(self, ordered)

    def _resolve_parallel_strategy(self, pagination, start, size):
        return query_builder.Query._resolve_parallel_strategy(self, pagination, start, size)

    def _resolve_tor_parallel_context(self):
        return query_builder.Query._resolve_tor_parallel_context(self)


def _resolve_for(service, **option_overrides):
    harness = _TorPolicyHarness(service=service)
    options = query_builder.ParallelOptions(
        max_workers=4,
        profile="large_query",
        pagination="offset",
        **option_overrides,
    )
    return query_builder.Query._resolve_parallel_options(harness, start=0, size=1000, options=options)


def _expected_default_prefetch(*, max_workers=4):
    return query_builder.resolve_prefetch(
        None,
        max_workers=max_workers,
        large_query_mode=True,
        default_parallel_prefetch=query_builder.DEFAULT_PARALLEL_PREFETCH,
    )


def _expected_default_inflight(prefetch):
    return query_builder.resolve_inflight_limit(
        None,
        prefetch=prefetch,
        default_parallel_inflight_limit=query_builder.DEFAULT_PARALLEL_INFLIGHT_LIMIT,
    )


def test_parallel_policy_defaults_without_tor_match_current_behavior():
    resolved = _resolve_for(SimpleNamespace(tor=False))
    expected_prefetch = _expected_default_prefetch(max_workers=4)
    expected_inflight = _expected_default_inflight(expected_prefetch)

    assert resolved.prefetch == expected_prefetch
    assert resolved.inflight_limit == expected_inflight
    assert resolved.tor_enabled is False
    assert resolved.tor_state_known is True
    assert resolved.tor_aware_defaults_applied is False


def test_parallel_policy_applies_tor_aware_defaults_when_unset():
    resolved = _resolve_for(SimpleNamespace(tor=True))
    expected_prefetch = min(_expected_default_prefetch(max_workers=4), 4)
    expected_inflight = _expected_default_inflight(expected_prefetch)
    expected_inflight = min(expected_inflight, expected_prefetch, 4)

    assert resolved.prefetch == expected_prefetch
    assert resolved.inflight_limit == expected_inflight
    assert resolved.tor_enabled is True
    assert resolved.tor_state_known is True
    assert resolved.tor_source == "service.tor"
    assert resolved.tor_aware_defaults_applied is True


def test_parallel_policy_preserves_explicit_overrides_under_tor():
    resolved = _resolve_for(
        SimpleNamespace(tor=True),
        prefetch=9,
        inflight_limit=7,
    )

    assert resolved.prefetch == 9
    assert resolved.inflight_limit == 7
    assert resolved.tor_enabled is True
    assert resolved.tor_aware_defaults_applied is False


def test_parallel_policy_tightens_inflight_when_only_prefetch_is_overridden_under_tor():
    resolved = _resolve_for(
        SimpleNamespace(tor=True),
        prefetch=9,
        inflight_limit=None,
    )

    assert resolved.prefetch == 9
    assert resolved.inflight_limit == 4
    assert resolved.tor_enabled is True
    assert resolved.tor_aware_defaults_applied is True


def test_parallel_policy_detects_tor_from_proxy_url_when_tor_flag_missing():
    service = SimpleNamespace(proxy_url="socks5h://127.0.0.1:9050")
    resolved = _resolve_for(service)

    assert resolved.tor_enabled is True
    assert resolved.tor_state_known is True
    assert resolved.tor_source == "service.proxy_url"
    assert resolved.prefetch == 4
    assert resolved.inflight_limit == 4


def test_parallel_policy_unknown_tor_state_keeps_defaults_and_emits_debug_note():
    harness = _TorPolicyHarness(service=_object_sentinel)
    options = query_builder.ParallelOptions(max_workers=4, profile="large_query", pagination="offset")

    with patch.object(query_builder._PARALLEL_LOG, "debug") as debug_mock:
        resolved = query_builder.Query._resolve_parallel_options(harness, start=0, size=1000, options=options)

    assert resolved.tor_enabled is False
    assert resolved.tor_state_known is False
    assert resolved.tor_source == "no_service"
    assert resolved.prefetch == _expected_default_prefetch(max_workers=4)
    assert resolved.inflight_limit == _expected_default_inflight(resolved.prefetch)
    assert any(
        str(call.args[0]).startswith("parallel_tor_state_unknown")
        for call in debug_mock.call_args_list
    )
