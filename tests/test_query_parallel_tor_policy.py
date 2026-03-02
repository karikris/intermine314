from types import SimpleNamespace

from intermine314.query import builder as query_builder


class _TorPolicyHarness:
    def __init__(self, service):
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


def test_parallel_policy_defaults_without_tor():
    resolved = _resolve_for(SimpleNamespace(tor=False))
    assert resolved.tor_enabled is False
    assert resolved.tor_state_known is True
    assert resolved.tor_aware_defaults_applied is False


def test_parallel_policy_applies_tor_aware_defaults_when_unset():
    resolved = _resolve_for(SimpleNamespace(tor=True))
    assert resolved.tor_enabled is True
    assert resolved.tor_state_known is True
    assert resolved.tor_source == "service.tor"
    assert resolved.tor_aware_defaults_applied is True
    assert resolved.prefetch == 4
    assert resolved.inflight_limit == 4


def test_parallel_policy_preserves_explicit_overrides_under_tor():
    resolved = _resolve_for(SimpleNamespace(tor=True), prefetch=9, inflight_limit=7)
    assert resolved.prefetch == 9
    assert resolved.inflight_limit == 7
    assert resolved.tor_enabled is True
    assert resolved.tor_aware_defaults_applied is False

