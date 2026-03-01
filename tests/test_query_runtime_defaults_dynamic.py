from types import SimpleNamespace

from intermine314.query import builder as query_builder


class _Harness:
    def __init__(self):
        self.service = SimpleNamespace(tor=False, root=None)

    def _apply_parallel_profile(self, profile, ordered, large_query_mode):
        return query_builder.Query._apply_parallel_profile(self, profile, ordered, large_query_mode)

    def _resolve_effective_workers(self, max_workers, size):
        return query_builder.Query._resolve_effective_workers(self, max_workers, size)

    def _normalize_order_mode(self, ordered):
        return query_builder.Query._normalize_order_mode(self, ordered)

    def _resolve_parallel_strategy(self, pagination, start, size):
        return query_builder.Query._resolve_parallel_strategy(self, pagination, start, size)

    def _resolve_tor_parallel_context(self):
        return query_builder.Query._resolve_tor_parallel_context(self)


def _runtime_defaults_stub(**overrides):
    query_defaults = {
        "default_parallel_workers": 7,
        "default_parallel_page_size": 222,
        "default_parallel_pagination": "auto",
        "default_parallel_profile": "default",
        "default_parallel_ordered_mode": "ordered",
        "default_large_query_mode": False,
        "default_parallel_prefetch": 9,
        "default_parallel_inflight_limit": 13,
        "default_order_window_pages": 6,
        "default_keyset_batch_size": 333,
        "keyset_auto_min_size": 25,
        "default_parallel_max_buffered_rows": 100_000,
    }
    query_defaults.update(overrides)
    return SimpleNamespace(query_defaults=SimpleNamespace(**query_defaults))


def test_parallel_options_default_factory_uses_runtime_defaults(monkeypatch):
    monkeypatch.setattr(
        query_builder,
        "get_runtime_defaults",
        lambda: _runtime_defaults_stub(
            default_parallel_page_size=321,
            default_order_window_pages=4,
            default_parallel_profile="mostly_ordered",
            default_large_query_mode=True,
            default_parallel_pagination="keyset",
            default_keyset_batch_size=444,
        ),
    )

    options = query_builder.ParallelOptions()

    assert options.page_size == 321
    assert options.ordered_window_pages == 4
    assert options.profile == "mostly_ordered"
    assert options.large_query_mode is True
    assert options.pagination == "keyset"
    assert options.keyset_batch_size == 444


def test_parallel_resolution_uses_runtime_worker_prefetch_and_inflight_defaults(monkeypatch):
    monkeypatch.setattr(query_builder, "get_runtime_defaults", lambda: _runtime_defaults_stub())
    harness = _Harness()
    options = query_builder.ParallelOptions(
        max_workers=None,
        prefetch=None,
        inflight_limit=None,
        pagination="offset",
    )

    resolved = query_builder.Query._resolve_parallel_options(harness, start=0, size=100, options=options)
    expected_prefetch = query_builder.resolve_prefetch(
        None,
        max_workers=7,
        large_query_mode=False,
        default_parallel_prefetch=9,
    )
    expected_inflight = query_builder.resolve_inflight_limit(
        None,
        prefetch=expected_prefetch,
        default_parallel_inflight_limit=13,
    )

    assert resolved.max_workers == 7
    assert resolved.prefetch == expected_prefetch
    assert resolved.inflight_limit == expected_inflight


def test_parallel_strategy_uses_runtime_keyset_threshold(monkeypatch):
    monkeypatch.setattr(
        query_builder,
        "get_runtime_defaults",
        lambda: _runtime_defaults_stub(keyset_auto_min_size=10),
    )
    harness = _Harness()

    strategy = query_builder.Query._resolve_parallel_strategy(harness, "auto", 0, 12)

    assert strategy == "keyset"
