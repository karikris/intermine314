"""Deterministic scheduling checks include results blocked behind earlier pages."""

from concurrent.futures import Future

import pytest

from intermine314.query import parallel_offset
from intermine314.query.inflight import BoundedInflightQueue


def capture_queue(monkeypatch, **options):
    queue = BoundedInflightQueue(**options)
    monkeypatch.setattr(parallel_offset, "BoundedInflightQueue", lambda **kwargs: queue)
    return queue


class PayloadQuery:
    def results(self, row="dict", start=0, size=None):
        return iter([{"value": start, "payload": "x" * 1024}])


class DelayedSecondPageExecutor:
    def __init__(self, **kwargs):
        self.delayed = None
        self.submitted = 0
        self.submitted_before_delayed_completes = None
        self.exit_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exit_calls += 1
        return False

    def submit(self, fn, index, offset):
        self.submitted += 1
        future = Future()
        if index == 1:
            self.delayed = (future, fn, index, offset)
        else:
            future.set_result(fn(index, offset))
        return future

    def complete_available_pages(self, futures, **kwargs):
        ready = {future for future in futures if future.done()}
        if not ready:
            self.submitted_before_delayed_completes = self.submitted
            future, fn, index, offset = self.delayed
            future.set_result(fn(index, offset))
            self.delayed = None
            ready = {future}
        return ready, set(futures) - ready


@pytest.mark.parametrize("bytes_budget", [None, 4096])
def test_ordered_pages_bound_buffered_results_when_second_page_stalls(monkeypatch, bytes_budget):
    executor = DelayedSecondPageExecutor()
    queue = capture_queue(monkeypatch, inflight_limit=2, max_inflight_bytes_estimate=bytes_budget)
    monkeypatch.setattr(parallel_offset, "wait", executor.complete_available_pages)
    iterator = parallel_offset.run_parallel_offset(
        PayloadQuery(), size=100, page_size=1, max_workers=2,
        inflight_limit=2, order_mode="ordered", max_inflight_bytes_estimate=bytes_budget,
        thread_name_prefix="memory-bounds", executor_cls=lambda **kwargs: executor,
    )
    assert [row["value"] for row in iterator] == list(range(100))
    # Only the consumed first page and two outstanding pages may be submitted.
    assert executor.submitted_before_delayed_completes <= 3
    assert executor.submitted == 100 and executor.exit_calls == 1
    stats = queue.stats_fields()
    assert stats["peak_outstanding_pages"] == stats["peak_buffered_pages"] == 2
    assert stats["outstanding_pages"] == stats["buffered_pages"] == stats["emitting_pages"] == 0
    if bytes_budget:
        assert 2048 <= stats["max_estimated_inflight_bytes"] <= bytes_budget
        assert stats["peak_estimated_buffered_bytes"] == stats["max_estimated_inflight_bytes"]


@pytest.mark.parametrize("order_mode", ["ordered", "unordered"])
def test_partial_page_emission_remains_charged(monkeypatch, order_mode):
    class Query:
        def results(self, start=0, size=None, **kwargs):
            return iter([{"value": value} for value in range(start, start + size)])

    queue = capture_queue(monkeypatch, inflight_limit=2, max_inflight_bytes_estimate=4096)
    iterator = parallel_offset.run_parallel_offset(
        Query(), size=6, page_size=3, max_workers=2, inflight_limit=2,
        max_inflight_bytes_estimate=4096, order_mode=order_mode, thread_name_prefix="partial-emission",
    )
    assert next(iterator) == {"value": 0}
    stats = queue.stats_fields()
    assert stats["outstanding_pages"] == stats["emitting_pages"] == 1
    assert stats["estimated_inflight_bytes"] >= 3 * 8
    assert sorted(row["value"] for row in iterator) == list(range(1, 6))
    assert queue.stats_fields()["estimated_inflight_bytes"] == 0


@pytest.mark.parametrize("order_mode", ["ordered", "unordered"])
@pytest.mark.parametrize("payload_kind", ["oversized", "unestimable"])
def test_large_or_unestimable_pages_make_progress_one_page_at_a_time(monkeypatch, order_mode, payload_kind):
    class Unestimable:
        def __str__(self):
            raise ValueError("cannot estimate")

    class Query:
        def results(self, start=0, **kwargs):
            payload = "x" * 10000 if payload_kind == "oversized" else Unestimable()
            return iter([{"value": start, "payload": payload}])

    queue = capture_queue(monkeypatch, inflight_limit=8, max_inflight_bytes_estimate=64)
    iterator = parallel_offset.run_parallel_offset(
        Query(), size=8, page_size=1, max_workers=4, inflight_limit=8,
        max_inflight_bytes_estimate=64, order_mode=order_mode, thread_name_prefix="single-page-progress",
    )
    assert [row["value"] for row in iterator] == list(range(8))
    stats = queue.stats_fields()
    assert stats["peak_outstanding_pages"] == 1
    assert stats["bytes_cap_hits"] > 0
    assert stats["bytes_estimator_failures"] == (1 if payload_kind == "unestimable" else 0)
