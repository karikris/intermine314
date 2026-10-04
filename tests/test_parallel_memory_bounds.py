"""Deterministic scheduling checks include results blocked behind earlier pages."""

from concurrent.futures import Future

import pytest

from intermine314.query import parallel_offset


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
