"""Closing or failing a parallel iterator cancels queued work and drops buffers."""

import weakref
from concurrent.futures import Future

import pytest

from intermine314.query import ParallelOptions, Query, parallel_offset
from intermine314.query.inflight import BoundedInflightQueue
from intermine314.query.parallel_offset import ParallelExecutionError


class QueuedExecutor:
    def __init__(self, **kwargs):
        self.tasks = []
        self.exit_calls = 0

    def __enter__(self):
        return self

    def submit(self, fn, index, offset):
        future = Future()
        self.tasks.append((future, fn, index, offset))
        if index == 0:
            future.set_result(fn(index, offset))
        elif index == 1:
            future.set_running_or_notify_cancel()
        return future

    def __exit__(self, *args):
        self.exit_calls += 1
        for future, fn, index, offset in self.tasks:
            if not future.done():
                future.set_result(fn(index, offset))
        return False


@pytest.mark.parametrize("order_mode", ["ordered", "unordered"])
def test_close_cancels_queued_pages_allows_running_page_to_finish_and_clears_reservations(monkeypatch, order_mode):
    executed = []

    class Pages:
        def results(self, start=0, **kwargs):
            executed.append(start)
            return iter([{"value": start}])

    executor = QueuedExecutor()
    queue = BoundedInflightQueue(inflight_limit=20, max_inflight_bytes_estimate=None)
    monkeypatch.setattr(parallel_offset, "BoundedInflightQueue", lambda **kwargs: queue)
    iterator = parallel_offset.run_parallel_offset(
        Pages(), size=20, page_size=1, max_workers=2, inflight_limit=20,
        order_mode=order_mode, thread_name_prefix="close-cancel", executor_cls=lambda **kwargs: executor,
    )
    assert next(iterator) == {"value": 0}
    iterator.close()
    assert executed == [0, 1]
    assert len(executor.tasks) == 20 and executor.exit_calls == 1
    assert all(task[0].cancelled() for task in executor.tasks[2:])
    stats = queue.stats_fields()
    assert stats["outstanding_pages"] == stats["buffered_pages"] == stats["emitting_pages"] == 0
    assert stats["estimated_inflight_bytes"] == 0


def test_public_instrumented_iterator_explicitly_closes_retained_inner_iterator(monkeypatch):
    closed = []

    def rows():
        try:
            yield {"Gene.id": 1}
            yield {"Gene.id": 2}
        finally:
            closed.append(True)

    inner = rows()
    query = Query().select("Gene.id")
    monkeypatch.setattr(query, "_run_parallel_offset", lambda **kwargs: inner)
    iterator = query.run_parallel(size=2, parallel_options=ParallelOptions(page_size=1, max_workers=1))
    assert next(iterator) == {"Gene.id": 1}
    iterator.close()
    assert closed == [True] and inner.gi_frame is None


@pytest.mark.parametrize("order_mode", ["ordered", "unordered"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_worker_failure_clears_all_reservations(monkeypatch, order_mode, failure):
    class Pages:
        def results(self, **kwargs):
            raise failure("worker failure")

    queue = BoundedInflightQueue(inflight_limit=2, max_inflight_bytes_estimate=4096)
    monkeypatch.setattr(parallel_offset, "BoundedInflightQueue", lambda **kwargs: queue)
    iterator = parallel_offset.run_parallel_offset(
        Pages(), size=6, page_size=2, max_workers=2, inflight_limit=2,
        max_inflight_bytes_estimate=4096, order_mode=order_mode, thread_name_prefix="failure-cleanup",
    )
    expected = ParallelExecutionError if failure is RuntimeError else KeyboardInterrupt
    with pytest.raises(expected):
        next(iterator)
    assert queue.stats_fields()["outstanding_pages"] == 0


@pytest.mark.parametrize("failure_site", ["submit", "wait", "estimate"])
def test_scheduler_interrupt_clears_reserved_and_buffered_pages(monkeypatch, failure_site):
    queue = BoundedInflightQueue(inflight_limit=2, max_inflight_bytes_estimate=4096)
    monkeypatch.setattr(parallel_offset, "BoundedInflightQueue", lambda **kwargs: queue)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("scheduler interrupt")

    class Pages:
        def results(self, **kwargs):
            return iter([{"value": 1}])

    executor = QueuedExecutor()
    if failure_site == "submit":
        monkeypatch.setattr(executor, "submit", interrupt)
    elif failure_site == "wait":
        monkeypatch.setattr(parallel_offset, "wait", interrupt)
    else:
        monkeypatch.setattr(queue, "observe_completed_page", interrupt)
    iterator = parallel_offset.run_parallel_offset(
        Pages(), size=6, page_size=1, max_workers=2, inflight_limit=2,
        max_inflight_bytes_estimate=4096, thread_name_prefix="scheduler-interrupt",
        executor_cls=lambda **kwargs: executor,
    )
    with pytest.raises(KeyboardInterrupt, match="scheduler interrupt"):
        next(iterator)
    assert executor.exit_calls == 1
    assert queue.stats_fields()["outstanding_pages"] == 0


@pytest.mark.parametrize("order_mode", ["ordered", "unordered"])
def test_closed_iterator_releases_payload_references(order_mode):
    references = []

    class Row:
        pass

    class Pages:
        def results(self, start=0, size=None, **kwargs):
            for _ in range(size):
                row = Row()
                references.append(weakref.ref(row))
                yield row

    iterator = parallel_offset.run_parallel_offset(
        Pages(), size=8, page_size=2, max_workers=2, inflight_limit=4,
        order_mode=order_mode, thread_name_prefix="payload-release",
    )
    first = next(iterator)
    iterator.close()
    assert first is not None
    del first
    assert references and all(reference() is None for reference in references)
