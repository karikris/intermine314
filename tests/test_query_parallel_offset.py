from unittest.mock import patch

from intermine314.query import builder as query_builder
from intermine314.query.parallel_offset import ParallelExecutionError
from concurrent.futures import Future


class _FakeQuery:
    def results(self, row="dict", start=0, size=None):
        stop = start + int(size or 0)
        for value in range(start, stop):
            yield {"value": value, "row": row}


class _TrackingExecutor:
    instances = []

    def __init__(self, *args, **kwargs):
        self.current_pending = 0
        self.max_pending = 0
        self.submitted_offsets = []
        self.submit_calls = 0
        self.enter_calls = 0
        self.exit_calls = 0
        _TrackingExecutor.instances.append(self)

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_calls += 1
        return False

    def submit(self, fn, *args, **kwargs):
        self.submit_calls += 1
        self.current_pending += 1
        self.max_pending = max(self.max_pending, self.current_pending)
        self.submitted_offsets.append(args[1])
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            fut.set_exception(exc)
        self.current_pending -= 1
        return fut


class _FailingQuery:
    def results(self, row="dict", start=0, size=None):
        if start >= 5:
            raise RuntimeError("boom")
        stop = start + int(size or 0)
        for value in range(start, stop):
            yield {"value": value, "row": row}


def test_ordered_mode_early_termination_closes_executor_context():
    fake_query = _FakeQuery()
    _TrackingExecutor.instances.clear()
    with patch("intermine314.query.parallel_offset.ThreadPoolExecutor", _TrackingExecutor):
        iterator = query_builder.Query._run_parallel_offset(
            fake_query,
            row="dict",
            start=0,
            size=20,
            page_size=1,
            max_workers=4,
            order_mode="ordered",
            inflight_limit=2,
        )
        first = next(iterator)
        iterator.close()
    assert first["value"] == 0
    instance = _TrackingExecutor.instances[0]
    assert instance.submit_calls > 0
    assert instance.max_pending <= 2
    assert instance.enter_calls == 1
    assert instance.exit_calls == 1


def test_parallel_failure_surfaces_index_and_offset():
    with patch("intermine314.query.parallel_offset.ThreadPoolExecutor", _TrackingExecutor):
        iterator = query_builder.Query._run_parallel_offset(
            _FailingQuery(),
            row="dict",
            start=0,
            size=10,
            page_size=5,
            max_workers=2,
            order_mode="ordered",
            inflight_limit=2,
        )
        try:
            list(iterator)
            assert False, "expected ParallelExecutionError"
        except ParallelExecutionError as exc:
            assert exc.page_index == 1
            assert exc.offset == 5
