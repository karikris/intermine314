from unittest.mock import patch

from intermine314.query import builder as query_builder


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
        self.map_calls = 0
        self.map_buffersizes = []
        self.enter_calls = 0
        self.exit_calls = 0
        _TrackingExecutor.instances.append(self)

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_calls += 1
        return False

    def map(self, fn, iterable, *, buffersize=None):
        self.map_calls += 1
        max_pending = max(1, int(buffersize or 1))
        self.map_buffersizes.append(max_pending)
        pending = []
        source_iter = iter(iterable)
        while True:
            while len(pending) < max_pending:
                try:
                    arg = next(source_iter)
                except StopIteration:
                    break
                self.current_pending += 1
                self.max_pending = max(self.max_pending, self.current_pending)
                self.submitted_offsets.append(arg)
                pending.append(fn(arg))
            if not pending:
                break
            result = pending.pop(0)
            self.current_pending -= 1
            yield result


def test_ordered_mode_early_termination_closes_executor_context():
    fake_query = _FakeQuery()
    _TrackingExecutor.instances.clear()
    with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
        iterator = query_builder.Query._run_parallel_offset(
            fake_query,
            row="dict",
            start=0,
            size=20,
            page_size=1,
            max_workers=4,
            order_mode="ordered",
            inflight_limit=2,
            ordered_window_pages=2,
        )
        first = next(iterator)
        iterator.close()
    assert first["value"] == 0
    instance = _TrackingExecutor.instances[0]
    assert instance.map_calls == 1
    assert instance.map_buffersizes == [2]
    assert instance.max_pending <= 2
    assert instance.enter_calls == 1
    assert instance.exit_calls == 1
