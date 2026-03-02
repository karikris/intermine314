from concurrent.futures import Future
from unittest.mock import patch

from intermine314.query import builder as query_builder
import pytest


class _FakeQuery:
    def count(self):
        return 0

    def results(self, row="dict", start=0, size=None):
        stop = start + int(size or 0)
        for value in range(start, stop):
            yield {"value": value, "row": row}


class _WideFakeQuery(_FakeQuery):
    def results(self, row="dict", start=0, size=None):
        stop = start + int(size or 0)
        payload = "X" * 4096
        for value in range(start, stop):
            yield {"value": value, "row": row, "payload": payload}


class _ImmediateFuture(Future):
    def __init__(self, fn, arg, executor):
        super().__init__()
        self._executor = executor
        self._released = False
        try:
            value = fn(arg)
        except Exception as exc:
            self.set_exception(exc)
        else:
            self.set_result(value)

    def result(self, timeout=None):
        if not self._released:
            self._executor.current_pending -= 1
            self._released = True
        return super().result(timeout=timeout)


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

    def submit(self, fn, arg):
        self.current_pending += 1
        self.max_pending = max(self.max_pending, self.current_pending)
        self.submitted_offsets.append(arg)
        return _ImmediateFuture(fn, arg, self)

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


class TestQueryParallelOffset:
    def test_ordered_mode_uses_executor_buffersize_to_bound_pending_tasks(self):
        fake_query = _FakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
            rows = list(
                query_builder.Query._run_parallel_offset(
                    fake_query,
                    row="dict",
                    start=0,
                    size=12,
                    page_size=1,
                    max_workers=4,
                    order_mode="ordered",
                    inflight_limit=3,
                    ordered_window_pages=2,
                )
            )
        assert [row["value"] for row in rows] == list(range(12))
        assert len(_TrackingExecutor.instances) == 1
        instance = _TrackingExecutor.instances[0]
        assert instance.map_calls == 1
        assert instance.map_buffersizes == [3]
        assert instance.max_pending <= 3
        assert instance.submitted_offsets == list(range(12))
        assert instance.enter_calls == 1
        assert instance.exit_calls == 1

    def test_ordered_mode_early_termination_closes_executor_context(self):
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
        assert len(_TrackingExecutor.instances) == 1
        instance = _TrackingExecutor.instances[0]
        assert instance.map_calls == 1
        assert instance.map_buffersizes == [2]
        assert instance.max_pending <= 2
        assert instance.enter_calls == 1
        assert instance.exit_calls == 1

    def test_unordered_mode_limits_pending_tasks_by_inflight_limit(self):
        fake_query = _FakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
            rows = list(
                query_builder.Query._run_parallel_offset(
                    fake_query,
                    row="dict",
                    start=0,
                    size=20,
                    page_size=1,
                    max_workers=8,
                    order_mode="unordered",
                    inflight_limit=3,
                    ordered_window_pages=2,
                )
            )

        assert len(rows) == 20
        assert len(_TrackingExecutor.instances) == 1
        instance = _TrackingExecutor.instances[0]
        assert instance.map_calls == 0
        assert instance.max_pending <= 3

    def test_window_mode_limits_pending_tasks_by_inflight_limit(self):
        fake_query = _FakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
            rows = list(
                query_builder.Query._run_parallel_offset(
                    fake_query,
                    row="dict",
                    start=0,
                    size=20,
                    page_size=1,
                    max_workers=8,
                    order_mode="window",
                    inflight_limit=4,
                    ordered_window_pages=2,
                )
            )

        assert len(rows) == 20
        assert len(_TrackingExecutor.instances) == 1
        instance = _TrackingExecutor.instances[0]
        assert instance.map_calls == 0
        assert instance.max_pending <= 4

    def test_long_ordered_export_memory_regression_guard(self):
        fake_query = _FakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
            row_count = 0
            first_value = None
            last_value = None
            for row in query_builder.Query._run_parallel_offset(
                fake_query,
                row="dict",
                start=0,
                size=5000,
                page_size=1,
                max_workers=8,
                order_mode="ordered",
                inflight_limit=4,
                ordered_window_pages=2,
            ):
                value = row["value"]
                if first_value is None:
                    first_value = value
                last_value = value
                row_count += 1

        assert row_count == 5000
        assert first_value == 0
        assert last_value == 4999
        assert len(_TrackingExecutor.instances) == 1
        instance = _TrackingExecutor.instances[0]
        assert instance.max_pending <= 4

    def test_ordered_mode_exception_has_offset_context(self):
        class _FailingQuery(_FakeQuery):
            def results(self, row="dict", start=0, size=None):
                if start == 7:
                    raise ValueError("boom")
                return super().results(row=row, start=start, size=size)

        with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
            with pytest.raises(RuntimeError) as cm:
                list(
                    query_builder.Query._run_parallel_offset(
                        _FailingQuery(),
                        row="dict",
                        start=0,
                        size=12,
                        page_size=1,
                        max_workers=4,
                        order_mode="ordered",
                        inflight_limit=3,
                        ordered_window_pages=2,
                    )
                )

        assert "offset=7" in str(cm.value)

    def test_ordered_mode_emits_scheduler_stats_fields(self):
        fake_query = _FakeQuery()
        events = []

        def capture_event(level, event, **fields):
            events.append((event, fields))

        with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
            with patch("intermine314.query.builder._log_parallel_event", side_effect=capture_event):
                list(
                    query_builder.Query._run_parallel_offset(
                        fake_query,
                        row="dict",
                        start=0,
                        size=12,
                        page_size=1,
                        max_workers=4,
                        order_mode="ordered",
                        inflight_limit=3,
                        ordered_window_pages=2,
                        ordered_max_in_flight=2,
                        job_id="job_stats_1",
                    )
                )

        scheduler_events = [fields for event, fields in events if event == "parallel_ordered_scheduler_stats"]
        assert len(scheduler_events) == 1
        payload = scheduler_events[0]
        assert "in_flight" in payload
        assert "completed_buffer_size" in payload
        assert "max_in_flight" in payload
        assert payload["job_id"] == "job_stats_1"

    def test_ordered_mode_bytes_cap_limits_pending_tasks(self):
        fake_query = _WideFakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
            rows = list(
                query_builder.Query._run_parallel_offset(
                    fake_query,
                    row="dict",
                    start=0,
                    size=20,
                    page_size=1,
                    max_workers=8,
                    order_mode="ordered",
                    inflight_limit=8,
                    ordered_window_pages=2,
                    max_inflight_bytes_estimate=1024,
                )
            )
        assert len(rows) == 20
        assert len(_TrackingExecutor.instances) == 1
        instance = _TrackingExecutor.instances[0]
        assert instance.map_calls == 0
        assert instance.max_pending <= 1

    def test_ordered_mode_bytes_cap_estimator_failure_falls_back_to_row_cap(self):
        fake_query = _FakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
            with patch("intermine314.query.builder._estimate_rows_payload_bytes", return_value=None):
                rows = list(
                    query_builder.Query._run_parallel_offset(
                        fake_query,
                        row="dict",
                        start=0,
                        size=20,
                        page_size=1,
                        max_workers=8,
                        order_mode="ordered",
                        inflight_limit=4,
                        ordered_window_pages=2,
                        max_inflight_bytes_estimate=1024,
                    )
                )

        assert len(rows) == 20
        assert len(_TrackingExecutor.instances) == 1
        instance = _TrackingExecutor.instances[0]
        assert instance.max_pending > 1
        assert instance.max_pending <= 4

    def test_unordered_mode_bytes_cap_limits_pending_tasks(self):
        fake_query = _WideFakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
            rows = list(
                query_builder.Query._run_parallel_offset(
                    fake_query,
                    row="dict",
                    start=0,
                    size=20,
                    page_size=1,
                    max_workers=8,
                    order_mode="unordered",
                    inflight_limit=8,
                    ordered_window_pages=2,
                    max_inflight_bytes_estimate=1024,
                )
            )

        assert len(rows) == 20
        assert len(_TrackingExecutor.instances) == 1
        instance = _TrackingExecutor.instances[0]
        assert instance.max_pending <= 1

    def test_window_mode_bytes_cap_limits_pending_tasks(self):
        fake_query = _WideFakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
            rows = list(
                query_builder.Query._run_parallel_offset(
                    fake_query,
                    row="dict",
                    start=0,
                    size=20,
                    page_size=1,
                    max_workers=8,
                    order_mode="window",
                    inflight_limit=8,
                    ordered_window_pages=2,
                    max_inflight_bytes_estimate=1024,
                )
            )

        assert len(rows) == 20
        assert len(_TrackingExecutor.instances) == 1
        instance = _TrackingExecutor.instances[0]
        assert instance.max_pending <= 1


class _RunParallelHarness:
    def __init__(self):
        self.offset_calls = []

    def _apply_parallel_profile(self, profile, ordered, large_query_mode):
        resolved_ordered = ordered if ordered is not None else "ordered"
        return profile or "default", resolved_ordered, bool(large_query_mode)

    def _resolve_effective_workers(self, max_workers, size):
        return int(max_workers) if max_workers is not None else 2

    def _normalize_order_mode(self, ordered):
        return str(ordered)

    def _resolve_parallel_strategy(self, pagination, start, size):
        return "offset"

    def _coerce_parallel_options(self, *, parallel_options=None):
        return query_builder.Query._coerce_parallel_options(self, parallel_options=parallel_options)

    def _resolve_parallel_options(self, **kwargs):
        return query_builder.Query._resolve_parallel_options(self, **kwargs)

    def _run_parallel_offset(self, **kwargs):
        self.offset_calls.append(kwargs)
        return iter([{"n": 1}, {"n": 2}])

    def _run_parallel_keyset(self, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("unexpected keyset path in test harness")


class TestRunParallelStructuredLogging:
    def test_legacy_parallel_warning_bridge_is_removed(self):
        assert not hasattr(query_builder, "_warn_legacy_parallel_args")
        assert not hasattr(query_builder, "_LEGACY_PARALLEL_ARGS_WARNING_EMITTED")

    def test_run_parallel_emits_structured_logs_with_job_id(self):
        harness = _RunParallelHarness()
        captured = []

        def fake_log(logger, level, event, **fields):
            captured.append((level, event, fields))
            return {"event": event, **fields}

        with patch("intermine314.query.builder.log_structured_event", side_effect=fake_log):
            with patch("intermine314.query.builder.new_job_id", return_value="qp_test123"):
                with patch("intermine314.query.builder._PARALLEL_LOG.isEnabledFor", return_value=True):
                    options = query_builder.ParallelOptions(
                        page_size=1,
                        max_workers=2,
                        ordered="ordered",
                        prefetch=2,
                        inflight_limit=2,
                        ordered_window_pages=2,
                        profile="default",
                        large_query_mode=False,
                        pagination="offset",
                    )
                    rows = list(
                        query_builder.Query.run_parallel(
                            harness,
                            row="dict",
                            start=0,
                            size=2,
                            parallel_options=options,
                        )
                    )

        assert rows == [{"n": 1}, {"n": 2}]
        assert [event for _, event, _ in captured] == ["parallel_export_start", "parallel_export_done"]
        assert all(fields["job_id"] == "qp_test123" for _, _, fields in captured)
        assert captured[0][2]["in_flight"] == 2
        assert captured[0][2]["max_in_flight"] == 2
        assert captured[1][2]["rows"] == 2
        assert harness.offset_calls[0]["inflight_limit"] == 2

    def test_run_parallel_accepts_parallel_options_value_object(self):
        harness = _RunParallelHarness()
        options = query_builder.ParallelOptions(
            page_size=3,
            max_workers=2,
            ordered="ordered",
            prefetch=2,
            inflight_limit=2,
            ordered_window_pages=2,
            profile="default",
            large_query_mode=False,
            pagination="offset",
            keyset_batch_size=10,
            max_inflight_bytes_estimate=4096,
        )

        rows = list(
            query_builder.Query.run_parallel(
                harness,
                row="dict",
                start=0,
                size=2,
                parallel_options=options,
            )
        )

        assert rows == [{"n": 1}, {"n": 2}]
        assert harness.offset_calls[0]["page_size"] == 3
        assert harness.offset_calls[0]["max_workers"] == 2
        assert harness.offset_calls[0]["inflight_limit"] == 2
        assert harness.offset_calls[0]["max_inflight_bytes_estimate"] == 4096

    def test_run_parallel_invalid_parallel_options_raises_single_exception(self):
        harness = _RunParallelHarness()
        bad = query_builder.ParallelOptions(page_size=0, pagination="offset")

        with pytest.raises(query_builder.ParallelOptionsError) as cm:
            list(query_builder.Query.run_parallel(harness, row="dict", start=0, size=2, parallel_options=bad))

        assert "Invalid parallel options:" in str(cm.value)
        assert "page_size" in str(cm.value)

    def test_run_parallel_invalid_bytes_cap_raises_single_exception(self):
        harness = _RunParallelHarness()
        bad = query_builder.ParallelOptions(max_inflight_bytes_estimate=0, pagination="offset")

        with pytest.raises(query_builder.ParallelOptionsError) as cm:
            list(query_builder.Query.run_parallel(harness, row="dict", start=0, size=2, parallel_options=bad))

        assert "Invalid parallel options:" in str(cm.value)
        assert "max_inflight_bytes_estimate" in str(cm.value)

    def test_run_parallel_rejects_legacy_parallel_keyword_arguments(self):
        harness = _RunParallelHarness()
        with pytest.raises(TypeError):
            list(
                query_builder.Query.run_parallel(
                    harness,
                    row="dict",
                    start=0,
                    size=2,
                    max_workers=3,
                )
            )

    def test_run_parallel_accepts_parallel_options_only(self):
        harness = _RunParallelHarness()
        options = query_builder.ParallelOptions(page_size=2, max_workers=2, pagination="offset")
        rows = list(
            query_builder.Query.run_parallel(
                harness,
                row="dict",
                start=0,
                size=2,
                parallel_options=options,
            )
        )
        assert rows == [{"n": 1}, {"n": 2}]
