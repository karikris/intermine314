import unittest
from concurrent.futures import Future
from unittest.mock import patch

from intermine314.query import builder as query_builder


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
        _TrackingExecutor.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, arg):
        self.current_pending += 1
        self.max_pending = max(self.max_pending, self.current_pending)
        self.submitted_offsets.append(arg)
        return _ImmediateFuture(fn, arg, self)

    def map(self, *args, **kwargs):  # pragma: no cover - defensive guard
        self.map_calls += 1
        raise AssertionError("ordered fallback should not use executor.map without buffersize support")


class TestQueryParallelOffset(unittest.TestCase):
    def test_ordered_mode_bounds_pending_tasks_without_buffersize_support(self):
        fake_query = _FakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder._EXECUTOR_MAP_SUPPORTS_BUFFERSIZE", False):
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

        self.assertEqual([row["value"] for row in rows], list(range(12)))
        self.assertEqual(len(_TrackingExecutor.instances), 1)
        instance = _TrackingExecutor.instances[0]
        self.assertLessEqual(instance.max_pending, 3)
        self.assertEqual(instance.submitted_offsets, list(range(12)))

    def test_long_ordered_export_memory_regression_guard(self):
        fake_query = _FakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder._EXECUTOR_MAP_SUPPORTS_BUFFERSIZE", False):
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

        self.assertEqual(row_count, 5000)
        self.assertEqual(first_value, 0)
        self.assertEqual(last_value, 4999)
        self.assertEqual(len(_TrackingExecutor.instances), 1)
        instance = _TrackingExecutor.instances[0]
        self.assertLessEqual(instance.max_pending, 4)

    def test_ordered_mode_exception_has_offset_context(self):
        class _FailingQuery(_FakeQuery):
            def results(self, row="dict", start=0, size=None):
                if start == 7:
                    raise ValueError("boom")
                return super().results(row=row, start=start, size=size)

        with patch("intermine314.query.builder._EXECUTOR_MAP_SUPPORTS_BUFFERSIZE", False):
            with patch("intermine314.query.builder.ThreadPoolExecutor", _TrackingExecutor):
                with self.assertRaises(RuntimeError) as cm:
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

        self.assertIn("offset=7", str(cm.exception))

    def test_ordered_mode_emits_scheduler_stats_fields(self):
        fake_query = _FakeQuery()
        events = []

        def capture_event(level, event, **fields):
            events.append((event, fields))

        with patch("intermine314.query.builder._EXECUTOR_MAP_SUPPORTS_BUFFERSIZE", False):
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
        self.assertEqual(len(scheduler_events), 1)
        payload = scheduler_events[0]
        self.assertIn("in_flight", payload)
        self.assertIn("completed_buffer_size", payload)
        self.assertIn("max_in_flight", payload)
        self.assertEqual(payload["job_id"], "job_stats_1")

    def test_ordered_mode_bytes_cap_limits_pending_tasks(self):
        fake_query = _WideFakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder._EXECUTOR_MAP_SUPPORTS_BUFFERSIZE", False):
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

        self.assertEqual(len(rows), 20)
        self.assertEqual(len(_TrackingExecutor.instances), 1)
        instance = _TrackingExecutor.instances[0]
        self.assertLessEqual(instance.max_pending, 1)

    def test_ordered_mode_bytes_cap_estimator_failure_falls_back_to_row_cap(self):
        fake_query = _FakeQuery()
        _TrackingExecutor.instances.clear()

        with patch("intermine314.query.builder._EXECUTOR_MAP_SUPPORTS_BUFFERSIZE", False):
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

        self.assertEqual(len(rows), 20)
        self.assertEqual(len(_TrackingExecutor.instances), 1)
        instance = _TrackingExecutor.instances[0]
        self.assertGreater(instance.max_pending, 1)
        self.assertLessEqual(instance.max_pending, 4)


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

    def _coerce_parallel_options(self, **kwargs):
        return query_builder.Query._coerce_parallel_options(self, **kwargs)

    def _resolve_parallel_options(self, **kwargs):
        return query_builder.Query._resolve_parallel_options(self, **kwargs)

    def _run_parallel_offset(self, **kwargs):
        self.offset_calls.append(kwargs)
        return iter([{"n": 1}, {"n": 2}])

    def _run_parallel_keyset(self, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("unexpected keyset path in test harness")


class TestRunParallelStructuredLogging(unittest.TestCase):
    def test_run_parallel_emits_structured_logs_with_job_id(self):
        harness = _RunParallelHarness()
        captured = []

        def fake_log(logger, level, event, **fields):
            captured.append((level, event, fields))
            return {"event": event, **fields}

        with patch("intermine314.query.builder.log_structured_event", side_effect=fake_log):
            with patch("intermine314.query.builder.new_job_id", return_value="qp_test123"):
                with patch("intermine314.query.builder._PARALLEL_LOG.isEnabledFor", return_value=True):
                    rows = list(
                        query_builder.Query.run_parallel(
                            harness,
                            row="dict",
                            start=0,
                            size=2,
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
                    )

        self.assertEqual(rows, [{"n": 1}, {"n": 2}])
        self.assertEqual([event for _, event, _ in captured], ["parallel_export_start", "parallel_export_done"])
        self.assertTrue(all(fields["job_id"] == "qp_test123" for _, _, fields in captured))
        self.assertEqual(captured[0][2]["in_flight"], 2)
        self.assertEqual(captured[0][2]["max_in_flight"], 2)
        self.assertEqual(captured[1][2]["rows"], 2)
        self.assertEqual(harness.offset_calls[0]["inflight_limit"], 2)

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

        self.assertEqual(rows, [{"n": 1}, {"n": 2}])
        self.assertEqual(harness.offset_calls[0]["page_size"], 3)
        self.assertEqual(harness.offset_calls[0]["max_workers"], 2)
        self.assertEqual(harness.offset_calls[0]["inflight_limit"], 2)
        self.assertEqual(harness.offset_calls[0]["max_inflight_bytes_estimate"], 4096)

    def test_run_parallel_accepts_legacy_max_inflight_bytes_argument(self):
        harness = _RunParallelHarness()

        rows = list(
            query_builder.Query.run_parallel(
                harness,
                row="dict",
                start=0,
                size=2,
                page_size=3,
                max_workers=2,
                ordered="ordered",
                prefetch=2,
                inflight_limit=2,
                max_inflight_bytes_estimate=2048,
                ordered_window_pages=2,
                profile="default",
                large_query_mode=False,
                pagination="offset",
            )
        )

        self.assertEqual(rows, [{"n": 1}, {"n": 2}])
        self.assertEqual(harness.offset_calls[0]["max_inflight_bytes_estimate"], 2048)

    def test_run_parallel_invalid_parallel_options_raises_single_exception(self):
        harness = _RunParallelHarness()
        bad = query_builder.ParallelOptions(page_size=0, pagination="offset")

        with self.assertRaises(query_builder.ParallelOptionsError) as cm:
            list(query_builder.Query.run_parallel(harness, row="dict", start=0, size=2, parallel_options=bad))

        self.assertIn("Invalid parallel options:", str(cm.exception))
        self.assertIn("page_size", str(cm.exception))

    def test_run_parallel_invalid_bytes_cap_raises_single_exception(self):
        harness = _RunParallelHarness()
        bad = query_builder.ParallelOptions(max_inflight_bytes_estimate=0, pagination="offset")

        with self.assertRaises(query_builder.ParallelOptionsError) as cm:
            list(query_builder.Query.run_parallel(harness, row="dict", start=0, size=2, parallel_options=bad))

        self.assertIn("Invalid parallel options:", str(cm.exception))
        self.assertIn("max_inflight_bytes_estimate", str(cm.exception))

    def test_run_parallel_warns_when_legacy_parallel_args_are_used(self):
        harness = _RunParallelHarness()
        with patch("intermine314.query.builder._LEGACY_PARALLEL_ARGS_WARNING_EMITTED", False):
            with patch("intermine314.query.builder._PARALLEL_LOG.warning") as warning_mock:
                list(
                    query_builder.Query.run_parallel(
                        harness,
                        row="dict",
                        start=0,
                        size=2,
                        max_workers=3,
                    )
                )

        warning_mock.assert_called_once()
        self.assertIn("parallel_options=ParallelOptions(...)", warning_mock.call_args[0][0])

    def test_run_parallel_does_not_warn_for_parallel_options_only(self):
        harness = _RunParallelHarness()
        options = query_builder.ParallelOptions(page_size=2, max_workers=2, pagination="offset")
        with patch("intermine314.query.builder._LEGACY_PARALLEL_ARGS_WARNING_EMITTED", False):
            with patch("intermine314.query.builder._PARALLEL_LOG.warning") as warning_mock:
                list(
                    query_builder.Query.run_parallel(
                        harness,
                        row="dict",
                        start=0,
                        size=2,
                        parallel_options=options,
                    )
                )

        warning_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
