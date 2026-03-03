from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import logging

from intermine314.query.inflight import InflightEstimateTracker
from intermine314.query.parallel_runtime import log_parallel_event


def run_parallel_offset(
    query,
    *,
    row="dict",
    start=0,
    size=None,
    page_size=None,
    max_workers=None,
    order_mode="ordered",
    inflight_limit=None,
    ordered_window_pages=None,
    ordered_max_in_flight=None,
    max_inflight_bytes_estimate=None,
    job_id=None,
    thread_name_prefix: str,
    executor_cls=ThreadPoolExecutor,
):
    del ordered_window_pages
    if size is None:
        total = max(query.count() - start, 0)
    else:
        total = size
    if total <= 0:
        return iter(())
    stop = start + total
    offsets = range(start, stop, page_size)

    def fetch_page(offset):
        limit = min(page_size, stop - offset)
        rows = list(islice(query.results(row=row, start=offset, size=limit), limit))
        return offset, rows

    def _mapped_iterator(*, mode_label, pending_limit):
        submission_buffersize = max(1, int(pending_limit))
        if max_inflight_bytes_estimate is not None:
            submission_buffersize = 1
        cap = InflightEstimateTracker(max_inflight_bytes_estimate=max_inflight_bytes_estimate)

        def fetch_page_with_context(offset):
            try:
                return fetch_page(offset)
            except Exception as exc:
                raise RuntimeError(f"parallel {mode_label} fetch failed at offset={offset}") from exc

        with executor_cls(max_workers=max_workers, thread_name_prefix=thread_name_prefix) as executor:
            for _, rows in executor.map(fetch_page_with_context, offsets, buffersize=submission_buffersize):
                cap.observe_page(max_pending=submission_buffersize, rows=rows)
                for item in rows:
                    yield item

        if mode_label == "ordered":
            log_parallel_event(
                logging.DEBUG,
                "parallel_ordered_scheduler_stats",
                job_id=job_id,
                in_flight=0,
                completed_buffer_size=0,
                max_in_flight=submission_buffersize,
                inflight_bytes_budget=max_inflight_bytes_estimate,
                **cap.stats_fields(),
            )
            return
        log_parallel_event(
            logging.DEBUG,
            "parallel_inflight_cap_stats",
            job_id=job_id,
            ordered_mode=mode_label,
            max_in_flight=submission_buffersize,
            inflight_bytes_budget=max_inflight_bytes_estimate,
            **cap.stats_fields(),
        )

    if order_mode == "unordered":
        return _mapped_iterator(mode_label="unordered", pending_limit=inflight_limit)
    if order_mode in ("window", "mostly_ordered"):
        return _mapped_iterator(mode_label="window", pending_limit=inflight_limit)
    return _mapped_iterator(mode_label="ordered", pending_limit=(ordered_max_in_flight or inflight_limit))
