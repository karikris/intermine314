from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
import logging

from intermine314.query.inflight import BoundedInflightQueue
from intermine314.query.parallel_runtime import log_parallel_event


class ParallelExecutionError(RuntimeError):
    """Raised when a parallel page fetch fails."""

    def __init__(self, *, order_mode: str, page_index: int, offset: int):
        self.order_mode = str(order_mode)
        self.page_index = int(page_index)
        self.offset = int(offset)
        super().__init__(
            f"parallel {self.order_mode} fetch failed at page_index={self.page_index} offset={self.offset}"
        )


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
    max_inflight_bytes_estimate=None,
    job_id=None,
    thread_name_prefix: str,
    executor_cls=ThreadPoolExecutor,
):
    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError("page_size must be a positive integer")

    if size is None:
        total = max(query.count() - start, 0)
    else:
        total = size
    if total <= 0:
        return iter(())
    start = int(start)
    total = int(total)
    stop = start + total
    page_count = (total + page_size - 1) // page_size

    def fetch_page(index: int, offset: int):
        limit = min(page_size, stop - offset)
        rows = list(islice(query.results(row=row, start=offset, size=limit), limit))
        return index, offset, rows

    def _iter_pages():
        queue = BoundedInflightQueue(
            inflight_limit=int(inflight_limit),
            max_inflight_bytes_estimate=max_inflight_bytes_estimate,
        )
        next_submit = 0
        next_emit = 0
        pending = {}
        completed = {}

        with executor_cls(max_workers=max_workers, thread_name_prefix=thread_name_prefix) as executor:
            while next_submit < page_count or pending:
                while next_submit < page_count and len(pending) < queue.target_pending():
                    offset = start + (next_submit * page_size)
                    fut = executor.submit(fetch_page, next_submit, offset)
                    pending[fut] = (next_submit, offset)
                    next_submit += 1
                if not pending:
                    continue
                done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    page_index, failed_offset = pending.pop(fut)
                    try:
                        completed_page_index, _offset, rows = fut.result()
                    except Exception as exc:
                        for remaining in tuple(pending.keys()):
                            remaining.cancel()
                        pending.clear()
                        raise ParallelExecutionError(
                            order_mode=str(order_mode),
                            page_index=page_index,
                            offset=failed_offset,
                        ) from exc
                    queue.observe_completed_page(rows=rows, current_pending=(len(pending) + 1))
                    if order_mode == "unordered":
                        for item in rows:
                            yield item
                    else:
                        completed[completed_page_index] = rows

                while order_mode == "ordered" and next_emit in completed:
                    rows = completed.pop(next_emit)
                    next_emit += 1
                    for item in rows:
                        yield item

        event_name = "parallel_unified_scheduler_stats"
        log_parallel_event(
            logging.DEBUG,
            event_name,
            job_id=job_id,
            ordered_mode=order_mode,
            in_flight=0,
            inflight_bytes_budget=max_inflight_bytes_estimate,
            **queue.stats_fields(),
        )

    return _iter_pages()
