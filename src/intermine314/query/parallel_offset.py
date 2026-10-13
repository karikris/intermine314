from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass
from functools import partial
from itertools import islice

from intermine314.export.runtime import get_runtime
from intermine314.query.inflight import BoundedInflightQueue
from intermine314.query.parallel_runtime import log_parallel_event
from intermine314.service.errors import WebserviceError
from intermine314.service.resource_utils import close_resource_quietly

_PACKAGE_EXECUTOR = ThreadPoolExecutor


@dataclass(frozen=True)
class ValidatedPage:
    index: int
    offset: int
    rows: list
    queue: object

    def batches(self, size):
        for start in range(0, len(self.rows), size):
            yield PageBatch(self, start, min(start + size, len(self.rows)))


@dataclass(frozen=True)
class PageBatch:
    """A bounded view of a validated page, without another row list."""

    page: ValidatedPage
    start: int
    stop: int

    def __len__(self):
        return self.stop - self.start

    def __iter__(self):
        return islice(self.page.rows, self.start, self.stop)


class ParallelExecutionError(RuntimeError):
    """Raised when a parallel page fetch fails."""

    def __init__(self, *, order_mode: str, page_index: int, offset: int):
        self.order_mode = str(order_mode)
        self.page_index = int(page_index)
        self.offset = int(offset)
        super().__init__(
            f"parallel {self.order_mode} fetch failed at page_index={self.page_index} offset={self.offset}"
        )


def run_parallel_pages(
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

    if size == 0:
        return iter(())
    prepare = getattr(query, '_prepare_parallel_execution', None)
    if callable(prepare):
        query = prepare(row=row)

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
        source = query.results(row=row, start=offset, size=limit)
        iterator = None
        try:
            iterator = iter(source)
            rows = list(islice(iterator, limit))
            # Exhaust the concrete stream, so a full page validates its footer
            # without reopening a ResultIterator or draining excess server rows.
            end = object()
            if next(iterator, end) is not end:
                raise WebserviceError(f"Page at offset {offset} returned more rows than requested ({limit})")
            return index, offset, rows
        finally:
            if iterator is not source:
                close_resource_quietly(iterator)
            close_resource_quietly(source)

    def _iter_pages():
        queue = BoundedInflightQueue(
            inflight_limit=int(inflight_limit),
            max_inflight_bytes_estimate=max_inflight_bytes_estimate,
        )
        next_submit = 0
        next_emit = 0
        pending = {}
        completed = {}
        done = set()

        try:
            shared = executor_cls is _PACKAGE_EXECUTOR
            manager = (get_runtime().executor(max_workers, thread_name_prefix) if shared
                       else executor_cls(max_workers=max_workers, thread_name_prefix=thread_name_prefix))
            with manager as executor:
                try:
                    while next_submit < page_count or pending or completed:
                        # Ready pages stay charged throughout emission. Release
                        # them before admitting replacement work.
                        while order_mode == "ordered" and next_emit in completed:
                            rows = completed.pop(next_emit)
                            emitting_index = next_emit
                            next_emit += 1
                            queue.start_emitting(emitting_index)
                            try:
                                page = ValidatedPage(emitting_index, start + emitting_index * page_size, rows, queue)
                                yield page
                            finally:
                                rows = page = None
                                queue.release_page(emitting_index)

                        while next_submit < page_count and queue.can_submit():
                            offset = start + (next_submit * page_size)
                            queue.reserve_page(next_submit)
                            task = fetch_page if shared else partial(copy_context().run, fetch_page)
                            fut = executor.submit(task, next_submit, offset)
                            pending[fut] = (next_submit, offset)
                            next_submit += 1
                        if not pending:
                            continue
                        done = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)[0]
                        while done:
                            fut = done.pop()
                            page_index, failed_offset = pending.pop(fut)
                            try:
                                completed_page_index, _offset, rows = fut.result()
                            except Exception as exc:
                                raise ParallelExecutionError(
                                    order_mode=str(order_mode),
                                    page_index=page_index,
                                    offset=failed_offset,
                                ) from exc
                            fut = None
                            queue.observe_completed_page(page_index=completed_page_index, rows=rows)
                            if order_mode == "unordered":
                                queue.start_emitting(completed_page_index)
                                try:
                                    page = ValidatedPage(completed_page_index, _offset, rows, queue)
                                    yield page
                                finally:
                                    rows = page = None
                                    queue.release_page(completed_page_index)
                            else:
                                completed[completed_page_index] = rows
                            rows = None
                finally:
                    # Cancel work that has not started before the executor waits
                    # for running requests to finish with their transport timeouts.
                    for remaining in tuple(pending):
                        remaining.cancel()
                    pending.clear()
                    completed.clear()
                    done.clear()
                    fut = remaining = rows = page = None
        finally:
            queue.clear()
            log_parallel_event(
                logging.DEBUG,
                "parallel_unified_scheduler_stats",
                job_id=job_id,
                ordered_mode=order_mode,
                in_flight=0,
                inflight_bytes_budget=max_inflight_bytes_estimate,
                **queue.stats_fields(),
            )

    return _iter_pages()


def run_parallel_offset(*args, **kwargs):
    """Compatibility row adapter over the single validated-page scheduler."""
    pages = run_parallel_pages(*args, **kwargs)

    def rows():
        try:
            for page in pages:
                yield from page.rows
                page = None
        finally:
            page = None
            close_resource_quietly(pages)

    return rows()
