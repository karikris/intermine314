from __future__ import annotations

import logging
import time

from intermine314.util.logging import log_structured_event

PARALLEL_LOG = logging.getLogger("intermine314.query.parallel")


def log_parallel_event(level, event, **fields):
    if not PARALLEL_LOG.isEnabledFor(level):
        return
    log_structured_event(PARALLEL_LOG, level, event, **fields)


def instrument_parallel_iterator(
    iterator,
    *,
    job_id,
    strategy,
    order_mode,
    start,
    size,
    page_size,
    max_workers,
    prefetch,
    inflight_limit,
    max_in_flight,
    ordered_window_pages,
    tor_enabled,
    tor_state_known,
    tor_aware_defaults_applied,
    tor_source,
    max_inflight_bytes_estimate,
):
    started = time.perf_counter()
    yielded_rows = 0
    log_parallel_event(
        logging.INFO,
        "parallel_export_start",
        job_id=job_id,
        strategy=strategy,
        ordered_mode=order_mode,
        start=start,
        size=size,
        page_size=page_size,
        max_workers=max_workers,
        prefetch=prefetch,
        in_flight=inflight_limit,
        max_in_flight=max_in_flight,
        ordered_window_pages=ordered_window_pages,
        tor_enabled=tor_enabled,
        tor_state_known=tor_state_known,
        tor_aware_defaults_applied=tor_aware_defaults_applied,
        tor_source=tor_source,
        max_inflight_bytes_estimate=max_inflight_bytes_estimate,
    )
    try:
        for item in iterator:
            yielded_rows += 1
            yield item
    except Exception as exc:
        log_parallel_event(
            logging.ERROR,
            "parallel_export_error",
            job_id=job_id,
            strategy=strategy,
            ordered_mode=order_mode,
            rows=yielded_rows,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            exception_type=type(exc).__name__,
        )
        raise
    log_parallel_event(
        logging.INFO,
        "parallel_export_done",
        job_id=job_id,
        strategy=strategy,
        ordered_mode=order_mode,
        rows=yielded_rows,
        duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )
