from __future__ import annotations

from itertools import islice

_BYTES_ESTIMATE_SAMPLE_ROWS = 32
_BYTES_ESTIMATE_SAMPLE_VALUES = 32
_BYTES_ESTIMATE_EMA_ALPHA = 0.25


def _estimate_scalar_payload_bytes(value, *, depth=0):
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return 8
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8", errors="ignore"))
    if depth >= 2:
        return min(len(str(value).encode("utf-8", errors="ignore")), 4096)
    if isinstance(value, dict):
        total = 0
        sampled = 0
        for key, item in islice(value.items(), _BYTES_ESTIMATE_SAMPLE_VALUES):
            total += _estimate_scalar_payload_bytes(key, depth=depth + 1)
            total += _estimate_scalar_payload_bytes(item, depth=depth + 1)
            sampled += 1
        if sampled == 0:
            return 0
        if len(value) > sampled:
            total = int(total * (len(value) / float(sampled)))
        return total
    if isinstance(value, (list, tuple)):
        total = 0
        sampled = 0
        for item in islice(value, _BYTES_ESTIMATE_SAMPLE_VALUES):
            total += _estimate_scalar_payload_bytes(item, depth=depth + 1)
            sampled += 1
        if sampled == 0:
            return 0
        if len(value) > sampled:
            total = int(total * (len(value) / float(sampled)))
        return total
    return min(len(str(value).encode("utf-8", errors="ignore")), 4096)


def _estimate_row_payload_bytes(row):
    if isinstance(row, dict):
        total = 0
        sampled = 0
        for key, value in islice(row.items(), _BYTES_ESTIMATE_SAMPLE_VALUES):
            total += _estimate_scalar_payload_bytes(key, depth=1)
            total += _estimate_scalar_payload_bytes(value, depth=1)
            sampled += 1
        if sampled == 0:
            return 0
        if len(row) > sampled:
            total = int(total * (len(row) / float(sampled)))
        return total + 32
    if isinstance(row, (list, tuple)):
        total = 0
        sampled = 0
        for item in islice(row, _BYTES_ESTIMATE_SAMPLE_VALUES):
            total += _estimate_scalar_payload_bytes(item, depth=1)
            sampled += 1
        if sampled == 0:
            return 0
        if len(row) > sampled:
            total = int(total * (len(row) / float(sampled)))
        return total + 32
    return _estimate_scalar_payload_bytes(row, depth=0) + 16


def estimate_rows_payload_bytes(rows):
    try:
        row_count = int(len(rows))
        if row_count <= 0:
            return 0
        sample_count = min(row_count, _BYTES_ESTIMATE_SAMPLE_ROWS)
        sample_total = 0
        for item in rows[:sample_count]:
            sample_total += _estimate_row_payload_bytes(item)
        avg = sample_total / float(sample_count)
        return max(0, int(avg * row_count))
    except Exception:
        return None


class InflightEstimateTracker:
    def __init__(self, *, max_inflight_bytes_estimate):
        self.initial_bytes_limit = (
            int(max_inflight_bytes_estimate) if max_inflight_bytes_estimate is not None else None
        )
        self.bytes_limit = self.initial_bytes_limit
        self.avg_page_bytes = None
        self.max_estimated_inflight_bytes = 0.0
        self.estimator_failures = 0
        self.bytes_cap_hits = 0

    @property
    def bytes_cap_configured(self):
        return self.initial_bytes_limit is not None

    @property
    def bytes_cap_active(self):
        return self.bytes_limit is not None

    def observe_page(self, *, rows):
        if self.bytes_limit is None:
            return None
        estimate = estimate_rows_payload_bytes(rows)
        if estimate is None:
            self.estimator_failures += 1
            self.bytes_limit = None
            return None
        if self.avg_page_bytes is None:
            self.avg_page_bytes = float(estimate)
        else:
            self.avg_page_bytes = (self.avg_page_bytes * (1.0 - _BYTES_ESTIMATE_EMA_ALPHA)) + (
                float(estimate) * _BYTES_ESTIMATE_EMA_ALPHA
            )
        return estimate

    def stats_fields(self):
        return {
            "bytes_cap_configured": self.bytes_cap_configured,
            "bytes_cap_active": self.bytes_cap_active,
            "estimated_page_bytes": (
                int(round(self.avg_page_bytes))
                if self.avg_page_bytes is not None
                else None
            ),
            "max_estimated_inflight_bytes": int(round(self.max_estimated_inflight_bytes)),
            "bytes_cap_hits": int(self.bytes_cap_hits),
            "bytes_estimator_failures": int(self.estimator_failures),
        }


class BoundedInflightQueue:
    """Charge every page until its rows are consumed or discarded.

    Submitted pages reserve the current EMA. Completion replaces that value with
    the page's sampled estimate. This bounds admission, not process RSS: an
    unexpectedly large page can exceed the estimate budget after it completes.
    """

    def __init__(self, *, inflight_limit: int, max_inflight_bytes_estimate: int | None):
        self._inflight_limit = max(1, int(inflight_limit))
        self._tracker = InflightEstimateTracker(max_inflight_bytes_estimate=max_inflight_bytes_estimate)
        self._max_target_pending = 0
        self._reservations = {}
        self._buffered = set()
        self._emitting = set()
        self._peak_outstanding_pages = 0
        self._peak_buffered_pages = 0
        self._peak_estimated_buffered_bytes = 0

    def target_pending(self) -> int:
        target = int(self._inflight_limit)
        if self._tracker.bytes_cap_configured:
            if self._tracker.avg_page_bytes is None or not self._tracker.bytes_cap_active:
                target = 1
            elif self._tracker.bytes_cap_active:
                bytes_budget = max(1, int(self._tracker.bytes_limit or 1))
                estimated_page = max(1, int(round(self._tracker.avg_page_bytes)))
                target = max(1, min(target, bytes_budget // estimated_page))
        self._max_target_pending = max(self._max_target_pending, target)
        return target

    def _estimated_bytes(self, indices=None):
        values = self._reservations.values() if indices is None else (
            self._reservations[index] for index in indices
        )
        return sum(value for value in values if value is not None)

    def _submission_estimate(self):
        if not self._tracker.bytes_cap_configured:
            return 0
        if self._tracker.avg_page_bytes is None or not self._tracker.bytes_cap_active:
            return None
        return max(1, int(round(self._tracker.avg_page_bytes)))

    def can_submit(self) -> bool:
        target = self.target_pending()
        count = len(self._reservations)
        if count >= target:
            if target < self._inflight_limit:
                self._tracker.bytes_cap_hits += 1
            return False
        if not count or not self._tracker.bytes_cap_active:
            return True
        estimate = self._submission_estimate()
        if estimate is None or self._estimated_bytes() + estimate > self._tracker.bytes_limit:
            self._tracker.bytes_cap_hits += 1
            return False
        return True

    def reserve_page(self, page_index: int) -> None:
        self._reservations[page_index] = self._submission_estimate()
        self._record_peaks()

    def observe_completed_page(self, *, page_index: int, rows) -> None:
        estimate = self._tracker.observe_page(rows=rows)
        self._reservations[page_index] = estimate if self._tracker.bytes_cap_configured else 0
        self._buffered.add(page_index)
        self._record_peaks()

    def start_emitting(self, page_index: int) -> None:
        self._buffered.discard(page_index)
        self._emitting.add(page_index)

    def release_page(self, page_index: int) -> None:
        self._reservations.pop(page_index, None)
        self._buffered.discard(page_index)
        self._emitting.discard(page_index)

    def clear(self) -> None:
        self._reservations.clear()
        self._buffered.clear()
        self._emitting.clear()

    def _record_peaks(self) -> None:
        self._peak_outstanding_pages = max(self._peak_outstanding_pages, len(self._reservations))
        self._peak_buffered_pages = max(self._peak_buffered_pages, len(self._buffered))
        self._peak_estimated_buffered_bytes = max(
            self._peak_estimated_buffered_bytes, self._estimated_bytes(self._buffered),
        )
        self._tracker.max_estimated_inflight_bytes = max(
            self._tracker.max_estimated_inflight_bytes, self._estimated_bytes(),
        )

    def stats_fields(self) -> dict[str, int | bool | None]:
        fields = dict(self._tracker.stats_fields())
        fields["queue_inflight_limit"] = int(self._inflight_limit)
        fields["queue_max_target_pending"] = int(self._max_target_pending)
        fields.update(
            outstanding_pages=len(self._reservations),
            buffered_pages=len(self._buffered),
            emitting_pages=len(self._emitting),
            unestimated_pages=sum(value is None for value in self._reservations.values()),
            estimated_inflight_bytes=self._estimated_bytes(),
            estimated_buffered_bytes=self._estimated_bytes(self._buffered),
            peak_outstanding_pages=self._peak_outstanding_pages,
            peak_buffered_pages=self._peak_buffered_pages,
            peak_estimated_buffered_bytes=self._peak_estimated_buffered_bytes,
        )
        return fields
