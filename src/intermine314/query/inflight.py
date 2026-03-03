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

    @property
    def bytes_cap_configured(self):
        return self.initial_bytes_limit is not None

    @property
    def bytes_cap_active(self):
        return self.bytes_limit is not None

    def observe_page(self, *, max_pending, rows):
        if self.bytes_limit is None:
            return
        estimate = estimate_rows_payload_bytes(rows)
        if estimate is None:
            self.estimator_failures += 1
            self.bytes_limit = None
            return
        if self.avg_page_bytes is None:
            self.avg_page_bytes = float(estimate)
        else:
            self.avg_page_bytes = (self.avg_page_bytes * (1.0 - _BYTES_ESTIMATE_EMA_ALPHA)) + (
                float(estimate) * _BYTES_ESTIMATE_EMA_ALPHA
            )
        self.max_estimated_inflight_bytes = max(
            self.max_estimated_inflight_bytes,
            self.avg_page_bytes * float(max_pending),
        )

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
            "bytes_cap_hits": 0,
            "bytes_estimator_failures": int(self.estimator_failures),
        }


class BoundedInflightQueue:
    """Single bounded in-flight abstraction for parallel scheduling."""

    def __init__(self, *, inflight_limit: int, max_inflight_bytes_estimate: int | None):
        self._inflight_limit = max(1, int(inflight_limit))
        self._tracker = InflightEstimateTracker(max_inflight_bytes_estimate=max_inflight_bytes_estimate)
        self._max_target_pending = 0

    def target_pending(self) -> int:
        target = int(self._inflight_limit)
        if self._tracker.bytes_cap_configured:
            if self._tracker.avg_page_bytes is None:
                target = 1
            elif self._tracker.bytes_cap_active:
                bytes_budget = max(1, int(self._tracker.bytes_limit or 1))
                estimated_page = max(1, int(round(self._tracker.avg_page_bytes)))
                target = max(1, min(target, bytes_budget // estimated_page))
        self._max_target_pending = max(self._max_target_pending, target)
        return target

    def observe_completed_page(self, *, rows, current_pending: int) -> None:
        self._tracker.observe_page(max_pending=max(1, int(current_pending)), rows=rows)

    def stats_fields(self) -> dict[str, int | bool | None]:
        fields = dict(self._tracker.stats_fields())
        fields["queue_inflight_limit"] = int(self._inflight_limit)
        fields["queue_max_target_pending"] = int(self._max_target_pending)
        return fields
