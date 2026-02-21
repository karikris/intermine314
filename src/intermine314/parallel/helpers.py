from __future__ import annotations

import math

from intermine314.parallel.policy import VALID_ORDER_MODES


def normalize_mode(mode):
    if isinstance(mode, bool):
        return "ordered" if mode else "unordered"
    text = str(mode).strip().lower()
    if text not in VALID_ORDER_MODES:
        raise ValueError("mode must be ordered|unordered|window|mostly_ordered")
    return text


def estimate_page_count(*, total_rows: int, page_size: int) -> int:
    if page_size <= 0:
        raise ValueError("page_size must be > 0")
    rows = max(0, int(total_rows))
    return int(math.ceil(rows / float(page_size))) if rows else 0


def iter_offset_pages(*, start: int, size: int | None, page_size: int):
    if page_size <= 0:
        raise ValueError("page_size must be > 0")
    offset = max(0, int(start))
    if size is None:
        while True:
            yield offset, page_size
            offset += page_size
    else:
        remaining = max(0, int(size))
        while remaining > 0:
            chunk = min(page_size, remaining)
            yield offset, chunk
            remaining -= chunk
            offset += chunk


__all__ = ["normalize_mode", "estimate_page_count", "iter_offset_pages"]
