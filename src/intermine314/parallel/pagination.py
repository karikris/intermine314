import math


def estimate_page_count(*, total_rows: int, page_size: int) -> int:
    if page_size <= 0:
        raise ValueError("page_size must be > 0")
    return max(1, int(math.ceil(total_rows / float(page_size))))


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
