from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timed():
    start = perf_counter()
    payload = {"seconds": 0.0}
    try:
        yield payload
    finally:
        payload["seconds"] = perf_counter() - start
