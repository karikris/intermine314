"""Calibrate sampled admission against allocated Python objects and Arrow bytes.

This local synthetic measurement is not an RSS bound or a public-mine result.
"""

import gc
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from intermine314.query.inflight import estimate_rows_payload_bytes


def object_bytes(value, seen):
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(object_bytes(k, seen) + object_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(object_bytes(v, seen) for v in value)
    return size


def main():
    results = []
    for repetition in range(7):
        gc.collect()
        tracemalloc.start()
        rows = [dict(id=i, label=f'gene-{i:08d}', length=i % 1000,
                     payload=f'{i:08d}' + 'x' * 120) for i in range(10000)]
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        start = time.perf_counter()
        estimated = estimate_rows_payload_bytes(rows)
        duration = time.perf_counter() - start
        allocated = object_bytes(rows, set())
        arrow = pa.Table.from_pylist(rows)
        assert arrow.num_rows == len(rows) and estimated >= allocated
        results.append(dict(repetition=repetition, rows=len(rows), python=platform.python_version(),
                            estimated_python_bytes=estimated, unique_python_object_bytes=allocated,
                            python_allocation_peak_bytes=peak, arrow_bytes=arrow.nbytes,
                            estimate_seconds=duration, estimate_to_object_ratio=estimated / allocated,
                            fixture='synthetic-flat-uniform', includes_rss=False))
        del rows, arrow
    pq.write_table(pa.Table.from_pylist(results), Path(__file__).with_name('phase4_memory_calibration.parquet'), compression='zstd')


if __name__ == '__main__':
    main()
