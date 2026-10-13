"""Reproduce conversion call profiles and representation sizes as Parquet.

Run after timed benchmarks, not concurrently. cProfile perturbs Python costs;
its times locate work and must not be substituted for uninstrumented timings.
"""
from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from run_benchmarks import COLUMNS, rows_fixture

from intermine314.export.model_schema import ModelSchema, model_batch_frame
from intermine314.export.parquet import _batch_frame
from intermine314.query.inflight import estimate_rows_payload_bytes

HERE = Path(__file__).resolve().parent


def deep_bytes(obj, seen):
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    total = sys.getsizeof(obj)
    if isinstance(obj, dict):
        total += sum(deep_bytes(key, seen) + deep_bytes(value, seen) for key, value in obj.items())
    elif isinstance(obj, (list, tuple)):
        total += sum(deep_bytes(value, seen) for value in obj)
    return total


def main():
    rows = rows_fixture()
    schema = ModelSchema(dict(zip(COLUMNS, [pl.Int64, pl.String, pl.Int32, pl.Float64,
                                           pl.String, pl.Int32, pl.String, pl.String])), ())
    records = []
    for name, action in [('inferred', lambda: _batch_frame(pl, rows, COLUMNS)),
                         ('model', lambda: model_batch_frame(pl, rows, COLUMNS, schema))]:
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(5):
            action()
        profile.disable()
        for (filename, line, function), values in pstats.Stats(profile).stats.items():
            primitive, calls, self_s, cumulative_s, _ = values
            records.append(dict(stage=name, file=filename, line=line, function=function,
                                primitive_calls=primitive, calls=calls, self_s=self_s,
                                cumulative_s=cumulative_s))
    pq.write_table(pa.Table.from_pylist(records), HERE/'conversion_profile.parquet', compression='zstd')
    frame = model_batch_frame(pl, rows, COLUMNS, schema)
    sizes = [dict(rows=len(rows), python_deep_bytes=deep_bytes(rows, set()),
                  payload_estimate_bytes=estimate_rows_payload_bytes(rows),
                  polars_estimated_bytes=frame.estimated_size(), arrow_nbytes=frame.to_arrow().nbytes,
                  notes='Deep Python size counts unique reachable objects once; native buffers are separate; synthetic flat fixture, not RSS or arbitrary nested data.')]
    pq.write_table(pa.Table.from_pylist(sizes), HERE/'memory_comparison.parquet', compression='zstd')
    print(f'Profiled two conversion paths; stored sizes for {len(rows)} rows.')


if __name__ == '__main__':
    main()
