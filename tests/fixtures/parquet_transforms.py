"""Importable spawn-worker callbacks used by lifecycle tests."""
import os
import time
from pathlib import Path


def add_value(source, destination):
    import duckdb
    import polars as pl
    import pyarrow as pa

    from intermine314.export import query_parquet

    assert pl.thread_pool_size() == pa.cpu_count() == pa.io_thread_count() == 1
    assert duckdb.sql("SELECT current_setting('threads')").fetchone()[0] == 1
    frame = query_parquet(source)
    frame.with_columns((pl.col('value') + 1).alias('value')).write_parquet(destination)


def fail(source, destination):
    Path(destination).write_bytes(b'partial output')
    raise ValueError('injected transform failure')


def crash(source, destination):
    Path(destination).write_bytes(b'partial output')
    os._exit(19)


def stall(source, destination):
    Path(destination).write_bytes(b'partial output')
    time.sleep(60)


def inconsistent(source, destination):
    import polars as pl

    value = int(Path(source).stem.split('-')[-1])
    pl.DataFrame({f'column{value}': [value]}).write_parquet(destination)
