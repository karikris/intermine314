"""Coarse path-based transformations in explicitly spawned Python workers."""
from __future__ import annotations

import math
import multiprocessing
import os
import queue
import time
from concurrent.futures import CancelledError
from contextlib import contextmanager
from importlib import import_module
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

from intermine314.export._schema import (
    read_parquet_schema_literal,
    validate_duckdb_schema,
)
from intermine314.export.csv import _local_path
from intermine314.export.parquet import _publish_dataset
from intermine314.export.resource_profile import ResourceProfile
from intermine314.export.runtime import (
    _ENGINE_LOCK,
    ResourceRuntime,
    get_runtime,
    resource_operation,
)
from intermine314.parallel.policy import require_positive_int
from intermine314.util.deps import require_polars

_THREAD_ENV = ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'MKL_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'VECLIB_MAXIMUM_THREADS',
               'ARROW_IO_THREADS', 'PYTHON_CPU_COUNT')


@contextmanager
def _spawn_environment():
    # Spawn reimports __main__ before the worker function runs. Set limits before
    # Process.start so even native imports at application module scope see them.
    # Coordinate package engine imports; unrelated application env edits remain
    # the application's responsibility during this short startup interval.
    with _ENGINE_LOCK:
        previous = {key: os.environ.get(key) for key in _THREAD_ENV}
        try:
            os.environ.update(dict.fromkeys(_THREAD_ENV, '1'))
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _worker(inbox, outbox, transform, memory_bytes):
    while task := inbox.get():
        index, source, destination = task
        try:
            # Each worker is an owned process. Native pools are initialized here
            # before importing the user transform; no live engine crosses IPC.
            import duckdb
            import polars as pl
            import pyarrow as pa
            import pyarrow.parquet as pq

            pa.set_cpu_count(1)
            pa.set_io_thread_count(1)
            if pl.thread_pool_size() != 1:
                raise RuntimeError('Spawned transform requires POLARS_MAX_THREADS=1 before native imports')
            duckdb.execute("SET threads=1")
            duckdb.execute(f"SET memory_limit='{max(1, memory_bytes * 3 // 5)}B'")
            spill = Path(destination).parent.parent / f'spill-{os.getpid()}'
            spill.mkdir(exist_ok=True)
            duckdb.execute("SET temp_directory = ?", [str(spill)])
            with pq.ParquetFile(source):
                pass
            profile = ResourceProfile(cpu_budget=1, memory_budget_bytes=memory_bytes,
                                      arrow_cpu_threads=1, arrow_io_threads=1,
                                      duckdb_threads=1, polars_threads=1, temp_dir=str(spill))
            with ResourceRuntime(profile) as runtime, runtime.activate():
                module, name = transform.split(':', 1)
                function = import_module(module)
                for component in name.split('.'):
                    function = getattr(function, component)
                function(source, destination)
                path = Path(destination)
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Transform must write its assigned Parquet file')
                with pq.ParquetFile(path) as parquet:
                    if not len(parquet.schema_arrow):
                        raise ValueError('Transform output must have a typed schema')
            outbox.put((index, None))
        except BaseException as exc:
            outbox.put((index, (type(exc).__name__, str(exc)[:4096])))


def _stop_workers(processes, inbox, outbox, *, successful):
    if successful:
        for _ in processes:
            inbox.put(None)
    else:
        for process in processes:
            if process.is_alive():
                process.terminate()
    for process in processes:
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        process.close()
    for channel in (inbox, outbox):
        if not successful:
            channel.cancel_join_thread()
        channel.close()
        if successful:
            channel.join_thread()


def _run(paths, staged, target, transform, workers, inflight, runtime, timeout, cancelled):
    context = multiprocessing.get_context('spawn')
    inbox, outbox = context.Queue(maxsize=inflight), context.Queue(maxsize=inflight)
    processes = []
    successful = False
    deadline = None if timeout is None else time.monotonic() + timeout
    pending = set()
    exhausted = False
    count = 0
    end = object()

    def check():
        if runtime._closing.is_set() or (cancelled is not None and cancelled.is_set()):
            raise CancelledError('Parquet transformation cancelled')
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Parquet transformation timed out')
        for process in processes:
            if process.exitcode is not None:
                raise RuntimeError(f'Transform worker exited unexpectedly with code {process.exitcode}')

    try:
        check()
        with _spawn_environment():
            for _ in range(workers):
                process = context.Process(target=_worker, args=(inbox, outbox, transform,
                                          max(1, runtime.memory_budget_bytes // workers)), daemon=True)
                process.start()
                processes.append(process)
        while not exhausted or pending:
            check()
            while not exhausted and len(pending) < inflight:
                source = next(paths, end)
                if source is end:
                    exhausted = True
                    break
                source = _local_path(source, 'Transform input').resolve()
                if not source.is_file():
                    raise ValueError('Each transform input must be a local Parquet file')
                if source.is_relative_to(target.resolve()):
                    raise ValueError('Transform output must not replace an input source')
                destination = staged / f'part-{count:05d}.parquet'
                inbox.put((count, str(source), str(destination)))
                pending.add(count)
                count += 1
            if pending:
                try:
                    index, error = outbox.get(timeout=.1)
                except queue.Empty:
                    continue
                if index not in pending:
                    raise RuntimeError('Unexpected transform completion')
                pending.remove(index)
                if error is not None:
                    raise RuntimeError(f'Transform task {index} failed ({error[0]}): {error[1]}')
        successful = True
    finally:
        _stop_workers(processes, inbox, outbox, successful=successful)
    return count


@resource_operation
def transform_parquet(
    paths, output_directory, transform, *, max_workers=None, inflight_limit=None,
    timeout=None, cancel_event=None, resource_profile=None, runtime=None,
):
    """Apply an importable ``module:function(input_path, output_path)`` in spawn workers.

    Each callback writes one Parquet file. Inputs/results stay on disk; IPC sends
    only paths, task IDs and error metadata. The default is the profile's coarse
    Python worker count (4 at cpu16), clipped to its usable CPU budget. Inflight
    tasks default to one per worker and may be raised to two per worker.

    Use an importable main module and an ``if __name__ == '__main__'`` guard.
    Children use one native thread per engine, and package DuckDB connections
    receive a share of the memory envelope. User-created external pools or new
    DuckDB connections must also use one thread. This API is for coarse Python
    transforms; ordinary Arrow/Polars/SQL expressions should use their native pool.

    Every output schema is validated before atomic directory publication; schemas
    must match. Input order determines part order, while callbacks control order
    within their own part. Existing output survives failures, interrupts and
    crashes; unrelated existing directory entries retain publication semantics.
    """
    active = get_runtime()
    if not isinstance(transform, str) or transform.count(':') != 1 or not all(
        component.isidentifier() for section in transform.split(':') for component in section.split('.')
    ):
        raise ValueError('transform must be an importable module:function string')
    requested = active.profile.workers_for('python_processes') if max_workers is None else require_positive_int('max_workers', max_workers)
    workers = min(requested, active.cpu_budget)
    inflight = workers if inflight_limit is None else require_positive_int('inflight_limit', inflight_limit)
    if inflight > workers * 2:
        raise ValueError('inflight_limit must be at most two tasks per worker')
    workers = min(workers, inflight)
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('timeout must be a positive finite number')
    if cancel_event is not None and not callable(getattr(cancel_event, 'is_set', None)):
        raise TypeError('cancel_event must provide is_set()')
    target = _local_path(output_directory, 'Transform output')
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError('Transform output must be a directory')
    if isinstance(paths, (str, Path)):
        raise TypeError('paths must be an iterable of local Parquet files')
    paths = iter(paths)
    sentinel = object()
    first = next(paths, sentinel)
    if first is sentinel:
        raise ValueError('At least one input path is required')
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{target.name}-transform-', dir=target.parent) as scratch:
        staged = Path(scratch).resolve() / 'output'
        staged.mkdir()
        with active.native('python_processes', threads=workers):
            _run(iter(chain([first], paths)), staged, target, transform, workers, inflight, active, timeout, cancel_event)
        pl = require_polars('transform_parquet()')
        schema = None
        for part in staged.glob('part-*.parquet'):
            current = read_parquet_schema_literal(pl, part)
            validate_duckdb_schema(pl, current, 'transform_parquet()')
            if schema is not None and current != schema:
                raise ValueError('Transform output schemas must match')
            schema = current
        _publish_dataset(staged, target, Path(scratch) / 'backup', ordered=True)
    return str(target)
