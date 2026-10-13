"""Isolated checkout comparisons, using the invoking interpreter/dependencies.

python compare.py --baseline /tmp/intermine-uplift-baseline --candidate . --label phase1
Checkout paths must contain src/intermine314. No network operations are used.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def extra_workload(case, stack):
    if case == 'template_execution':
        from intermine314.webservice import Service
        from tests.test_template_execution import TemplateSession, template

        service = stack.enter_context(Service('https://offline.example/service', session=TemplateSession()))
        query = template(service)

        def run():
            return sum(len(query.get_row_list()) for _ in range(100))
        return run
    from intermine314.export.runtime import ResourceRuntime

    runtime = stack.enter_context(ResourceRuntime('cpu16'))
    if case == 'stage_lookup':
        workers = getattr(runtime, '_workers_for', runtime.profile.workers_for)

        def run():
            with runtime.activate():
                return sum(workers('ordered_copy') for _ in range(1000))
        return run
    import polars as pl
    import pyarrow as pa

    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    if case == 'worker_setup':
        from intermine314.export.transforms import _worker

        directory = Path(stack.enter_context(TemporaryDirectory()))
        source = directory / 'input.parquet'
        pl.DataFrame({'value': range(256)}).write_parquet(source)
        output = directory / 'out'
        output.mkdir()

        def run():
            tasks = iter([*( (i, str(source), str(output / f'{i}.parquet')) for i in range(100)), None])
            responses = []
            _worker(SimpleNamespace(get=lambda: next(tasks)), SimpleNamespace(put=responses.append),
                    'shutil:copyfile', 128 * 1024**2)
            assert responses == [(i, None) for i in range(100)]
            return len(responses)
        return run
    if case == 'identifier_body':
        from intermine314.lists._csv import identifier_body

        directory = Path(stack.enter_context(TemporaryDirectory()))
        source = directory / 'identifiers.csv'
        source.write_text('id\n' + ''.join(f'{i:09d}\n' for i in range(100000)))

        def run():
            with runtime.activate(), identifier_body(source, 'id', None) as body:
                return len(body.read())
        return run
    if case == 'http_export':
        from intermine314.query import ParallelOptions
        from intermine314.service import Service
        from tests.fixtures.compatibility import FixtureSession

        body = b'{"results":[\n' + b',\n'.join(json.dumps([str(i), i, True]).encode() for i in range(10000)) + b'\n],"wasSuccessful":true}\n'
        service = stack.enter_context(Service('https://offline.example/service', session=FixtureSession.service(rows=body)))
        query = service.select('Employee.name', 'Employee.age', 'Employee.fullTime')
        directory = Path(stack.enter_context(TemporaryDirectory()))
        options = ParallelOptions(page_size=10000, max_workers=1, inflight_limit=1)

        def run():
            query.to_parquet(directory / 'data', size=10000, batch_size=10000, parallel_options=options, runtime=runtime)
            return pl.scan_parquet(directory / 'data' / 'part-*.parquet').select(pl.len()).collect().item()
        return run
    if case == 'typed_batch':
        from intermine314.export.arrow import typed_batch
        from intermine314.export.model_schema import ModelSchema

        columns = [f'c{i}' for i in range(64)]
        rows = [dict.fromkeys(columns, i) for i in range(1000)]
        schema = ModelSchema(dict.fromkeys(columns, pl.Int64), ())

        def run():
            with runtime.activate():
                return typed_batch(pl, rows, columns, schema).num_rows
        return run
    if case in ('parquet_export', 'numeric_export'):
        from intermine314.export.parquet import write_parquet_batches

        directory = Path(stack.enter_context(TemporaryDirectory()))
        rows = [{'id': i, 'value': str(i) if case == 'parquet_export' else i * 2} for i in range(10000)]

        def run():
            with runtime.activate():
                output = write_parquet_batches(batches=[rows], target=directory / 'data.parquet',
                                              columns=['id', 'value'], polars_module=pl,
                                              compression='zstd', single_file=True)
            return pl.scan_parquet(output).select(pl.len()).collect().item()
        return run
    raise ValueError(case)


def worker(root, case):
    sys.path[:0] = [str(Path(root).resolve() / "src"), str(Path(root).resolve())]
    def source_digest():
        digest = hashlib.sha256()
        for path in sorted((Path(root) / 'src/intermine314').rglob('*.py')):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()
    source_sha256 = source_digest()
    from intermine314.service.session import ResultIterator

    if os.environ.get("UPLIFT_JSON_BACKEND") == "stdlib":
        from intermine314.util import json as backend

        backend._JSON_BACKEND = "json"
        backend._loads = json.loads

    class Opener:
        requests = 0

        def open(self, *args, **kwargs):
            self.requests += 1
            return io.BytesIO(payload)

    views = [f"Gene.field{i}" for i in range(20)]
    payload = b'{"results":[\n' + b',\n'.join(json.dumps(list(range(20))).encode() for _ in range(10000)) + b'\n],"wasSuccessful":true,"statusCode":200}\n'
    opener = Opener()
    service = SimpleNamespace(root="https://offline.invalid", version=8, opener=opener)
    cld = None
    if case == "object_decode":
        from intermine314.model import Model
        from tests.fixtures.compatibility import fixture_bytes

        cld = Model(fixture_bytes("model.xml")).get_class("Employee")
        views = ["Employee.name", "Employee.age"]
        payload = b'{"results":[\n' + b',\n'.join([b'{"name":"A","age":30}'] * 10000) + b'\n],"wasSuccessful":true}\n'

    def run():
        mode = "jsonobjects" if case == "object_decode" else "rr" if case in ("keyed_rows", "retained_rows", "retained_keyed_rows") else "list"
        iterator = ResultIterator(service, "/query/results", {}, mode, views, cld)
        if case == "retained_keyed_rows":
            rows = [row for row in iterator]
            return sum(row["field19"] for row in rows)
        if case == "object_decode":
            return sum(row.age for row in iterator)
        if case == "keyed_rows":
            return sum(row["field19"] for row in iterator)
        if case in ("materialize", "retained_rows"):
            return len(list(iterator))
        return sum(row[19] for row in iterator)

    stack = ExitStack()
    if case in ('template_execution', 'stage_lookup', 'typed_batch', 'parquet_export', 'numeric_export', 'http_export', 'identifier_body', 'worker_setup'):
        run = extra_workload(case, stack)
    run()  # exclude imports and one warmup
    gc.collect()
    before = opener.requests
    start = time.perf_counter_ns()
    result = run()
    elapsed = time.perf_counter_ns() - start
    requests = opener.requests - before
    gc.collect()
    tracemalloc.start()
    run()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stack.close()
    assert source_digest() == source_sha256, 'Runtime changed during measurement'
    print(json.dumps(dict(ns=elapsed, requests=requests, result=result, python_peak_bytes=peak,
                          source_sha256=source_sha256, python=sys.version)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--candidate", default=".")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--cases", nargs="+", default=["materialize", "keyed_rows", "list_decode", "retained_rows", "object_decode"])
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng, rows = random.Random(314), []
    env = dict(os.environ, POLARS_MAX_THREADS="12", PYTHON_JIT="0")
    for case in args.cases:
        env['POLARS_MAX_THREADS'] = '1' if case == 'worker_setup' else '12'
        for repetition in range(args.repetitions):
            variants = [("baseline", args.baseline), ("candidate", args.candidate)]
            rng.shuffle(variants)
            pair = []
            for variant, root in variants:
                start = time.perf_counter_ns()
                output = subprocess.check_output([sys.executable, __file__, "--worker", root, case], env=env, text=True)
                cold_ns = time.perf_counter_ns() - start
                row = dict(case=case, variant=variant, repetition=repetition, root=str(Path(root).resolve()), process_ns=cold_ns, **json.loads(output))
                rows.append(row)
                pair.append(row)
            assert pair[0]["result"] == pair[1]["result"]
        medians = {v: statistics.median(r["ns"] for r in rows if r["case"] == case and r["variant"] == v) / 1e6 for v in ("baseline", "candidate")}
        print(case, medians, flush=True)
    pq.write_table(pa.Table.from_pylist(rows), HERE / f"{args.label}_trials.parquet", compression="zstd")


if __name__ == "__main__":
    main()
