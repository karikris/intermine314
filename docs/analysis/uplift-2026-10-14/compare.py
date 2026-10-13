"""Isolated checkout comparisons, using the invoking interpreter/dependencies.

python compare.py --baseline /tmp/intermine-uplift-baseline --candidate . --label phase1
Checkout paths must contain src/intermine314. No network operations are used.
"""
from __future__ import annotations

import argparse
import gc
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
    if case in ('template_execution', 'stage_lookup', 'typed_batch', 'parquet_export', 'numeric_export'):
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
    print(json.dumps(dict(ns=elapsed, requests=requests, result=result, python_peak_bytes=peak)))


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
