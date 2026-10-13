"""Reproduce the October 14 optimization audit without changing package code.

Run from the repository root with the development environment. All inputs are
local fixtures; no mine requests are sent. Timings exclude imports and fixture
creation, use randomized warm trials, and are microbenchmarks, not job speedups.
Use separate processes for PYTHON_JIT=0 and PYTHON_JIT=1.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import sysconfig
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("POLARS_MAX_THREADS", "12")


def write_table(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def inventory(destination):
    records, bodies, declarations = [], defaultdict(list), []
    for area in ("src/intermine314", "tests"):
        for path in sorted((ROOT / area).rglob("*.py")):
            source = path.read_text()
            tree = ast.parse(source)
            docs, functions, classes = set(), 0, 0
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes += 1
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if ast.get_docstring(node):
                        docs.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions += 1
                    body = node.body[1:] if ast.get_docstring(node) else node.body
                    if sum(1 for statement in body for _ in ast.walk(statement)) >= 20:
                        signature = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
                        bodies[area, signature].append(dict(path=str(path.relative_to(ROOT)),
                                                           name=node.name, line=node.lineno,
                                                           lines=node.end_lineno - node.lineno + 1))
            records.append(dict(area=area, path=str(path.relative_to(ROOT)),
                                sha256=hashlib.sha256(source.encode()).hexdigest(),
                                bytes=len(source.encode()), lines=len(source.splitlines()),
                                docstring_lines=len(docs), functions=functions, classes=classes))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    declarations.append(dict(path=str(path.relative_to(ROOT)), name=node.name,
                                             line=node.lineno, lines=node.end_lineno - node.lineno + 1))
    duplicates = []
    for (area, signature), members in bodies.items():
        if len(members) > 1:
            for member in members:
                duplicates.append(dict(area=area, group=hashlib.sha256(signature.encode()).hexdigest(),
                                       group_size=len(members), **member))
    write_table(destination / "inventory.parquet", records)
    write_table(destination / "duplicate_bodies.parquet", duplicates)
    write_table(destination / "declarations.parquet", declarations)


def structural_probes(destination):
    """Record actual clone counts and counterexamples to tempting shortcuts."""
    from intermine314.query import Query
    from intermine314.service.session import ResultRow
    from intermine314.webservice import Service
    from tests.test_template_execution import TemplateSession, template

    records = []
    original = Query.clone
    session = TemplateSession()
    with Service("https://offline.example/service", session=session) as service:
        item = template(service)
        for name in ("results", "get_results_list", "rows", "get_row_list"):
            calls = []

            def tracked(query):
                calls.append(type(query).__name__)
                return original(query)

            before = len(session.requests)
            with patch.object(Query, "clone", tracked):
                result = getattr(item, name)(row="list") if name in ("results", "get_results_list") else getattr(item, name)()
                rows = [row for row in result]
            records.append(dict(probe="template_clones", variant=name,
                                detail=json.dumps(dict(clones=len(calls), rows=len(rows),
                                                       requests=len(session.requests) - before))))
    row = ResultRow([1, 2], ["Gene.a", "X.Gene.a"])
    actual, naive = row.to_d(), dict(zip(row.views, row.data))
    assert actual != naive
    records.append(dict(probe="alias_collision", variant="to_d_vs_zip",
                        detail=json.dumps(dict(actual=actual, naive=naive))))
    data = [1, 2]
    iterator = iter(ResultRow(data, ["Gene.a", "Gene.b"]))
    data[0] = 9
    assert list(iterator) == [1, 2]
    records.append(dict(probe="iteration_copy", variant="mutation_after_iter",
                        detail="Existing ResultRow iteration snapshots values; iter(data) would observe mutation."))
    write_table(destination / "structural.parquet", records)


class Opener:
    def __init__(self, payload):
        self.payload = payload
        self.streams = []

    def open(self, *args, **kwargs):
        stream = io.BytesIO(self.payload)
        self.streams.append(stream)
        return stream


def run(destination, repetitions):
    import duckdb
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    from intermine314._result_object import ResultObject
    from intermine314.export import parquet as writer
    from intermine314.export.arrow import typed_batch
    from intermine314.export.layout import can_compact_directory
    from intermine314.export.resource_profile import ResourceProfile
    from intermine314.export.runtime import ResourceRuntime
    from intermine314.service.session import ResultIterator, ResultRow
    from intermine314.util.json import json_backend, json_loads
    from intermine314.webservice import Service
    from tests.fixtures.compatibility import FixtureSession

    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    tag = "jit-on" if sys._jit.is_enabled() else "jit-off"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_table(destination / f"environment-{tag}.parquet", [dict(
        revision=revision, python=sys.version, platform=platform.platform(),
        jit_available=sys._jit.is_available(), jit_enabled=sys._jit.is_enabled(),
        gil_enabled=sys._is_gil_enabled(),
        tail_call=sysconfig.get_config_var("Py_TAIL_CALL_INTERP"),
        affinity=len(os.sched_getaffinity(0)), cpu_budget=16, polars_threads=pl.thread_pool_size(),
        arrow_cpu=pa.cpu_count(), arrow_io=pa.io_thread_count(),
        polars=pl.__version__, arrow=pa.__version__, duckdb=duckdb.__version__,
        json_backend=json_backend(), config_args=sysconfig.get_config_var("CONFIG_ARGS"),
        repetitions=repetitions,
    )])
    columns = [f"Employee.c{i}" for i in range(20)]
    data = list(range(20))
    total = 20000
    common_map = {key: i for i, view in enumerate(columns) for key in (view, view.partition(".")[2])}

    def keyed_rows(shared=False):
        checksum = 0
        for _ in range(total):
            row = ResultRow(data, columns)
            if shared:
                row.index_map = common_map
            checksum += row["c19"]
        return checksum

    def list_rows(direct=False):
        checksum = 0
        for _ in range(total):
            row = list(data) if direct else ResultRow(data, columns).to_l()
            checksum += row[-1]
        return checksum

    payload = json.dumps(["é-000123", 123, 3.25, True, None] * 4).encode()

    def decode_rows(binary=False):
        checksum = 0
        for _ in range(total):
            value = payload if binary else payload.decode("utf-8")
            checksum += len(json_loads(value))
        return checksum

    service = Service("https://offline.example/service", session=FixtureSession.service())
    cld = service.model.get_class("Employee")
    objects = ResultIterator(service, "/query/results", {}, "objects", ["Employee.name"], cld)
    object_payload = {"class": "Employee", "name": "A", "objectId": 1}

    def make_objects(bound=False):
        checksum = 0
        for _ in range(total):
            if bound:
                obj = ResultObject(object_payload, cld, objects.view)
                obj._service = service
            else:
                obj = objects._row_as_object(object_payload)
            checksum += obj.id
        return checksum

    wire = b'{"results":[\n' + b',\n'.join([b'["A",123]'] * 10000) + b'\n],"wasSuccessful":true}\n'
    request_records = []

    def consume_results(concrete=False):
        opener = Opener(wire)
        stub = SimpleNamespace(root="https://offline.example/service", version=8, opener=opener)
        rows = ResultIterator(stub, "/query/results", {}, "list", ["Employee.name", "Employee.age"])
        output = list(iter(rows)) if concrete else list(rows)
        assert len(output) == 10000 and output[-1] == ["A", 123]
        assert all(stream.closed for stream in opener.streams)
        request_records.append(dict(variant="concrete" if concrete else "facade",
                                    requests=len(opener.streams), rows=len(output)))
        return len(output)

    profile = ResourceProfile(cpu_budget=16)
    cached_count = profile.workers_for("arrow_cpu")

    def stage_lookup(cached=False):
        return sum(cached_count if cached else profile.workers_for("arrow_cpu") for _ in range(1000))

    pairs = [
        ("row_key_map", total, lambda: keyed_rows(False), lambda: keyed_rows(True)),
        ("list_parser", total, lambda: list_rows(False), lambda: list_rows(True)),
        ("json_bytes", total, lambda: decode_rows(False), lambda: decode_rows(True)),
        ("object_binding", total, lambda: make_objects(False), lambda: make_objects(True)),
        ("result_length_hint", 10000, lambda: consume_results(False), lambda: consume_results(True)),
        ("stage_lookup", 1000, lambda: stage_lookup(False), lambda: stage_lookup(True)),
    ]
    trials = []
    rng = random.Random(314)

    def measure(name, units, variants):
        expected = None
        for fn in variants.values():
            value = fn()
            if expected is None:
                expected = value
            assert value == expected
        for repeat in range(repetitions):
            names = list(variants)
            rng.shuffle(names)
            for variant in names:
                gc.collect()
                start = time.perf_counter_ns()
                value = variants[variant]()
                elapsed = (time.perf_counter_ns() - start) / 1e9
                assert value == expected
                trials.append(dict(case=name, variant=variant, repeat=repeat, seconds=elapsed,
                                   units=units, jit=tag))

    for name, units, baseline, candidate in pairs:
        measure(name, units, {"current": baseline, "prototype": candidate})
    write_table(destination / f"requests-{tag}.parquet", request_records)

    # Separate retained Python-allocation experiment, not native RSS.
    memory = []
    for shared in (False, True):
        gc.collect()
        tracemalloc.start()
        retained = []
        for _ in range(10000):
            row = ResultRow(data, columns)
            if shared:
                row.index_map = common_map
            assert row["c19"] == 19
            retained.append(row)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory.append(dict(case="retained_row_maps", variant="shared" if shared else "current",
                           rows=len(retained), columns=len(columns), retained_bytes=current,
                           peak_bytes=peak, jit=tag))
        del retained
    write_table(destination / f"memory-{tag}.parquet", memory)

    # One-stage reservation prototype. Pure flat inferred primitives only; this
    # nested same-stage reservation uses the existing supported runtime behavior.
    batch_columns = [f"c{i}" for i in range(64)]
    batch = [{name: row + i for i, name in enumerate(batch_columns)} for row in range(1000)]
    frame = pl.DataFrame({"identifier": [f"id-{i:08}" for i in range(10000)],
                          "value": list(range(10000))})
    original_copy = writer.write_single_parquet_from_parts

    def promote_only(**kwargs):
        parts = list(Path(kwargs["staged_dir"]).glob("part-*.parquet"))
        assert len(parts) == 1 and not kwargs.get("directory", False)
        schema = pl.read_parquet_schema(parts[0])
        assert can_compact_directory(pl, schema)
        # Fixture is below its configured row-group limit and contains only
        # String/Int64. This is not a production eligibility implementation.
        assert pq.ParquetFile(parts[0]).metadata.num_row_groups == 1
        writer._move_or_copy(parts[0], kwargs["target"])

    with TemporaryDirectory(prefix="intermine314-opt-") as scratch, ResourceRuntime(profile) as runtime:
        with runtime.activate():
            def convert(reserve_once=False):
                if reserve_once:
                    with runtime.native("arrow_cpu"):
                        record = typed_batch(pl, batch, batch_columns)
                else:
                    record = typed_batch(pl, batch, batch_columns)
                return (record.num_rows, record.column(63)[999].as_py())

            measure("arrow_reservation", 64000,
                    {"current": lambda: convert(False), "prototype": lambda: convert(True)})

            target = Path(scratch) / "output.parquet"

            def write(promote=False):
                with patch.object(writer, "write_single_parquet_from_parts", promote_only if promote else original_copy):
                    writer.write_parquet_batches(batches=[frame], target=target, columns=frame.columns,
                                                 polars_module=pl, compression="zstd", single_file=True)
                return target.exists()

            measure("single_part_file", frame.height,
                    {"current": lambda: write(False), "prototype": lambda: write(True)})
            for promote in (False, True):
                write(promote)
                with duckdb.connect(config={"threads": 1}) as db:
                    assert db.execute("SELECT count(*), sum(value) FROM read_parquet(?)", [str(target)]).fetchone() == (10000, 49995000)
                assert pl.read_parquet(target).equals(frame)
    service.close()
    write_table(destination / f"trials-{tag}.parquet", trials)
    print(f"{tag}: {len(trials)} trials; equality, request counts, closures, and Parquet readback passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--structural-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory(args.output)
    if not args.inventory_only:
        structural_probes(args.output)
    if not args.inventory_only and not args.structural_only:
        run(args.output, args.repetitions)


if __name__ == "__main__":
    main()
