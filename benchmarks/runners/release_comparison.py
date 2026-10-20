"""Common APIs on identical modern dependencies; not stock 0.1.7 installs."""

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
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path("/tmp/intermine-v017-review/sdist/intermine314-0.1.7")


def worker(root, case):
    sys.path.insert(0, str(Path(root) / "src"))
    from intermine314.service.session import ResultIterator

    digest = hashlib.sha256()
    for path in sorted((Path(root) / "src/intermine314").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    source_sha = digest.hexdigest()
    views = [f"Gene.c{i}" for i in range(20)]
    wire = (
        b'{"results":[\n'
        + b",\n".join([json.dumps(list(range(20))).encode()] * 10000)
        + b'\n],"wasSuccessful":true}\n'
    )

    class Opener:
        requests = 0

        def open(self, *args, **kwargs):
            self.requests += 1
            return io.BytesIO(wire)

    opener = Opener()
    service = SimpleNamespace(root="https://offline.invalid", version=8, opener=opener)
    with TemporaryDirectory() as directory:

        def run():
            source = ResultIterator(service, "/query/results", {}, "dict", views)
            if case == "dict_list":
                return len(list(source))
            return sum(row["Gene.c19"] for row in source)

        details = {}
        if case.startswith("export"):
            import polars as pl
            import pyarrow as pa
            import pyarrow.parquet as pq

            pa.set_cpu_count(4)
            pa.set_io_thread_count(2)
            from intermine314.query import Query

            count = 10000 if case == "export_single_numeric" else 100000
            data = [
                {
                    "Gene.id": i,
                    "Gene.value": i * 2
                    if case == "export_single_numeric"
                    else f"group-{i % 100}",
                }
                for i in range(count)
            ]
            query = Query(None)
            query.views = ["Gene.id", "Gene.value"]
            query.iter_batches = lambda **kwargs: (
                data[i : i + 1000] for i in range(0, count, 1000)
            )
            target = Path(directory) / "output"
            single = case == "export_single_numeric"

            def run():
                query.to_parquet(target, batch_size=1000, single_file=single)
                return (
                    pl.scan_parquet(target if single else target / "part-*.parquet")
                    .select(pl.col("Gene.id").sum())
                    .collect()
                    .item()
                )

        run()
        gc.collect()
        before = opener.requests
        start = time.perf_counter_ns()
        value = run()
        elapsed = time.perf_counter_ns() - start
        requests = opener.requests - before
        gc.collect()
        tracemalloc.start()
        assert run() == value
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if case.startswith("export"):
            files = [target] if single else sorted(target.glob("part-*.parquet"))
            actual = pl.read_parquet(files)
            assert actual.rows() == [(r["Gene.id"], r["Gene.value"]) for r in data]
            details = dict(
                data_bytes=sum(p.stat().st_size for p in files),
                parts=len(files),
                row_groups=sum(pq.read_metadata(p).num_row_groups for p in files),
                total_bytes=sum(
                    p.stat().st_size for p in target.rglob("*") if p.is_file()
                )
                if target.is_dir()
                else target.stat().st_size,
            )
        print(
            json.dumps(
                dict(
                    ns=elapsed,
                    python_peak_bytes=peak,
                    requests=requests,
                    result=value,
                    python=sys.version,
                    source_sha256=source_sha,
                    **details,
                )
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--baseline", type=Path, default=OLD)
    parser.add_argument("--candidate", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("/tmp/intermine314-release-comparison"))
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = random.Random(314)
    env = dict(
        os.environ, POLARS_MAX_THREADS="12", ARROW_IO_THREADS="2", PYTHON_JIT="0"
    )
    for case in ("dict_iter", "dict_list", "export_single_numeric", "export_directory"):
        for repetition in range(args.repeats):
            variants = [("0.1.7", args.baseline), ("candidate", args.candidate)]
            rng.shuffle(variants)
            pair = []
            for variant, root in variants:
                output = subprocess.check_output(
                    [sys.executable, __file__, "--worker", str(root), case],
                    env=env,
                    text=True,
                )
                row = dict(
                    case=case,
                    variant=variant,
                    repetition=repetition,
                    **json.loads(output),
                )
                rows.append(row)
                pair.append(row)
            assert pair[0]["result"] == pair[1]["result"]
        # Explicitly unify optional metrics; dict-first Arrow inference drops later keys.
        keys = set().union(*(row.keys() for row in rows))
        pq.write_table(
            pa.Table.from_pylist([{key: row.get(key) for key in keys} for row in rows]),
            args.output / "trials.parquet",
            compression="zstd",
        )
        print(
            case,
            {
                variant: statistics.median(
                    r["ns"]
                    for r in rows
                    if r["case"] == case and r["variant"] == variant
                )
                / 1e6
                for variant in ("0.1.7", "candidate")
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
