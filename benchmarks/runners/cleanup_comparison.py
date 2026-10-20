"""Matched, isolated cleanup measurements; no server or native-engine claims."""

import argparse
import hashlib
import io
import json
import random
import subprocess
import sys
import tarfile
import time
import tracemalloc
from pathlib import Path

BASE = "5ea5b765d9edfff04fda3aff82cc9a14bc34e977"


def worker(root, fixture, case):
    sys.path.insert(0, str(Path(root) / "src"))
    if case == "query_xml":
        from intermine314.model import Model
        from intermine314.query import Query

        model = Model(Path(fixture).read_bytes())
        xml = (
            '<query view="Employee.name"><constraint path="Employee.name" op="ONE OF">'
            + "".join(f"<value>gene_{i}</value>" for i in range(2000))
            + "</constraint></query>"
        )

        def operation():
            query = Query.from_xml(xml, model)
            assert len(query.get_constraint("A").values) == 2000
    else:
        from types import SimpleNamespace

        from intermine314.service.session import ResultIterator

        wire = (
            b'{"results":[\n'
            + b",\n".join([b"[1,2,3,4,5,6,7,8]"] * 1000)
            + b'\n],"wasSuccessful":true}\n'
        )

        class Opener:
            def open(self, *args, **kwargs):
                return io.BytesIO(wire)

        service = SimpleNamespace(
            root="https://offline.invalid", version=8, opener=Opener()
        )

        def operation():
            assert (
                sum(
                    row["Gene.c7"]
                    for row in ResultIterator(
                        service,
                        "/query/results",
                        {},
                        "dict",
                        [f"Gene.c{i}" for i in range(8)],
                    )
                )
                == 8000
            )

    digest = hashlib.sha256()
    for p in sorted((Path(root) / "src/intermine314").rglob("*.py")):
        digest.update(str(p.relative_to(root)).encode())
        digest.update(p.read_bytes())
    operation()
    start = time.perf_counter()
    for _ in range(20):
        operation()
    seconds = time.perf_counter() - start
    tracemalloc.start()
    operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    rss_peak_kib = int(status["VmHWM"].split()[0])
    print(
        json.dumps(
            dict(
                case=case,
                iterations=20,
                seconds=seconds,
                python_peak_bytes=peak,
                max_rss_kib=rss_peak_kib,
                source_sha256=digest.hexdigest(),
                python=sys.version,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/intermine314-cleanup/measurements.parquet"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.worker:
        return worker(*args.worker)
    import pyarrow as pa
    import pyarrow.parquet as pq

    baseline = args.output.parent / "baseline-source"
    baseline.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "archive", BASE, "src"], cwd=args.root)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(baseline, filter="data")
    rows = []
    rng = random.Random(314)
    for case in ["query_xml", "result_stream"]:
        for repeat in range(5):
            sources = [("before", baseline), ("after", args.root)]
            rng.shuffle(sources)
            for variant, root in sources:
                raw = subprocess.check_output(
                    [
                        sys.executable,
                        "-I",
                        __file__,
                        "--worker",
                        str(root),
                        str(args.root / "tests/fixtures/compatibility/model.xml"),
                        case,
                    ],
                    text=True,
                    timeout=60,
                )
                rows.append(dict(variant=variant, repeat=repeat, **json.loads(raw)))
    pq.write_table(pa.Table.from_pylist(rows), args.output, compression="zstd")
    for case in ["query_xml", "result_stream"]:
        import statistics

        print(
            case,
            {
                variant: {
                    metric: statistics.median(
                        r[metric]
                        for r in rows
                        if r["case"] == case and r["variant"] == variant
                    )
                    for metric in ["seconds", "python_peak_bytes", "max_rss_kib"]
                }
                for variant in ["before", "after"]
            },
        )


if __name__ == "__main__":
    main()
