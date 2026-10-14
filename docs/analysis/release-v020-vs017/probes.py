"""Deterministic defect and migration probes against both source distributions."""

import gzip
import importlib
import io
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from xml.etree.ElementTree import fromstring

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = Path("/tmp/intermine-v017-review/sdist/intermine314-0.1.7")


def worker(root):
    sys.path.insert(0, str(Path(root) / "src"))
    from intermine314.query.parallel_offset import run_parallel_offset
    from intermine314.service.session import ResultIterator, _ResponseStreamAdapter

    wire = b'{"results":[\n[1],\n[2]\n],"wasSuccessful":false,"error":"failed terminal status"}\n'
    service = SimpleNamespace(
        root="https://offline.invalid",
        version=8,
        opener=SimpleNamespace(open=lambda *a, **k: io.BytesIO(wire)),
    )
    query = SimpleNamespace(
        results=lambda **kw: ResultIterator(
            service, "/query/results", {}, "dict", ["Gene.id"]
        )
    )
    result = {}
    try:
        result["footer_rows"] = list(
            run_parallel_offset(
                query,
                size=2,
                page_size=2,
                max_workers=1,
                inflight_limit=1,
                thread_name_prefix="review",
            )
        )
        result["footer_rejected"] = False
    except Exception as exc:
        result.update(
            footer_rejected=True,
            footer_exception=type(exc).__name__,
            footer_cause=type(exc.__cause__).__name__,
        )
    import urllib3

    xml = b'<templates><template name="example"/></templates>'
    raw = urllib3.response.HTTPResponse(
        body=io.BytesIO(gzip.compress(xml)),
        headers={"Content-Encoding": "gzip"},
        preload_content=False,
        decode_content=False,
    )
    response = SimpleNamespace(
        raw=raw, iter_lines=lambda **kw: iter(()), close=raw.close
    )
    with _ResponseStreamAdapter(response) as stream:
        body = b"".join(iter(lambda: stream.read(4), b""))
    result["sized_gzip_decoded"] = body == xml
    try:
        fromstring(body)
        result["sized_gzip_parsed"] = True
    except Exception as exc:
        result["sized_gzip_parsed"] = False
        result["sized_gzip_exception"] = type(exc).__name__
    import polars as pl

    from intermine314.query import Query

    with TemporaryDirectory() as temporary:
        target = Path(temporary) / "data"
        target.mkdir()
        previous = target / "part-00000.parquet"
        pl.DataFrame({"Gene.id": [99]}).write_parquet(previous)
        before = previous.read_bytes()

        def batches(**kwargs):
            yield [{"Gene.id": 1}]
            raise RuntimeError("late producer failure")

        query = Query(None)
        query.views = ["Gene.id"]
        query.iter_batches = batches
        try:
            query.to_parquet(target)
        except RuntimeError:
            pass
        result["failed_directory_preserved"] = (
            previous.exists() and previous.read_bytes() == before
        )
    imports = {}
    for module in (
        "intermine314.webservice",
        "intermine314.results",
        "intermine314.lists.listmanager",
        "intermine314.model.class_",
        "intermine314.registry.mines",
        "intermine314.service.tor",
        "intermine314.service.iterators",
    ):
        try:
            importlib.import_module(module)
            imports[module] = True
        except ImportError:
            imports[module] = False
    result["imports"] = imports
    print(json.dumps(result))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker(sys.argv[1])
    else:
        import os

        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = []
        for version, root in [("0.1.7", OLD), ("candidate", ROOT)]:
            output = subprocess.check_output(
                [sys.executable, __file__, str(root)],
                text=True,
                env=dict(os.environ, POLARS_MAX_THREADS="12"),
            )
            rows.append(dict(version=version, result=output))
            print(version, output)
        pq.write_table(
            pa.Table.from_pylist(rows), HERE / "probes.parquet", compression="zstd"
        )
