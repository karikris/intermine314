"""Compressed pages, ordered backpressure, and atomic exports work together."""

import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs

import polars as pl
import pytest

from intermine314.export import query_parquet
from intermine314.query import ParallelOptions, parallel_offset
from intermine314.query.inflight import BoundedInflightQueue
from intermine314.query.parallel_offset import ParallelExecutionError
from intermine314.service import Service
from intermine314.service.errors import WebserviceError
from tests.fixtures.compatibility.http import loopback_service

MODEL = b'''<model name="records" package="org.example">
<class name="Record"><attribute name="identifier" type="java.lang.String"/>
<attribute name="number" type="java.lang.Long"/>
<attribute name="label" type="java.lang.String"/></class></model>'''
TEMPLATES = '''<templates><template name="records" description="Müller 🧬">
<query name="records" model="records" view="Record.identifier Record.number Record.label"/>
</template></templates>'''.encode()
COLUMNS = ["Record.identifier", "Record.number", "Record.label"]
VALUES = [[f"{index:04}", 9007199254740993 + index, 'comma, "quote"\nMüller 🧬'] for index in range(10)]


def wire(rows, successful=True):
    return ('{"results":[\n' + ',\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)
            + '\n],"wasSuccessful":' + str(successful).lower() + ',"error":"failure after rows"}\n').encode()


def routes_for(encoding, page):
    return {
        ("GET", "/service/version/ws"): b"30",
        ("GET", "/service/model"): (MODEL, encoding),
        ("GET", "/service/templates"): (TEMPLATES, encoding),
        ("POST", "/service/query/results"): page,
    }


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
@pytest.mark.parametrize("output", ["parquet", "parts", "csv"])
@pytest.mark.parametrize("failure", [False, True])
def test_http_export_with_delayed_page_and_full_terminal_status(
    tmp_path, monkeypatch, profile, encoding, output, failure,
):
    release_slow = threading.Event()
    slow_started = threading.Event()
    buffered_later = threading.Event()

    class ObservedQueue(BoundedInflightQueue):
        def observe_completed_page(self, *, page_index, rows):
            super().observe_completed_page(page_index=page_index, rows=rows)
            if page_index == 2:
                buffered_later.set()

    queue = ObservedQueue(inflight_limit=2, max_inflight_bytes_estimate=4096)
    monkeypatch.setattr(parallel_offset, "BoundedInflightQueue", lambda **kwargs: queue)

    def page(request):
        params = parse_qs(request.data.decode())
        start, size = int(params["start"][0]), int(params["size"][0])
        if start == 2:
            slow_started.set()
            assert release_slow.wait(timeout=5)
        return wire(VALUES[start:start + size], not (failure and start == 2)), encoding

    target = tmp_path / ("parts" if output == "parts" else "result." + output)
    old = pl.DataFrame(dict(zip(COLUMNS, [["previous"], [-1], ["previous"]])))
    if output == "parts":
        target.mkdir()
        old.write_parquet(target / "part-0.parquet")
    elif output == "parquet":
        old.write_parquet(target)
    else:
        old.write_csv(target)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    scratch = tmp_path / "scratch"
    options = ParallelOptions(page_size=2, max_workers=2, ordered=True, inflight_limit=2, max_inflight_bytes_estimate=4096)
    with loopback_service(routes_for(encoding, page)) as (url, received):
        with Service(url + "/service", compatibility=profile, request_timeout=5) as service:
            assert "Müller 🧬" in service.templates["records"]
            assert service.get_template("records").compatibility == profile
            query = service.select(*COLUMNS).order_by("Record.identifier")
            with ThreadPoolExecutor(max_workers=1) as consumer:
                future = consumer.submit(
                    query.export, target, format="csv" if output == "csv" else "parquet",
                    single_file=output != "parts", size=10, batch_size=1,
                    parallel_options=options, temp_dir=scratch,
                )
                try:
                    assert buffered_later.wait(timeout=5)
                    assert slow_started.wait(timeout=5)
                    stats = queue.stats_fields()
                    assert stats["outstanding_pages"] == 2 and stats["buffered_pages"] == 1
                    assert 0 < stats["estimated_inflight_bytes"] <= 4096
                    offsets = [int(parse_qs(call.data.decode())["start"][0]) for call in received if call.method == "POST"]
                    assert sorted(offsets) == [0, 2, 4]
                finally:
                    release_slow.set()
                if failure:
                    with pytest.raises(ParallelExecutionError) as caught:
                        future.result(timeout=10)
                    assert caught.value.offset == 2 and isinstance(caught.value.__cause__, WebserviceError)
                else:
                    assert future.result(timeout=10) == str(target)
    if failure:
        assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    elif output == "csv":
        with target.open(newline="", encoding="utf8") as stream:
            assert list(csv.reader(stream)) == [COLUMNS, *[[str(value) for value in row] for row in VALUES]]
    else:
        frame = query_parquet(target, 'SELECT * FROM results ORDER BY "Record.identifier"')
        assert frame.rows() == [tuple(row) for row in VALUES]
        assert frame.schema == dict(zip(COLUMNS, [pl.String, pl.Int64, pl.String]))
    assert list(scratch.iterdir()) == []
    assert queue.stats_fields()["outstanding_pages"] == 0
    assert queue.stats_fields()["peak_outstanding_pages"] == 2
    assert not list(tmp_path.glob(".*-publish-*"))


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
def test_unordered_compressed_duckdb_and_dataframe_use_exact_arrow_polars_values(tmp_path, profile, encoding):
    def page(request):
        params = parse_qs(request.data.decode())
        start, size = int(params["start"][0]), int(params["size"][0])
        return wire(VALUES[start:start + size]), encoding

    options = ParallelOptions(page_size=2, max_workers=2, ordered=False, inflight_limit=2, max_inflight_bytes_estimate=4096)
    with loopback_service(routes_for(encoding, page)) as (url, received):
        with Service(url + "/service", compatibility=profile, request_timeout=5) as service:
            query = service.select(*COLUMNS)
            with query.to_duckdb(
                tmp_path / "results.parquet", start=2, size=6, single_file=True,
                batch_size=1, parallel_options=options, managed=True,
            ) as connection:
                frame = pl.from_arrow(connection.execute("SELECT * FROM results").to_arrow_table())
            dataframe = query.dataframe(start=2, size=6)
    assert isinstance(frame, pl.DataFrame)
    assert frame.sort("Record.identifier").rows() == [tuple(row) for row in VALUES[2:8]]
    assert frame.schema == dict(zip(COLUMNS, [pl.String, pl.Int64, pl.String]))
    assert dataframe.sort("Record.identifier").rows() == [tuple(row) for row in VALUES[2:8]]
    assert dataframe.schema == frame.schema
    assert len([call for call in received if call.method == "POST"]) == 4
