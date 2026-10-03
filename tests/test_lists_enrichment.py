"""Pinned enrichment wire, mapping, managed streams and explicit persistence."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections import UserDict
from functools import partial
from pathlib import Path
from urllib.parse import parse_qs

import pytest

from intermine314.service.errors import ServiceError, WebserviceError
from tests.test_lists_crud import assert_closed, client

FIELDS = ["identifier", "description", "p-value", "matches", "populationAnnotationCount"]
ROWS = [
    dict(zip(FIELDS, ["0007", 'α, "label"\nnext', 0.002, 3, 71])),
    dict(zip(FIELDS, ["GO:2", "second", 0.0001, 1, 8])),
    dict(zip(FIELDS, ["GO:3", "third", 0.04, 9, 105])),
]


def payload(rows=ROWS, footer=b'],\n"wasSuccessful":true,\n"statusCode":200,\n"error":null\n}'):
    return b'{\n"results":[\n' + b",\n".join(json.dumps(row, ensure_ascii=False).encode() for row in rows) + (b"\n" if rows else b"") + footer


def enrichment_client(profile="native", version=11, body=None):
    service, session = client(profile, name="list α & β")
    service._version = version
    session.routes[("POST", "/service/list/enrichment")] = payload() if body is None else body
    item = service.get_list("list α & β")
    return service, session, item


def line_type():
    return importlib.import_module("intermine314.results").EnrichmentLine


def test_enrichment_line_exact_userdict_aliases_mutation_and_representation():
    kind = line_type()
    line = kind(ROWS[0])
    assert isinstance(line, UserDict) and not callable(line)
    assert dict(line) == ROWS[0] and list(line) == FIELDS
    assert line.identifier == "0007" and line.p_value == line["p-value"] == 0.002
    assert line.populationAnnotationCount == 71 and line.description == ROWS[0]["description"]
    assert str(line) == str(ROWS[0]) and repr(line) == f"EnrichmentLine({ROWS[0]})"
    line["p-value"] = 0.5
    line.update({"extra-field": "kept"})
    assert line.p_value == 0.5 and line.extra_field == "kept" and line.get("missing") is None
    assert ROWS[0]["p-value"] == 0.002
    with pytest.raises(AttributeError, match="missing"):
        _ = line.missing
    with pytest.raises(AttributeError):
        line.__getattr__(None)


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("custom", [False, True])
def test_enrichment_actual_post_options_and_streamed_five_fields(profile, custom):
    service, session, item = enrichment_client(profile)
    args = ("widget α & x", "background β & y", "Benjamini-Hochberg", 0.01, "filter α & x") if custom else ("widget α & x",)
    stream = item.calculate_enrichment(*args)
    request = session.requests[-1]
    form = parse_qs(request.data.decode(), keep_blank_values=True)
    assert request.method == "POST" and request.path == "/service/list/enrichment"
    assert form == {"list": [item.name], "widget": [args[0]], "correction": ["Benjamini-Hochberg" if custom else "Holm-Bonferroni"], "maxp": ["0.01" if custom else "0.05"], "filter": [args[4] if custom else ""], **({"population": [args[1]]} if custom else {})}
    assert request.headers["Content-Type"] == "application/x-www-form-urlencoded; charset=utf-8"
    assert request.headers["User-Agent"] == "list-client" and request.headers["Authorization"] == "Token secret"
    assert request.options["verify"] == "/custom/ca.pem" and request.options["timeout"][1] == 37
    assert not session.responses[-1].closed
    first = next(stream)
    assert isinstance(first, line_type()) and dict(first) == ROWS[0]
    assert [dict(row) for row in stream] == ROWS[1:]
    assert service.opener._session is session
    assert_closed(session)


@pytest.mark.parametrize("version,background,message", [(7, None, "enrichment requests"), (10, "background", "custom background")])
def test_enrichment_version_gates_before_request(version, background, message):
    _, session, item = enrichment_client(version=version)
    before = len(session.requests)
    with pytest.raises(ServiceError, match=message):
        item.calculate_enrichment("widget", background)
    assert len(session.requests) == before
    assert_closed(session)


def test_enrichment_version_eight_default_and_background_list_string():
    _, session, item = enrichment_client(version=8)
    assert [dict(row) for row in item.calculate_enrichment("widget")] == ROWS
    item._service._version = 11
    list(item.calculate_enrichment("widget", background=item))
    assert parse_qs(session.requests[-1].data.decode())["population"] == [str(item)]
    assert_closed(session)


@pytest.mark.parametrize("body,message", [
    (b"broken", "bad header"),
    (b'{\n"results":[\nnot-json\n', "Error parsing line"),
    (b'{\n"results":[\n', "Connection interrupted"),
    (payload(footer=b'],\n"wasSuccessful":false,\n"statusCode":500,\n"error":"denied"\n}'), "denied"),
    (payload(footer=b"]\ninvalid footer"), "status fragment"),
])
def test_enrichment_protocol_errors_close_response(body, message):
    _, session, item = enrichment_client(body=body)
    with pytest.raises(WebserviceError, match=message):
        list(item.calculate_enrichment("widget"))
    assert_closed(session)


def test_enrichment_empty_and_explicit_early_close():
    _, session, item = enrichment_client(body=payload([]))
    assert list(item.calculate_enrichment("widget")) == []
    session.routes[("POST", "/service/list/enrichment")] = payload()
    stream = item.calculate_enrichment("widget")
    assert dict(next(stream)) == ROWS[0]
    stream.close()
    stream.close()
    assert list(stream) == []
    assert_closed(session)


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["header", "row", "parser"])
def test_enrichment_interrupt_closes_response(monkeypatch, error, stage):
    _, session, item = enrichment_client()
    original = session.request

    def request(*args, **kwargs):
        response = original(*args, **kwargs)
        iterator = response.iter_lines

        def lines(*args, **kwargs):
            for index, line in enumerate(iterator(*args, **kwargs)):
                if index == (0 if stage == "header" else 2):
                    raise error("interrupted")
                yield line

        if stage != "parser":
            response.iter_lines = lines
        return response

    monkeypatch.setattr(session, "request", request)
    with pytest.raises(error, match="interrupted"):
        stream = item.calculate_enrichment("widget")
        if stage == "parser":
            def parser(row):
                raise error("interrupted")
            stream.parser = parser
        next(stream)
    assert_closed(session)


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("format", ["parquet", "csv"])
def test_enrichment_persistence_detached_polars_typed_schema_and_order(tmp_path, monkeypatch, profile, empty, format):
    import polars as pl

    parquet = importlib.import_module("intermine314.export.parquet")
    original = parquet.write_parquet_batches
    sizes = []

    def writer(**kwargs):
        batches = kwargs["batches"]
        def observe():
            for batch in batches:
                sizes.append(len(batch))
                yield batch
        kwargs["batches"] = observe()
        return original(**kwargs)

    module = importlib.import_module("intermine314.lists._enrichment")
    monkeypatch.setattr(module, "write_parquet_batches", writer)
    rows = [] if empty else ROWS
    _, session, item = enrichment_client(profile, body=payload(rows))
    target = tmp_path / ("results.parquet" if format == "parquet" else "results.csv")
    options = {} if format == "parquet" else {"format": "csv"}
    result = item.calculate_enrichment("widget", output_path=target, batch_size=2, **options)
    assert isinstance(result, pl.DataFrame) and result.to_dicts() == rows
    assert result.schema == {"identifier": pl.String, "description": pl.String, "p-value": pl.Float64, "matches": pl.Int64, "populationAnnotationCount": pl.Int64}
    assert sizes == ([] if empty else [2, 1])
    assert target.is_file()
    if format == "parquet":
        assert pl.read_parquet(target).equals(result)
    else:
        assert pl.read_csv(target, schema=dict(result.schema)).equals(result)
    assert list(tmp_path.iterdir()) == [target]
    assert_closed(session)


@pytest.mark.parametrize("options,message", [
    ({"format": "csv"}, "output_path"),
    ({"output_path": "bad.csv"}, "conflicts"),
    ({"output_path": "bad.parquet", "format": "csv"}, "conflicts"),
    ({"output_path": "bad.txt", "format": "json"}, "format"),
    ({"output_path": "bad.parquet", "batch_size": 0}, "batch_size"),
    ({"output_path": "bad.parquet", "batch_size": True}, "batch_size"),
])
def test_enrichment_persistence_options_reject_before_http(options, message):
    _, session, item = enrichment_client()
    # A version lookup would make HTTP; invalid local options must precede it.
    item._service._version = None
    before = len(session.requests)
    with pytest.raises((ValueError, TypeError), match=message):
        item.calculate_enrichment("widget", **options)
    assert len(session.requests) == before
    assert_closed(session)


@pytest.mark.parametrize("fault", ["footer", "schema", "interrupt", "writer", "sql", "arrow", "csv"])
def test_enrichment_persistence_failure_preserves_output_and_cleans(tmp_path, monkeypatch, fault):
    module = importlib.import_module("intermine314.lists._enrichment")
    parquet = importlib.import_module("intermine314.export.parquet")
    query = importlib.import_module("intermine314.export.query")
    import duckdb

    connections = []

    class Connection:
        def __init__(self, **kwargs):
            self.inner = duckdb.connect(**kwargs)
            self.closed = False
            connections.append(self)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def execute(self, sql, parameters=None):
            if fault == "sql" and sql.startswith("SELECT"):
                raise RuntimeError("sql failed")
            self.inner.execute(sql, parameters)
            return self

        def to_arrow_table(self):
            if fault == "arrow":
                raise RuntimeError("arrow failed")
            return self.inner.to_arrow_table()

        def close(self):
            self.closed = True
            self.inner.close()

    monkeypatch.setattr(query, "require_duckdb", lambda api: type("DuckDB", (), {"connect": Connection}))
    body = payload()
    if fault == "footer":
        body = payload(footer=b'],\n"wasSuccessful":false,"error":"denied"\n}')
    elif fault == "schema":
        body = payload([dict(ROWS[0], matches=3.5)])
    _, session, item = enrichment_client(body=body)
    target = tmp_path / "output.txt"
    target.write_bytes(b"previous result")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(module, "TemporaryDirectory", partial(__import__("tempfile").TemporaryDirectory, dir=scratch))
    if fault in ("writer", "interrupt"):
        def fail(*args, **kwargs):
            raise KeyboardInterrupt("writer stopped") if fault == "interrupt" else OSError("writer failed")
        monkeypatch.setattr(parquet, "write_single_parquet_from_parts", fail)
    if fault == "csv":
        def fail(*args, **kwargs):
            raise OSError("csv failed")
        monkeypatch.setattr(module, "write_csv_from_parquet", fail)
    with pytest.raises((WebserviceError, ValueError, KeyboardInterrupt, OSError, RuntimeError)):
        item.calculate_enrichment("widget", output_path=target, format="csv" if fault == "csv" else "parquet", batch_size=1)
    assert target.read_bytes() == b"previous result"
    assert list(scratch.iterdir()) == [] and sorted(path.name for path in tmp_path.iterdir()) == ["output.txt", "scratch"]
    assert all(connection.closed for connection in connections)
    if fault in ("sql", "arrow", "csv"):
        assert len(connections) == 1
    assert_closed(session)


@pytest.mark.parametrize("format", ["parquet", "csv"])
def test_enrichment_publication_interrupt_rolls_back_existing_target(tmp_path, monkeypatch, format):
    parquet = importlib.import_module("intermine314.export.parquet")
    _, session, item = enrichment_client()
    target = tmp_path / "output.txt"
    target.write_bytes(b"previous")
    replace = parquet.os.replace
    interrupted = False

    def interrupt(source, destination):
        nonlocal interrupted
        replace(source, destination)
        if Path(destination) == target and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("published then interrupted")

    monkeypatch.setattr(parquet.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt, match="published then interrupted"):
        item.calculate_enrichment("widget", output_path=target, format=format, batch_size=1)
    assert interrupted and target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]
    assert_closed(session)


def test_enrichment_ordinary_calls_keep_analytics_imports_lazy():
    source = "\n".join([
        "import sys", "from tests.test_lists_enrichment import enrichment_client",
        "_, session, item = enrichment_client()", "list(item.calculate_enrichment('widget'))",
        "assert not any(name in sys.modules for name in ('polars', 'duckdb', 'pyarrow', 'pandas'))",
    ])
    result = subprocess.run([sys.executable, "-c", source], cwd=Path(__file__).resolve().parents[1], env=dict(os.environ), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
