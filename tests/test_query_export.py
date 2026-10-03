import csv
from io import StringIO
from pathlib import Path

import polars as pl
import pytest

from intermine314.export import query_parquet
from intermine314.query import ParallelOptions, Query


def remote_query(monkeypatch, rows, columns, profile="native"):
    query = Query(compatibility=profile)
    query.add_view(*columns)
    calls = []
    closed = []

    def batches(**kwargs):
        calls.append(kwargs)
        try:
            for row in rows:
                yield [row]
        finally:
            closed.append(True)

    monkeypatch.setattr(query, "iter_batches", batches)
    return query, calls, closed


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_export_defaults_to_single_parquet_and_preserves_partition_api(tmp_path, monkeypatch, profile):
    rows = [{"Gene.id": "0007"}]
    query, calls, closed = remote_query(monkeypatch, rows, ["Gene.id"], profile)
    target = tmp_path / "result"
    assert query.export(target) == str(target)
    assert target.is_file() and target.read_bytes()[:4] == b"PAR1"
    assert query_parquet(target).to_dicts() == rows
    assert closed == [True]
    parts = tmp_path / "parts"
    query.to_parquet(parts)
    assert parts.is_dir()
    more = tmp_path / "more-parts"
    query.export(more, single_file=False)
    assert more.is_dir() and query_parquet(more).to_dicts() == rows
    assert not list(tmp_path.rglob("*.csv"))


@pytest.mark.parametrize("format,suffix", [("parquet", ".csv"), ("parquet", ".CSV"), ("csv", ".parquet"), ("csv", ".PARQUET"), ("json", ".txt"), (None, ".txt")])
def test_export_format_errors_precede_all_io(tmp_path, monkeypatch, format, suffix):
    query = Query()
    monkeypatch.setattr(query, "iter_batches", lambda **kwargs: pytest.fail("producer started"))
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("directory created"))
    with pytest.raises(ValueError, match="format|suffix"):
        query.export(tmp_path / ("result" + suffix), format=format, temp_dir=tmp_path / "scratch")
    assert list(tmp_path.iterdir()) == []


def test_csv_suffix_does_not_request_csv(tmp_path, monkeypatch):
    query = Query()
    monkeypatch.setattr(query, "iter_batches", lambda **kwargs: pytest.fail("producer started"))
    with pytest.raises(ValueError, match="format|suffix"):
        query.export(tmp_path / "result.csv")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_explicit_csv_streams_quoted_exact_values_and_forwards_controls(tmp_path, monkeypatch, profile):
    columns = ["Gene.id", "Gene.number", "Gene.label"]
    rows = [dict(zip(columns, ["0007", 9007199254740993, 'comma, quote" newline\nÅ'])), dict(zip(columns, ["0002", -9007199254740993, None]))]
    query, calls, closed = remote_query(monkeypatch, rows, columns, profile)
    options = ParallelOptions(page_size=3, max_workers=2)
    target = tmp_path / "literal'[*?].csv"
    scratch = tmp_path / "scratch'[*?]"

    def forbidden(*args, **kwargs):
        pytest.fail("CSV export must not collect a full dataframe")

    monkeypatch.setattr(pl.LazyFrame, "collect", forbidden)
    monkeypatch.setattr(pl, "read_parquet", forbidden)
    assert query.export(target, format="csv", start=3, size=2, batch_size=1, compression="gzip", parallel_options=options, temp_dir=scratch, temp_dir_min_free_bytes=0) == str(target)
    with target.open(newline="", encoding="utf8") as stream:
        assert list(csv.reader(stream)) == [columns, ["0007", "9007199254740993", 'comma, quote" newline\nÅ'], ["0002", "-9007199254740993", ""]]
    assert calls == [{"start": 3, "size": 2, "batch_size": 1, "row_mode": "dict", "parallel_options": options}]
    assert closed == [True]
    assert list(scratch.iterdir()) == []
    assert list(tmp_path.glob("*.csv")) == [target]


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("format", ["parquet", "csv"])
def test_empty_remote_export_preserves_selected_headers(tmp_path, monkeypatch, profile, format):
    columns = ["Gene.id", "Gene.name"]
    query, calls, closed = remote_query(monkeypatch, [], columns, profile)
    target = tmp_path / ("empty." + format)
    query.export(target, format=format, size=0)
    if format == "csv":
        assert list(csv.reader(StringIO(target.read_text()))) == [columns]
    else:
        frame = query_parquet(target)
        assert frame.columns == columns and frame.height == 0
    assert closed == [True]


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("format", ["parquet", "csv"])
def test_export_csv_input_options_pagination_and_borrowed_ownership(tmp_path, monkeypatch, profile, format):
    query = Query(compatibility=profile)
    query.service = object()
    monkeypatch.setattr(query, "iter_batches", lambda **kwargs: pytest.fail("remote fetch"))
    source = StringIO("id;number;label\n0007;9007199254740993;Å\n0002;-9007199254740993;\n")
    options = {"separator": ";", "schema_overrides": {"id": pl.String, "number": pl.Int64, "label": pl.String}}
    target = tmp_path / ("result." + format)
    query.export(target, format=format, start=1, size=1, batch_size=1, csv_input=source, csv_options=options)
    if format == "csv":
        assert list(csv.reader(StringIO(target.read_text()))) == [["id", "number", "label"], ["0002", "-9007199254740993", ""]]
    else:
        assert query_parquet(target).rows() == [("0002", -9007199254740993, None)]
    assert not source.closed
    assert options["schema_overrides"]["id"] == pl.String
    assert sorted(p.name for p in tmp_path.iterdir()) == [target.name]


@pytest.mark.parametrize("format", ["parquet", "csv"])
@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_export_producer_failure_closes_iterator_and_preserves_old_target(tmp_path, monkeypatch, format, failure):
    target = tmp_path / ("result." + format)
    target.write_bytes(b"previous output")
    query = Query()
    query.add_view("Gene.id")
    closed = []

    def batches(**kwargs):
        try:
            yield [{"Gene.id": "0007"}]
            raise failure("injected producer failure")
        finally:
            closed.append(True)

    monkeypatch.setattr(query, "iter_batches", batches)
    scratch = tmp_path / "scratch"
    with pytest.raises(failure, match="injected producer"):
        query.export(target, format=format, batch_size=1, temp_dir=scratch)
    assert closed == [True]
    assert target.read_bytes() == b"previous output"
    assert list(scratch.iterdir()) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == [target.name, "scratch"]


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_csv_copy_failure_closes_connection_and_cleans_temporary_parquet(tmp_path, monkeypatch, failure):
    from contextlib import contextmanager

    from intermine314.export import output

    original = output._connection
    closed = []
    scratch_paths = []

    class Connection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql):
            if sql.startswith("COPY"):
                assert "FORMAT CSV, HEADER true, COMPRESSION none" in sql
                raise failure("injected CSV COPY failure")
            return self.connection.execute(sql)

    @contextmanager
    def connection(scratch):
        scratch_paths.append(Path(scratch))
        with original(scratch) as owned:
            try:
                yield Connection(owned)
            finally:
                closed.append(True)

    monkeypatch.setattr(output, "_connection", connection)
    target = tmp_path / "output.csv"
    target.write_text("previous output")
    with pytest.raises(failure, match="injected CSV COPY"):
        Query().export(target, format="csv", csv_input=StringIO("id\n0007\n"), csv_options={"schema_overrides": {"id": pl.String}})
    assert closed == [True]
    assert target.read_text() == "previous output"
    assert all(not path.exists() for path in scratch_paths)
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("after_rename", [False, True])
@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_csv_publication_rollback_before_and_after_rename(tmp_path, monkeypatch, existing, after_rename, failure):
    from intermine314.export import parquet

    target = tmp_path / "output.csv"
    if existing:
        target.write_text("previous output")
    original = parquet.os.replace
    interrupted = []

    def replace(source, destination):
        if destination == target and not interrupted:
            interrupted.append(True)
            if after_rename:
                original(source, destination)
            raise failure("injected publication failure")
        return original(source, destination)

    monkeypatch.setattr(parquet.os, "replace", replace)
    with pytest.raises(failure, match="injected publication"):
        Query().export(target, format="csv", csv_input=StringIO("id\n0007\n"))
    if existing:
        assert target.read_text() == "previous output"
    else:
        assert not target.exists()
    assert list(tmp_path.iterdir()) == ([target] if existing else [])


@pytest.mark.parametrize("format", ["parquet", "csv"])
@pytest.mark.parametrize("kind", ["directory", "symlink", "dangling_symlink"])
def test_export_rejects_wrong_target_before_producer_and_scratch(tmp_path, monkeypatch, format, kind):
    target = tmp_path / ("output." + format)
    previous = tmp_path / "previous.txt"
    previous.write_text("keep")
    if kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(previous if kind == "symlink" else tmp_path / "missing")
    query = Query()
    monkeypatch.setattr(query, "iter_batches", lambda **kwargs: pytest.fail("producer started"))
    with pytest.raises(ValueError, match="file|symbolic"):
        query.export(target, format=format, temp_dir=tmp_path / "scratch")
    assert previous.read_text() == "keep"
    assert not (tmp_path / "scratch").exists()


@pytest.mark.parametrize("format", ["parquet", "csv"])
@pytest.mark.parametrize("kind", ["path", "hardlink", "borrowed"])
def test_export_rejects_source_output_collision_without_mutating_input(tmp_path, format, kind):
    source = tmp_path / "input.txt"
    source.write_text("id\n0007\n")
    target = tmp_path / "output.txt"
    if kind == "hardlink":
        target.hardlink_to(source)
        csv_input = source
    elif kind == "borrowed":
        target.hardlink_to(source)
        csv_input = source.open()
    else:
        target = source
        csv_input = str(source)
    try:
        with pytest.raises(ValueError, match="same file|overwrite"):
            Query().export(target, format=format, csv_input=csv_input, temp_dir=tmp_path / "scratch")
        assert source.read_text() == "id\n0007\n" and target.read_text() == "id\n0007\n"
        assert not (tmp_path / "scratch").exists()
        if kind == "borrowed":
            assert not csv_input.closed
    finally:
        if kind == "borrowed":
            csv_input.close()


@pytest.mark.parametrize("format", ["parquet", "csv"])
@pytest.mark.parametrize("controls,match", [({"compression": "invalid"}, "compression"), ({"batch_size": 0}, "batch_size"), ({"start": -1}, "start"), ({"size": True}, "size"), ({"csv_options": {}}, "csv_options.*csv_input")])
def test_export_invalid_controls_reject_before_producer_or_output(tmp_path, monkeypatch, format, controls, match):
    query = Query()
    monkeypatch.setattr(query, "iter_batches", lambda **kwargs: pytest.fail("producer started"))
    with pytest.raises((TypeError, ValueError), match=match):
        query.export(tmp_path / "output", format=format, temp_dir=tmp_path / "scratch", **controls)
    assert list(tmp_path.iterdir()) == []


def test_csv_rejects_partition_request_before_io(tmp_path):
    with pytest.raises(ValueError, match="CSV.*single_file"):
        Query().export(tmp_path / "output.csv", format="csv", single_file=False, temp_dir=tmp_path / "scratch")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("format", ["parquet", "csv"])
def test_export_checks_free_bytes_before_producer(tmp_path, monkeypatch, format):
    query = Query()
    monkeypatch.setattr(query, "iter_batches", lambda **kwargs: pytest.fail("producer started"))
    target = tmp_path / "output"
    with pytest.raises(ValueError, match="free bytes"):
        query.export(target, format=format, temp_dir=tmp_path / "scratch", temp_dir_min_free_bytes=2**100)
    assert not target.exists()
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.parametrize("format", ["parquet", "csv"])
def test_export_returns_expanded_written_path(tmp_path, monkeypatch, format):
    requested = "~/task26/result." + format
    actual = tmp_path / "home" / "task26" / ("result." + format)
    original = Path.expanduser
    monkeypatch.setattr(Path, "expanduser", lambda path: actual if str(path) == requested else original(path))
    result = Query().export(requested, format=format, csv_input=StringIO("id\n0007\n"))
    assert result == str(actual) and actual.is_file()


def test_csv_output_stays_uncompressed_with_unknown_suffix_and_parquet_codec(tmp_path):
    target = tmp_path / "output.csv.gz"
    Query().export(target, format="csv", compression="gzip", csv_input=StringIO("id\n0007\n"), csv_options={"schema_overrides": {"id": pl.String}})
    assert target.read_bytes() == b"id\n0007\n"


@pytest.mark.parametrize("empty", [False, True])
def test_csv_typed_input_exact_decimal_date_null_and_empty_page(tmp_path, empty):
    source = StringIO('id,amount,day,label\n0007,12345678901234.0001,2026-10-03,""\n0002,,2026-10-04,\n')
    options = {"schema_overrides": {"id": pl.String, "amount": pl.Decimal(20, 4), "day": pl.Date, "label": pl.String}}
    target = tmp_path / "output.csv"
    Query().export(target, format="csv", size=0 if empty else None, csv_input=source, csv_options=options)
    expected = [["id", "amount", "day", "label"]]
    if not empty:
        expected += [["0007", "12345678901234.0001", "2026-10-03", ""], ["0002", "", "2026-10-04", ""]]
    assert list(csv.reader(StringIO(target.read_text()))) == expected
    assert not source.closed


@pytest.mark.parametrize("content", ["views", "constraint_dict", "uncoded_constraints", "joins", "_sort_order_list"])
def test_export_rejects_csv_input_with_query_content_before_output(tmp_path, content):
    query = Query()
    setattr(query, content, ["remote content"])
    with pytest.raises(ValueError, match="CSV.*query|query.*CSV"):
        query.export(tmp_path / "output.csv", format="csv", csv_input=StringIO("id\n0007\n"), temp_dir=tmp_path / "scratch")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("method", ["export", "to_parquet", "to_duckdb"])
def test_partition_export_cannot_replace_csv_source_disguised_as_managed_part(tmp_path, borrowed, method):
    target = tmp_path / "output"
    target.mkdir()
    source = target / "part-00000.parquet"
    source.write_text("id\n0007\n")
    csv_input = source.open() if borrowed else source
    try:
        with pytest.raises(ValueError, match="same file|overwrite"):
            getattr(Query(), method)(target, single_file=False, csv_input=csv_input)
        assert source.read_text() == "id\n0007\n"
        assert list(target.iterdir()) == [source]
        if borrowed:
            assert not csv_input.closed
    finally:
        if borrowed:
            csv_input.close()
