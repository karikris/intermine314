from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from io import BytesIO, StringIO
from pathlib import Path
from types import MappingProxyType

import polars as pl
import pytest


def import_csv(*args, **kwargs):
    return importlib.import_module("intermine314.export").import_csv(*args, **kwargs)


def read_output(path):
    return importlib.import_module("intermine314.export").query_parquet(path)


def test_public_import_signature_and_imports_are_lazy():
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; from intermine314.export import import_csv; "
                "print(json.dumps(sorted(m for m in sys.modules "
                "if any(m==p or m.startswith(p+'.') "
                "for p in ('pandas','polars','duckdb','pyarrow')))))"
            ),
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == []
    exports = importlib.import_module("intermine314.export")
    assert "import_csv" in exports.__all__
    assert "import_csv" in dir(exports)
    assert str(inspect.signature(exports.import_csv)) == (
        "(csv_input, parquet_path, *, csv_options=None)"
    )


@pytest.mark.parametrize("source_kind", ["path", "string", "text", "binary", "file"])
def test_typed_csv_preserves_values_and_borrowed_streams(
    tmp_path, csv_input_factory, source_kind
):
    text = csv_input_factory().getvalue()
    path = tmp_path / "input.txt"
    path.write_text(text, encoding="utf-8")
    source = {
        "path": lambda: path,
        "string": lambda: str(path),
        "text": lambda: StringIO(text),
        "binary": lambda: BytesIO(text.encode()),
        "file": lambda: path.open("r", encoding="utf-8"),
    }[source_kind]()
    schema = {
        "identifier": pl.String,
        "name": pl.String,
        "large_integer": pl.Int64,
        "active": pl.Boolean,
        "amount": pl.Decimal(20, 4),
        "observed_at": pl.Datetime("us"),
    }
    options = MappingProxyType({"schema_overrides": schema, "try_parse_dates": True})
    target = tmp_path / "nested" / "output.parquet"
    try:
        assert import_csv(source, target, csv_options=options) == str(target)
        result = read_output(target)
        assert result.schema == schema
        assert result.rows() == [
            (
                "0007",
                "Müller, Ada",
                9007199254740993,
                True,
                Decimal("123456789.1234"),
                datetime(2026, 1, 2, 3, 4, 5),
            ),
            ("0002", None, 9007199254740995, False, None, datetime(2026, 1, 1)),
            ("0010", "line\nbreak", -9007199254740993, None, None, None),
        ]
        assert options == {"schema_overrides": schema, "try_parse_dates": True}
        assert list(target.parent.iterdir()) == [target]
        if hasattr(source, "closed"):
            assert not source.closed
    finally:
        if source_kind == "file":
            source.close()


def test_scan_and_sink_are_direct_without_dataframe_collection_or_csv_staging(
    tmp_path, monkeypatch
):
    calls = []
    scan, sink = pl.scan_csv, pl.LazyFrame.sink_parquet

    def scan_csv(source, **options):
        calls.append(("scan", source, options))
        return scan(source, **options)

    def sink_parquet(frame, path, **options):
        calls.append(("sink", path))
        assert sorted(p.name for p in Path(path).parent.iterdir()) == []
        return sink(frame, path, **options)

    def forbidden(*args, **kwargs):
        pytest.fail("CSV import must scan directly into the Parquet sink")

    monkeypatch.setattr(pl, "scan_csv", scan_csv)
    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", sink_parquet)
    monkeypatch.setattr(pl.LazyFrame, "collect", forbidden)
    monkeypatch.setattr(pl, "read_csv", forbidden)
    monkeypatch.setattr(pl, "read_parquet", forbidden)
    target = tmp_path / "output.parquet"
    import_csv(StringIO("id\n0007\n"), target, csv_options={"infer_schema": False})
    assert [call[0] for call in calls] == ["scan", "sink"]
    assert calls[0][2]["glob"] is False
    assert calls[1][1] != target
    assert list(tmp_path.iterdir()) == [target]
    assert read_output(target).item() == "0007"


def test_csv_options_separator_header_quote_null_encoding_and_new_columns(tmp_path):
    options = {
        "separator": ";",
        "has_header": False,
        "quote_char": "'",
        "null_values": "MISSING",
        "encoding": "utf8-lossy",
        "new_columns": ["id", "name"],
        "schema_overrides": [pl.String, pl.String],
    }
    original = dict(
        options,
        new_columns=list(options["new_columns"]),
        schema_overrides=list(options["schema_overrides"]),
    )
    source = BytesIO(b"0007;'a; b'\n0002;MISSING\n0010;bad\xff\n")
    target = tmp_path / "output.parquet"
    import_csv(source, target, csv_options=options)
    assert read_output(target).rows() == [
        ("0007", "a; b"),
        ("0002", None),
        ("0010", "bad�"),
    ]
    assert options == original
    assert not source.closed


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("existed", [False, True])
def test_sink_failures_preserve_output_and_stream_and_clean_scratch(
    tmp_path, monkeypatch, failure, existed
):
    target = tmp_path / "output.parquet"
    if existed:
        target.write_bytes(b"previous output")
    source = StringIO("id\n1\n")

    def fail(frame, path, **kwargs):
        Path(path).write_bytes(b"partial sink")
        raise failure("injected sink failure")

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", fail)
    with pytest.raises(failure, match="injected sink"):
        import_csv(source, target)
    assert not source.closed
    assert target.read_bytes() == b"previous output" if existed else not target.exists()
    assert list(tmp_path.iterdir()) == ([target] if existed else [])


@pytest.mark.parametrize("existed", [False, True])
def test_interrupt_after_rename_rolls_back_output(tmp_path, monkeypatch, existed):
    from intermine314.export import parquet

    target = tmp_path / "output.parquet"
    if existed:
        target.write_bytes(b"previous output")
    source = BytesIO(b"id\n1\n")
    replace = parquet.os.replace

    def interrupt(source, destination):
        replace(source, destination)
        if source.name == "output":
            raise KeyboardInterrupt("after rename")

    monkeypatch.setattr(parquet.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt, match="after rename"):
        import_csv(source, target)
    assert not source.closed
    assert target.read_bytes() == b"previous output" if existed else not target.exists()
    assert list(tmp_path.iterdir()) == ([target] if existed else [])


@pytest.mark.parametrize("text", ["id,name\n1,ok,extra\n", "id\n1\n2\nbad\n"])
def test_parse_errors_including_late_errors_preserve_existing_output(tmp_path, text):
    target = tmp_path / "output.parquet"
    target.write_bytes(b"previous output")
    source = StringIO(text)
    with pytest.raises(pl.exceptions.PolarsError):
        import_csv(source, target, csv_options={"infer_schema_length": 1})
    assert target.read_bytes() == b"previous output"
    assert not source.closed
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize(
    "collision", ["same", "symlink", "hardlink", "stream", "named_stream"]
)
def test_input_output_collision_does_not_overwrite_csv(tmp_path, collision):
    source = tmp_path / "source.txt"
    text = "id\n0007\n"
    source.write_text(text)
    target = tmp_path / "output.parquet"
    borrowed = None
    if collision == "same":
        target = source
    elif collision == "symlink":
        target.symlink_to(source)
    elif collision == "hardlink":
        target.hardlink_to(source)
    elif collision == "stream":
        target = source
        borrowed = source.open("rb")
    else:
        target = source
        borrowed = StringIO(text)
        borrowed.name = str(source)
    try:
        with pytest.raises(ValueError, match="same|source|symbolic"):
            import_csv(borrowed if borrowed is not None else source, target)
        assert source.read_text() == text
        if borrowed is not None:
            assert not borrowed.closed
    finally:
        if borrowed is not None:
            borrowed.close()


def test_glob_characters_are_literal(tmp_path):
    source = tmp_path / "data[*?].txt"
    source.write_text("id\n0007\n")
    (tmp_path / "datax.txt").write_text("id\nwrong\n")
    target = tmp_path / "output[*?].parquet"
    import_csv(source, target, csv_options={"infer_schema": False})
    assert read_output(target).item() == "0007"


@pytest.mark.parametrize(
    "options, error, message",
    [
        ([], TypeError, "csv_options"),
        ({1: True}, TypeError, "keys"),
        ({"unknown": True}, TypeError, "unknown"),
        ({"source": "other.csv"}, TypeError, "source"),
        ({"glob": True}, ValueError, "glob"),
        ({"separator": "::"}, ValueError, "separator"),
        ({"quote_char": "☃"}, ValueError, "quote_char"),
        ({"eol_char": ""}, ValueError, "eol_char"),
        ({"encoding": "latin1"}, ValueError, "encoding"),
        ({"new_columns": ["a"], "with_column_names": str}, ValueError, "mutually"),
    ],
)
def test_invalid_options_are_clear_and_do_not_touch_output(
    tmp_path, options, error, message
):
    target = tmp_path / "output.parquet"
    source = StringIO("id\n1\n")
    with pytest.raises(error, match=message):
        import_csv(source, target, csv_options=options)
    assert not source.closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "source, error, message",
    [
        (None, TypeError, "CSV input"),
        (["input.csv"], TypeError, "CSV input"),
        (b"id\n1\n", TypeError, "CSV input"),
        ("", ValueError, "CSV input"),
        ("bad\x00.csv", ValueError, "CSV input"),
        ("https://example.org/data.csv", ValueError, "local"),
        ("s3://bucket/data.csv", ValueError, "local"),
    ],
)
def test_invalid_sources_are_clear_without_network_or_output(
    tmp_path, source, error, message
):
    with pytest.raises(error, match=message):
        import_csv(source, tmp_path / "output.parquet")
    assert list(tmp_path.iterdir()) == []


def test_nonseekable_readable_stream_is_borrowed(tmp_path):
    class Reader:
        closed = False

        def read(self, size=-1):
            return "id\n0007\n"

        def close(self):
            self.closed = True

    source = Reader()
    target = tmp_path / "output.parquet"
    import_csv(source, target, csv_options={"infer_schema": False})
    assert read_output(target).item() == "0007"
    assert not source.closed


@pytest.mark.parametrize(
    "text, options, expected",
    [
        (
            "id,name\n",
            {"schema_overrides": {"id": pl.String, "name": pl.String}},
            {"id": pl.String, "name": pl.String},
        ),
        ("", {"schema": {"id": pl.String}, "raise_if_empty": False}, {"id": pl.String}),
    ],
)
def test_empty_typed_csv_preserves_schema(tmp_path, text, options, expected):
    target = tmp_path / "output.parquet"
    import_csv(StringIO(text), target, csv_options=options)
    result = read_output(target)
    assert result.height == 0
    assert result.schema == expected


def test_columnless_empty_csv_does_not_publish(tmp_path):
    target = tmp_path / "output.parquet"
    target.write_bytes(b"previous output")
    with pytest.raises(ValueError, match="column|schema"):
        import_csv(StringIO(""), target, csv_options={"raise_if_empty": False})
    assert target.read_bytes() == b"previous output"
    assert list(tmp_path.iterdir()) == [target]


def test_empty_csv_with_full_schema_defaults_to_typed_zero_rows(tmp_path):
    target = tmp_path / "output.parquet"
    options = {"schema": {"id": pl.String, "number": pl.Int64}}
    source = StringIO("")
    import_csv(source, target, csv_options=options)
    result = read_output(target)
    assert result.height == 0
    assert result.schema == options["schema"]
    assert options == {"schema": {"id": pl.String, "number": pl.Int64}}
    assert not source.closed


def test_empty_csv_with_full_schema_respects_explicit_raise_if_empty(tmp_path):
    target = tmp_path / "output.parquet"
    target.write_bytes(b"previous output")
    source = StringIO("")
    with pytest.raises(pl.exceptions.NoDataError):
        import_csv(
            source,
            target,
            csv_options={
                "schema": {"id": pl.String},
                "raise_if_empty": True,
            },
        )
    assert target.read_bytes() == b"previous output"
    assert not source.closed
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("schema_key", ["schema", "schema_overrides"])
def test_int128_schema_is_normalized_for_exact_duckdb_reads(tmp_path, schema_key):
    value = 2**80 + 1
    target = tmp_path / "output.parquet"
    options = {schema_key: {"number": pl.Int128}}
    import_csv(StringIO(f"number\n{value}\n"), target, csv_options=options)
    result = read_output(target)
    assert result.schema == {"number": pl.Decimal(38, 0)}
    assert result.item() == Decimal(value)
    assert options == {schema_key: {"number": pl.Int128}}


def test_timezone_nanoseconds_fail_before_sink(tmp_path, monkeypatch):
    target = tmp_path / "output.parquet"
    target.write_bytes(b"previous output")

    def forbidden(*args, **kwargs):
        pytest.fail("precision errors must be detected before sinking")

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", forbidden)
    with pytest.raises(ValueError, match="nanosecond.*timezone.*precision"):
        import_csv(
            StringIO("at\n2026-01-01T00:00:00.123456789Z\n"),
            target,
            csv_options={"schema_overrides": {"at": pl.Datetime("ns", "UTC")}},
        )
    assert target.read_bytes() == b"previous output"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize(
    "source_kind", ["missing", "directory", "closed", "unreadable"]
)
def test_missing_and_unreadable_sources_fail_before_publication(tmp_path, source_kind):
    target = tmp_path / "output.parquet"
    target.write_bytes(b"previous output")
    if source_kind == "missing":
        source, error = tmp_path / "missing.txt", FileNotFoundError
    elif source_kind == "directory":
        source, error = tmp_path, ValueError
    elif source_kind == "closed":
        source, error = StringIO("id\n1\n"), ValueError
        source.close()
    else:
        source, error = target.open("ab"), ValueError
    try:
        with pytest.raises(error, match="CSV input"):
            import_csv(source, target)
        assert target.read_bytes() == b"previous output"
        assert list(tmp_path.iterdir()) == [target]
        if source_kind == "unreadable":
            assert not source.closed
    finally:
        if source_kind == "unreadable":
            source.close()


@pytest.mark.parametrize(
    "target", [None, [], "", "bad\x00.parquet", "s3://bucket/output.parquet"]
)
def test_invalid_output_paths_leave_borrowed_input_open(target):
    source = StringIO("id\n1\n")
    with pytest.raises((TypeError, ValueError), match="Parquet output"):
        import_csv(source, target)
    assert not source.closed


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_failure_before_publication_rename_preserves_output(
    tmp_path, monkeypatch, failure
):
    from intermine314.export import parquet

    target = tmp_path / "output.parquet"
    target.write_bytes(b"previous output")
    source = StringIO("id\n1\n")
    replace = parquet.os.replace

    def fail(source, destination):
        if source.name == "output":
            raise failure("before rename")
        return replace(source, destination)

    monkeypatch.setattr(parquet.os, "replace", fail)
    with pytest.raises(failure, match="before rename"):
        import_csv(source, target)
    assert target.read_bytes() == b"previous output"
    assert not source.closed
    assert list(tmp_path.iterdir()) == [target]


def test_int128_decimal_overflow_preserves_previous_output(tmp_path):
    target = tmp_path / "output.parquet"
    target.write_bytes(b"previous output")
    source = StringIO(f"number\n{10**38}\n")
    with pytest.raises(pl.exceptions.PolarsError):
        import_csv(
            source, target, csv_options={"schema_overrides": {"number": pl.Int128}}
        )
    assert target.read_bytes() == b"previous output"
    assert not source.closed
    assert list(tmp_path.iterdir()) == [target]


def test_naive_nanosecond_csv_timestamp_retains_precision(tmp_path):
    target = tmp_path / "output.parquet"
    import_csv(
        StringIO("at\n2026-01-01T00:00:00.123456789\n"),
        target,
        csv_options={"schema_overrides": {"at": pl.Datetime("ns")}},
    )
    result = read_output(target)
    assert result.schema == {"at": pl.Datetime("ns")}
    assert result["at"].cast(pl.Int64).item() == 1767225600123456789
