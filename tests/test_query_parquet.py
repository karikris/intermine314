from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import duckdb
import polars as pl
import pytest


def query_parquet(*args, **kwargs):
    return importlib.import_module("intermine314.export").query_parquet(*args, **kwargs)


def test_metadata_and_data_reads_treat_wildcard_paths_as_literal(tmp_path):
    path = tmp_path / "literal[*?].parquet"
    pl.DataFrame({"identifier": ["0007"]}).write_parquet(path)
    assert query_parquet(path).item() == "0007"


@pytest.mark.parametrize("container", ["scalar", "list", "struct"])
def test_external_int128_metadata_fails_before_duckdb_data_read(
    tmp_path, monkeypatch, container
):
    value = 2**80 + 1
    dtype = pl.Int128
    if container == "list":
        value, dtype = [value], pl.List(dtype)
    elif container == "struct":
        value, dtype = {"number": value}, pl.Struct({"number": dtype})
    path = tmp_path / "external.parquet"
    pl.DataFrame({"number": [value]}, schema={"number": dtype}).write_parquet(path)

    def forbidden(*args, **kwargs):
        pytest.fail("Int128 must be rejected before opening DuckDB")

    monkeypatch.setattr(duckdb, "connect", forbidden)
    with pytest.raises(ValueError, match="Int128.*Decimal.*binary"):
        query_parquet(path)


def test_public_helper_signature_and_lazy_imports():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))
    script = (
        "import json,sys; from intermine314.export import query_parquet; "
        "print(json.dumps(sorted(m for m in sys.modules "
        "if any(m==p or m.startswith(p+'.') "
        "for p in ('pandas','polars','duckdb','pyarrow')))))"
    )
    process = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == []
    helper = importlib.import_module("intermine314.export").query_parquet
    assert str(inspect.signature(helper)) == (
        "(path, sql='SELECT * FROM results', *, parameters=None, database=':memory:')"
    )


def test_typed_parquet_preserves_schema_values_and_column_order(typed_parquet):
    result = query_parquet(typed_parquet)
    assert isinstance(result, pl.DataFrame)
    assert result.schema == {
        "identifier": pl.String,
        "name": pl.String,
        "large_integer": pl.Int64,
        "active": pl.Boolean,
        "amount": pl.Decimal(20, 4),
        "observed_at": pl.Datetime("us"),
    }
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
        ("0010", "line\nbreak", -9007199254740993, None, Decimal("-0.0001"), None),
    ]


@pytest.mark.parametrize("parameters", [["0007"], ("0007",)])
def test_sql_filters_and_reorders_columns_with_positional_parameters(
    typed_parquet, parameters
):
    result = query_parquet(
        typed_parquet,
        "SELECT amount, identifier, large_integer FROM results WHERE identifier = ?",
        parameters=parameters,
    )
    assert result.columns == ["amount", "identifier", "large_integer"]
    assert result.rows() == [(Decimal("123456789.1234"), "0007", 9007199254740993)]


def test_sql_grouping_and_aggregation_preserve_decimal_values(typed_parquet):
    result = query_parquet(
        typed_parquet,
        "SELECT active, COUNT(*) AS n, SUM(amount) AS total "
        "FROM results GROUP BY active ORDER BY active NULLS LAST",
    )
    assert result.schema == {
        "active": pl.Boolean,
        "n": pl.Int64,
        "total": pl.Decimal(38, 4),
    }
    assert result.rows() == [
        (False, 1, None),
        (True, 1, Decimal("123456789.1234")),
        (None, 1, Decimal("-0.0001")),
    ]


def test_named_parameters_keep_quoted_values_literal(tmp_path):
    value = "O'Brien'; DROP VIEW results; --"
    path = tmp_path / "quoted.parquet"
    pl.DataFrame({"name": [value, "other"]}).write_parquet(path)
    result = query_parquet(
        path,
        "SELECT name FROM results WHERE name = $name",
        parameters={"name": value},
    )
    assert result.rows() == [(value,)]


@pytest.mark.parametrize(
    ("filename", "matching_sibling"),
    [
        ("quo'te[01].parquet", "quo'te0.parquet"),
        ("star*.parquet", "star-other.parquet"),
        ("question?.parquet", "question0.parquet"),
    ],
)
def test_explicit_filename_quotes_and_brackets_are_literal(
    tmp_path, filename, matching_sibling
):
    path = tmp_path / filename
    pl.DataFrame({"id": [7]}).write_parquet(path)
    pl.DataFrame({"id": [99]}).write_parquet(tmp_path / matching_sibling)
    assert query_parquet(str(path)).rows() == [(7,)]


def test_managed_dataset_reads_only_parts_in_numeric_order(tmp_path):
    dataset = tmp_path / "managed'parts[01]"
    dataset.mkdir()
    for index in [100000, 2, 99999, 10, 0]:
        pl.DataFrame({"id": [index]}).write_parquet(
            dataset / f"part-{index:05d}.parquet"
        )
    pl.DataFrame({"id": [-1]}).write_parquet(dataset / "unrelated.parquet")
    nested = dataset / "other"
    nested.mkdir()
    pl.DataFrame({"id": [-2]}).write_parquet(nested / "part-00001.parquet")
    (dataset / "part-00003.parquet").mkdir()
    sibling = tmp_path / "managed'parts0"
    sibling.mkdir()
    for index in [100000, 2, 99999, 10, 0]:
        pl.DataFrame({"id": [-100]}).write_parquet(
            sibling / f"part-{index:05d}.parquet"
        )
    assert query_parquet(dataset).rows() == [(0,), (2,), (10,), (99999,), (100000,)]
    assert query_parquet(dataset, "SELECT COUNT(*) AS n FROM results").item() == 5


def test_unmanaged_directory_reads_recursive_regular_parquet_files(tmp_path):
    dataset = tmp_path / "ordinary"
    dataset.mkdir()
    pl.DataFrame({"id": [1]}).write_parquet(dataset / "part-a1b2c3.parquet")
    nested = dataset / "partition=a"
    nested.mkdir()
    pl.DataFrame({"id": [2]}).write_parquet(nested / "two.parquet")
    (dataset / "readme.txt").write_text("metadata")
    (dataset / "fake.parquet").mkdir()
    result = query_parquet(dataset, "SELECT id FROM results ORDER BY id")
    assert result.rows() == [(1,), (2,)]


def test_empty_dataset_and_missing_path_have_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Parquet"):
        query_parquet(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing"):
        query_parquet(tmp_path / "missing.parquet")


def test_empty_typed_parquet_keeps_schema(typed_parquet, tmp_path):
    frame = pl.read_parquet(typed_parquet).head(0)
    path = tmp_path / "empty.parquet"
    frame.write_parquet(path)
    result = query_parquet(path)
    assert result.schema == frame.schema
    assert result.height == 0


@pytest.mark.parametrize("container", ["scalar", "list", "struct", "array", "nested"])
def test_nanosecond_timezone_parquet_is_rejected_before_connect(
    tmp_path, monkeypatch, container
):
    series = pl.Series("at", [1760000000000000123], dtype=pl.Int64).cast(
        pl.Datetime("ns", "UTC")
    )
    frame = series.to_frame()
    if container in ("list", "array", "nested"):
        frame = frame.select(pl.col("at").implode())
    if container == "array":
        frame = frame.select(pl.col("at").list.to_array(1))
    if container in ("struct", "nested"):
        frame = frame.select(pl.struct("at").alias("event"))
    path = tmp_path / "nanoseconds.parquet"
    frame.write_parquet(path)
    assert series.cast(pl.Int64).item() == 1760000000000000123

    def forbidden_connect(*args, **kwargs):
        pytest.fail("Unsupported timestamps must fail before opening DuckDB")

    monkeypatch.setattr(duckdb, "connect", forbidden_connect)
    with pytest.raises(ValueError, match="nanosecond.*timezone.*precision"):
        query_parquet(path)


def test_dataset_validates_every_part_for_nanosecond_timezones(tmp_path):
    pl.DataFrame({"at": [1]}).write_parquet(tmp_path / "part-00000.parquet")
    series = pl.Series("at", [1760000000000000123], dtype=pl.Int64).cast(
        pl.Datetime("ns", "UTC")
    )
    series.to_frame().write_parquet(tmp_path / "part-00001.parquet")
    with pytest.raises(ValueError, match="nanosecond.*timezone.*precision"):
        query_parquet(tmp_path)


@pytest.mark.parametrize("zone", ["UTC", "Europe/Helsinki", "Australia/Sydney"])
def test_microsecond_timezone_values_normalize_to_utc_without_precision_loss(
    tmp_path, zone
):
    value = 1760000000000123
    series = pl.Series("at", [value, None], dtype=pl.Int64).cast(
        pl.Datetime("us", zone)
    )
    path = tmp_path / "microseconds.parquet"
    series.to_frame().write_parquet(path)
    result = query_parquet(path)
    assert result.schema == {"at": pl.Datetime("us", "UTC")}
    assert result["at"].cast(pl.Int64).to_list() == [value, None]


def test_naive_nanosecond_timestamp_precision_is_preserved(tmp_path):
    value = 1760000000000000123
    series = pl.Series("at", [value, None], dtype=pl.Int64).cast(pl.Datetime("ns"))
    path = tmp_path / "naive.parquet"
    series.to_frame().write_parquet(path)
    result = query_parquet(path)
    assert result.schema == {"at": pl.Datetime("ns")}
    assert result["at"].cast(pl.Int64).to_list() == [value, None]


@pytest.mark.parametrize("existing", ["view", "table"])
@pytest.mark.parametrize("sql_failure", [False, True])
def test_results_view_is_local_and_preserves_existing_database_objects(
    tmp_path, existing, sql_failure
):
    database = str(tmp_path / "existing.duckdb")
    path = tmp_path / "one.parquet"
    pl.DataFrame({"id": [1]}).write_parquet(path)
    with duckdb.connect(database=database) as connection:
        connection.execute(f"CREATE {existing} results AS SELECT 42 AS id")
    if sql_failure:
        with pytest.raises(duckdb.Error):
            query_parquet(path, "SELECT missing_column FROM results", database=database)
    else:
        assert query_parquet(path, database=database).rows() == [(1,)]
    with duckdb.connect(database=database) as connection:
        assert connection.execute("SELECT * FROM results").fetchall() == [(42,)]
        assert connection.execute(
            "SELECT table_type FROM information_schema.tables "
            "WHERE table_name = 'results'"
        ).fetchall() == [("VIEW" if existing == "view" else "BASE TABLE",)]


def test_reader_uses_arrow_without_pandas_csv_or_polars_parquet_reads(
    typed_parquet, monkeypatch
):
    scan_parquet = pl.scan_parquet

    def metadata_scan(path, **kwargs):
        assert kwargs == {"glob": False}
        return scan_parquet(path, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("The query helper must read Parquet through DuckDB and Arrow")

    monkeypatch.setattr(pl, "scan_parquet", metadata_scan)
    monkeypatch.setattr(pl.LazyFrame, "collect", forbidden)
    for name in ["read_parquet", "read_csv", "scan_csv", "from_pandas"]:
        monkeypatch.setattr(pl, name, forbidden)
    before = sorted(path.name for path in typed_parquet.parent.iterdir())
    assert query_parquet(typed_parquet).height == 3
    assert sorted(path.name for path in typed_parquet.parent.iterdir()) == before


def test_csv_is_not_automatically_imported(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("id\n7\n")
    with pytest.raises((duckdb.Error, pl.exceptions.PolarsError), match="Parquet"):
        query_parquet(path)


@pytest.mark.parametrize("failure", ["read", "sql", "arrow", "conversion", None])
def test_real_connection_closes_and_success_result_is_detached(
    typed_parquet, tmp_path, monkeypatch, failure
):
    connections = []
    original_connect = duckdb.connect
    original_from_arrow = pl.from_arrow

    class Connection:
        def __init__(self, database):
            self.inner = original_connect(database=database)
            self.closed = 0
            connections.append(self)

        def read_parquet(self, paths):
            if failure == "read":
                raise RuntimeError("read failed")
            return self.inner.read_parquet(paths)

        def register(self, name, relation):
            return self.inner.register(name, relation)

        def execute(self, sql, parameters=None):
            self.inner.execute(sql, parameters)
            return self

        def to_arrow_table(self):
            if failure == "arrow":
                raise RuntimeError("arrow failed")
            return self.inner.to_arrow_table()

        def close(self):
            self.closed += 1
            self.inner.close()

    def from_arrow(table):
        if failure == "conversion":
            raise RuntimeError("conversion failed")
        return original_from_arrow(table)

    monkeypatch.setattr(duckdb, "connect", lambda *, database: Connection(database))
    monkeypatch.setattr(pl, "from_arrow", from_arrow)
    database = str(tmp_path / "query.duckdb")
    if failure:
        with pytest.raises((RuntimeError, duckdb.Error)):
            query_parquet(
                typed_parquet,
                "SELECT invalid_column FROM results"
                if failure == "sql"
                else "SELECT * FROM results",
                database=database,
            )
    else:
        result = query_parquet(typed_parquet, database=database)
        assert result["identifier"].to_list() == ["0007", "0002", "0010"]
    assert len(connections) == 1
    assert connections[0].closed == 1
    with pytest.raises(duckdb.Error, match="closed"):
        connections[0].inner.execute("SELECT 1")
    with original_connect(database=database) as reopened:
        assert reopened.execute("SELECT 1").fetchone() == (1,)


@pytest.mark.parametrize("stage", ["read", "view", "sql", "arrow", "conversion"])
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_connection_closes_on_exceptions_and_interrupts(
    typed_parquet, monkeypatch, stage, exception
):
    closed = []

    def fail_at(current):
        if stage == current:
            raise exception(current)

    class Connection:
        def read_parquet(self, paths):
            fail_at("read")
            assert paths == str(typed_parquet)
            return self

        def register(self, name, relation):
            fail_at("view")
            assert name == "results"
            assert relation is self

        def execute(self, sql, parameters=None):
            fail_at("sql")
            return self

        def to_arrow_table(self):
            fail_at("arrow")
            return object()

        def close(self):
            closed.append(True)

    module = importlib.import_module("intermine314.export.query")
    monkeypatch.setattr(
        module,
        "require_duckdb",
        lambda api: SimpleNamespace(connect=lambda *, database: Connection()),
    )

    def from_arrow(table):
        fail_at("conversion")

    monkeypatch.setattr(pl, "from_arrow", from_arrow)
    with pytest.raises(exception, match=stage):
        query_parquet(typed_parquet)
    assert closed == [True]
