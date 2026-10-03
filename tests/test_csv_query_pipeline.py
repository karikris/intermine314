from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import polars as pl
import pytest

from intermine314.export import fetch_from_mine, query_parquet
from intermine314.query.builder import Query


def csv_source():
    return StringIO("id,number,label\n0007,9007199254740993,one\n0002,9007199254740995,\n0010,-9007199254740993,three\n")


CSV_OPTIONS = {"schema_overrides": {"id": pl.String, "number": pl.Int64, "label": pl.String}}
EXPECTED = [("0007", 9007199254740993, "one"), ("0002", 9007199254740995, None), ("0010", -9007199254740993, "three")]


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("single_file", [False, True])
def test_csv_exports_are_offline_typed_paginated_and_compressed(tmp_path, monkeypatch, profile, single_file):
    query = Query(root="Gene", compatibility=profile)
    query.service = object()  # Attached remote identity must never be consulted.

    def forbidden(*args, **kwargs):
        pytest.fail("CSV mode must never fetch remote rows")

    monkeypatch.setattr(query, "iter_batches", forbidden)
    target = tmp_path / "output[*?]"
    scratch = tmp_path / "scratch"
    source = csv_source()
    assert query.to_parquet(target, start=1, size=2, batch_size=1, compression="gzip", single_file=single_file, temp_dir=scratch, csv_input=source, csv_options=CSV_OPTIONS) == str(target)
    assert query_parquet(target).rows() == EXPECTED[1:]
    files = [target] if single_file else sorted(target.glob("*.parquet"))
    assert len(files) == (1 if single_file else 2)
    import pyarrow.parquet as pq
    assert all(pq.ParquetFile(file).metadata.row_group(0).column(0).compression == "GZIP" for file in files)
    assert list(scratch.iterdir()) == []
    assert not source.closed


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("persistent", [False, True])
def test_dataframe_detaches_and_cleans_only_managed_parquet(tmp_path, monkeypatch, profile, persistent):
    from intermine314.export import query as export_query

    query = Query(compatibility=profile)
    paths = []
    original = export_query.query_parquet

    def read(path, *args, **kwargs):
        paths.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(export_query, "query_parquet", read)
    target = tmp_path / "saved.parquet" if persistent else None
    frame = query.dataframe(1, 1, csv_input=csv_source(), csv_options=CSV_OPTIONS, parquet_path=target)
    assert isinstance(frame, pl.DataFrame)
    assert frame.rows() == EXPECTED[1:2]
    assert paths[0].exists() == persistent
    assert frame.clone().rows() == EXPECTED[1:2]


@pytest.mark.parametrize("method", ["to_parquet", "to_duckdb", "dataframe"])
@pytest.mark.parametrize("content", ["views", "constraint_dict", "uncoded_constraints", "joins", "_sort_order_list"])
def test_csv_conflicting_query_content_rejects_before_output(tmp_path, method, content):
    query = Query()
    setattr(query, content, ["remote content"])
    target = tmp_path / "output"
    kwargs = {"csv_input": csv_source()}
    args = () if method == "dataframe" else (target,)
    if method == "dataframe":
        kwargs["parquet_path"] = target
    with pytest.raises(ValueError, match="CSV.*query|query.*CSV"):
        getattr(query, method)(*args, **kwargs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("method", ["to_parquet", "to_duckdb", "dataframe"])
def test_csv_options_without_input_rejects(tmp_path, method):
    args = () if method == "dataframe" else (tmp_path / "output",)
    with pytest.raises(ValueError, match="csv_options.*csv_input"):
        getattr(Query(), method)(*args, csv_options={})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("start,size", [(-1, None), (True, None), (0, -1), (0, 1.5), ("1", None)])
def test_csv_pagination_rejects_before_writing(tmp_path, start, size):
    with pytest.raises((TypeError, ValueError), match="start|size"):
        Query().to_parquet(tmp_path / "output", start=start, size=size, csv_input=csv_source())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("single_file", [False, True])
def test_csv_empty_page_preserves_schema(tmp_path, single_file):
    target = tmp_path / "output"
    Query().to_parquet(target, size=0, single_file=single_file, csv_input=csv_source(), csv_options=CSV_OPTIONS)
    frame = query_parquet(target)
    assert frame.height == 0
    assert frame.schema == CSV_OPTIONS["schema_overrides"]


def test_csv_partitions_preserve_nanoseconds_and_read_bounded_arrow(tmp_path, monkeypatch):
    from intermine314.export import parquet

    sizes = []
    original = parquet._cast_losslessly

    def cast(frame, schema):
        sizes.append(frame.height)
        return original(frame, schema)

    def forbidden(*args, **kwargs):
        pytest.fail("CSV partitions must not collect the full dataset")

    monkeypatch.setattr(parquet, "_cast_losslessly", cast)
    monkeypatch.setattr(pl.LazyFrame, "collect", forbidden)
    source = StringIO("at\n2026-01-01T00:00:00.123456789\n2026-01-01T00:00:00.123456790\n")
    Query().to_parquet(tmp_path / "output", batch_size=1, csv_input=source, csv_options={"schema_overrides": {"at": pl.Datetime("ns")}})
    assert sizes and max(sizes) == 1
    assert query_parquet(tmp_path / "output")["at"].cast(pl.Int64).to_list() == [1767225600123456789, 1767225600123456790]


def test_fetch_csv_never_constructs_service_and_keeps_return_contract(tmp_path, monkeypatch):
    from intermine314.export import fetch

    def forbidden(*args, **kwargs):
        pytest.fail("CSV fetch must never construct Service")

    monkeypatch.setattr(fetch, "Service", forbidden)
    target = tmp_path / "output[*?].parquet"
    payload = fetch_from_mine(parquet_path=target, csv_input=csv_source(), csv_options=CSV_OPTIONS, start=1, size=1, parquet_compression="gzip", managed=True)
    assert set(payload) == {"parquet_path", "duckdb_table", "duckdb_connection"}
    assert payload["parquet_path"] == str(target)
    with payload["duckdb_connection"] as connection:
        assert connection.execute("SELECT * FROM results").fetchall() == EXPECTED[1:2]
    assert target.is_file()


@pytest.mark.parametrize("api", ["dataframe", "fetch"])
def test_csv_callers_read_the_expanded_output_path(tmp_path, monkeypatch, api):
    target = "~/intermine314-task25/output.parquet"
    actual = tmp_path / "task25-home" / "intermine314-task25" / "output.parquet"
    expanduser = Path.expanduser

    def expand(path):
        return actual if str(path) == target else expanduser(path)

    monkeypatch.setattr(Path, "expanduser", expand)
    if api == "dataframe":
        assert Query().dataframe(csv_input=csv_source(), csv_options=CSV_OPTIONS, parquet_path=target).rows() == EXPECTED
    else:
        payload = fetch_from_mine(csv_input=csv_source(), csv_options=CSV_OPTIONS, parquet_path=target, managed=True)
        assert payload["parquet_path"] == str(actual)
        with payload["duckdb_connection"] as connection:
            assert connection.execute("SELECT * FROM results").fetchall() == EXPECTED
    assert actual.is_file()


@pytest.mark.parametrize("configuration", [{"mine_url": "https://example.org"}, {"root_class": "Gene"}, {"views": []}])
def test_fetch_rejects_csv_with_remote_configuration(tmp_path, configuration):
    with pytest.raises(ValueError, match="CSV.*remote|remote.*CSV"):
        fetch_from_mine(parquet_path=tmp_path / "output", csv_input=csv_source(), **configuration)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("missing", ["mine_url", "root_class", "views"])
def test_remote_fetch_requires_remote_fields(tmp_path, missing):
    config = {"mine_url": "https://example.org", "root_class": "Gene", "views": ["Gene.id"]}
    del config[missing]
    with pytest.raises(ValueError, match="mine_url.*root_class.*views"):
        fetch_from_mine(parquet_path=tmp_path / "output", **config)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_remote_dataframe_uses_existing_export_pagination(tmp_path, monkeypatch, profile):
    query = Query(compatibility=profile)
    query.views = ["Gene.id"]
    calls = []

    def batches(**kwargs):
        calls.append(kwargs)
        yield [{"Gene.id": "0007"}]

    monkeypatch.setattr(query, "iter_batches", batches)
    target = tmp_path / "output.parquet"
    frame = query.dataframe(2, 1, parquet_path=target)
    assert frame.to_dicts() == [{"Gene.id": "0007"}]
    assert calls[0]["start"] == 2 and calls[0]["size"] == 1
    assert target.is_file()


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("single_file", [False, True])
def test_csv_to_duckdb_creates_persistent_literal_path_view(tmp_path, profile, single_file):
    import duckdb

    database = tmp_path / "persistent.duckdb"
    target = tmp_path / "source[*?]"
    connection = Query(compatibility=profile).to_duckdb(target, database=str(database), single_file=single_file, csv_input=csv_source(), csv_options=CSV_OPTIONS)
    assert connection.execute("SELECT * FROM results").fetchall() == EXPECTED
    connection.close()
    with duckdb.connect(str(database)) as reopened:
        assert reopened.execute("SELECT * FROM results").fetchall() == EXPECTED


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_dataframe_failure_removes_managed_parquet(tmp_path, monkeypatch, failure):
    from intermine314.export import query as export_query

    paths = []

    def fail(path):
        paths.append(path)
        assert path.exists()
        raise failure("injected reader failure")

    monkeypatch.setattr(export_query, "query_parquet", fail)
    with pytest.raises(failure, match="injected reader"):
        Query().dataframe(csv_input=csv_source())
    assert not paths[0].exists() and not paths[0].parent.exists()


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_csv_partition_reader_and_connection_close_and_rollback(tmp_path, monkeypatch, failure):
    from intermine314.export import csv

    closed = []
    original = csv._connection

    class Reader:
        def __init__(self, reader):
            self.reader = reader
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.count += 1
            if self.count == 2:
                raise failure("injected batch failure")
            return next(self.reader)

        def close(self):
            closed.append("reader")
            self.reader.close()

    class Connection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, *args):
            self.connection.execute(*args)
            return self

        def to_arrow_reader(self, size):
            return Reader(self.connection.to_arrow_reader(size))

    @contextmanager
    def connection(scratch):
        with original(scratch) as owned:
            try:
                yield Connection(owned)
            finally:
                closed.append("connection")

    monkeypatch.setattr(csv, "_connection", connection)
    target = tmp_path / "output"
    target.mkdir()
    (target / "previous.txt").write_text("previous output")
    scratch = tmp_path / "scratch"
    with pytest.raises(failure, match="injected batch"):
        Query().to_parquet(target, batch_size=1, temp_dir=scratch, csv_input=csv_source(), csv_options=CSV_OPTIONS)
    assert closed == ["reader", "connection"]
    assert list(scratch.iterdir()) == []
    assert [p.name for p in target.iterdir()] == ["previous.txt"]


@pytest.mark.parametrize("api", ["to_duckdb", "fetch", "remote_fetch"])
@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_persistent_view_failure_closes_owned_connection(tmp_path, monkeypatch, api, failure):
    from intermine314.export import fetch
    from intermine314.query import builder

    class Connection:
        closed = False

        def execute(self, sql):
            raise failure("injected view failure")

        def close(self):
            self.closed = True

    connection = Connection()

    class DuckDB:
        def connect(self, **kwargs):
            return connection

    class RemoteQuery:
        def clear_view(self):
            pass

        def add_view(self, *views):
            pass

        def to_parquet(self, path, **kwargs):
            pl.DataFrame({"id": ["0007"]}).write_parquet(path)

    class Service:
        closed = False

        def __init__(self, url):
            pass

        def select(self, root):
            return RemoteQuery()

        def close(self):
            Service.closed = True

    monkeypatch.setattr(fetch, "Service", Service)
    module = builder if api == "to_duckdb" else fetch
    monkeypatch.setattr(module, "_require_duckdb", lambda name: DuckDB())
    target = tmp_path / "output.parquet"
    with pytest.raises(failure, match="injected view"):
        if api == "fetch":
            fetch_from_mine(parquet_path=target, csv_input=csv_source())
        elif api == "remote_fetch":
            fetch_from_mine(parquet_path=target, mine_url="https://example.org", root_class="Gene", views=["Gene.id"])
        else:
            Query().to_duckdb(target, single_file=True, csv_input=csv_source())
    assert connection.closed
    assert Service.closed == (api == "remote_fetch")


@pytest.mark.parametrize("schema", [pl.Int128, pl.Datetime("ns", "UTC")])
def test_native_duckdb_guards_real_source_before_connect(tmp_path, monkeypatch, schema):
    from intermine314.query import builder

    path = tmp_path / "source.parquet"
    pl.DataFrame({"value": [1]}, schema={"value": schema}).write_parquet(path)

    class Harness:
        def _coerce_parallel_options(self, **kwargs):
            return None

        def to_parquet(self, *args, **kwargs):
            return str(path)

    class DuckDB:
        def connect(self, **kwargs):
            pytest.fail("unsupported metadata must reject before DuckDB reads data")

    monkeypatch.setattr(builder, "_require_duckdb", lambda name: DuckDB())
    with pytest.raises(ValueError, match="Int128|nanosecond"):
        Query.to_duckdb(Harness(), path)
