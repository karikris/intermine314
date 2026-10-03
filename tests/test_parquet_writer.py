from datetime import UTC, datetime
from decimal import Decimal

import polars as pl
import pytest

from intermine314.query import Query


def query_with_rows(rows, columns, *, profile="native"):
    query = Query(compatibility=profile)
    query.add_view(*columns)
    query.iter_rows = lambda **kwargs: iter(rows)
    return query


@pytest.mark.parametrize("container", ["scalar", "list", "struct", "array", "nested"])
@pytest.mark.parametrize("empty", [False, True])
def test_single_file_rejects_nanosecond_timezones_and_preserves_old_output(
    tmp_path, container, empty
):
    target = tmp_path / "result.parquet"
    query_with_rows([{"Employee.id": 42}], ["Employee.id"]).to_parquet(
        target, single_file=True
    )
    before = target.read_bytes()
    dtype = pl.Datetime("ns", "UTC")
    value = datetime(2026, 1, 1, tzinfo=UTC)
    if container in ("list", "nested"):
        dtype, value = pl.List(dtype), [value]
    elif container == "array":
        dtype, value = pl.Array(dtype, 1), [value]
    if container in ("struct", "nested"):
        dtype, value = pl.Struct({"at": dtype}), {"at": value}
    query = query_with_rows([] if empty else [{"Employee.id": value}], ["Employee.id"])
    query._parquet_schema = lambda: {"Employee.id": dtype}
    with pytest.raises(ValueError, match="nanosecond.*timezone.*precision"):
        query.to_parquet(target, single_file=True)
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("single_file", [False, True])
@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_null_first_batches_reconcile_precision_and_column_order(
    tmp_path, single_file, profile
):
    columns = ["Employee.id", "Employee.amount", "Employee.seen"]
    rows = [
        dict.fromkeys(columns),
        dict(
            zip(
                reversed(columns),
                [
                    datetime(2026, 1, 2, 3, 4, 5, 123456),
                    Decimal("123456789.1234"),
                    9007199254740993,
                ],
            )
        ),
        dict(zip(columns, [-9007199254740993, Decimal("0.000001"), None])),
    ]
    query = query_with_rows(rows, columns, profile=profile)
    target = tmp_path / "result"
    assert query.to_parquet(target, 0, None, 1, "zstd", single_file) == str(target)
    parts = [target] if single_file else sorted(target.glob("part-*.parquet"))
    frames = [pl.read_parquet(part) for part in parts]
    assert all(frame.schema == frames[0].schema for frame in frames)
    result = pl.concat(frames)
    assert result.columns == columns
    assert result.to_dicts() == rows


@pytest.mark.parametrize("single_file", [False, True])
def test_empty_export_keeps_selected_columns(tmp_path, single_file):
    target = tmp_path / "result"
    query_with_rows([], ["Employee.id", "Employee.name"]).to_parquet(
        target, single_file=single_file
    )
    result = pl.read_parquet(target if single_file else target / "part-00000.parquet")
    assert result.columns == ["Employee.id", "Employee.name"]
    assert result.height == 0


@pytest.mark.parametrize("single_file", [False, True])
@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_failed_write_keeps_old_output_and_closes_rows(
    tmp_path, monkeypatch, single_file, failure
):
    target = tmp_path / "result"
    query_with_rows([{"Employee.id": 42}], ["Employee.id"]).to_parquet(
        target, single_file=single_file
    )
    closed = []

    def rows(**kwargs):
        try:
            yield {"Employee.id": 1}
            yield {"Employee.id": 2}
        finally:
            closed.append(True)

    query = query_with_rows([], ["Employee.id"])
    source = rows()
    query.iter_rows = lambda **kwargs: source

    def fail(*args, **kwargs):
        raise failure("injected write failure")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail)
    with pytest.raises(failure):
        query.to_parquet(
            target, batch_size=1, single_file=single_file, temp_dir=tmp_path / "scratch"
        )
    assert closed == [True]
    old = pl.read_parquet(target if single_file else target / "part-00000.parquet")
    assert old.to_dicts() == [{"Employee.id": 42}]
    assert list((tmp_path / "scratch").iterdir()) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result", "scratch"]


def test_success_replaces_only_managed_parts(tmp_path):
    target = tmp_path / "result"
    target.mkdir()
    (target / "notes.txt").write_text("keep")
    pl.DataFrame({"other": [99]}).write_parquet(target / "unrelated.parquet")
    pl.DataFrame({"Employee.id": [42]}).write_parquet(target / "part-99999.parquet")
    query_with_rows([{"Employee.id": 1}], ["Employee.id"]).to_parquet(target)
    assert sorted(p.name for p in target.iterdir()) == [
        "notes.txt",
        "part-00000.parquet",
        "unrelated.parquet",
    ]
    assert (target / "notes.txt").read_text() == "keep"
    assert pl.read_parquet(target / "unrelated.parquet").item() == 99


@pytest.mark.parametrize(
    "rows",
    [
        [{"Employee.id": 1}, {"Employee.id": 2, "extra": 3}],
        [{"Employee.id": 9007199254740993}, {"Employee.id": 1.5}],
        [{"Employee.id": 1}, {"other": 2}],
    ],
)
def test_schema_mismatch_never_silently_loses_fields_or_precision(tmp_path, rows):
    target = tmp_path / "result"
    with pytest.raises((ValueError, TypeError, pl.exceptions.PolarsError)):
        query_with_rows(rows, ["Employee.id"]).to_parquet(target, batch_size=1)
    assert not target.exists()


def test_partitioned_export_checks_scratch_free_space_before_reading(tmp_path):
    query = query_with_rows([], ["Employee.id"])

    def unexpected(**kwargs):
        pytest.fail("producer started before resource validation")

    query.iter_rows = unexpected
    with pytest.raises(ValueError, match="free bytes"):
        query.to_parquet(
            tmp_path / "result",
            temp_dir=tmp_path / "scratch",
            temp_dir_min_free_bytes=2**100,
        )
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_directory_publication_rolls_back_if_final_rename_fails(
    tmp_path, monkeypatch, failure
):
    from intermine314.export import parquet

    target = tmp_path / "result"
    query_with_rows([{"Employee.id": 42}], ["Employee.id"]).to_parquet(target)
    (target / "notes").write_text("keep")
    monkeypatch.setattr(parquet, "_exchange_directories", lambda *args: False)
    original = parquet.os.replace

    def replace(source, destination):
        if source.name == "output":
            raise failure("publication interrupted")
        return original(source, destination)

    monkeypatch.setattr(parquet.os, "replace", replace)
    with pytest.raises(failure):
        query_with_rows([{"Employee.id": 1}], ["Employee.id"]).to_parquet(target)
    assert pl.read_parquet(target / "part-00000.parquet").item() == 42
    assert (target / "notes").read_text() == "keep"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_single_file_merge_failure_preserves_target(tmp_path, monkeypatch, failure):
    target = tmp_path / "result"
    query_with_rows([{"Employee.id": 42}], ["Employee.id"]).to_parquet(
        target, single_file=True
    )
    import duckdb

    original_connect = duckdb.connect
    closed = []

    class Connection:
        def __init__(self, *args, **kwargs):
            self.inner = original_connect(*args, **kwargs)

        def execute(self, sql, *args, **kwargs):
            if sql.startswith("COPY"):
                raise failure("merge interrupted")
            return self.inner.execute(sql, *args, **kwargs)

        def close(self):
            closed.append(True)
            self.inner.close()

    monkeypatch.setattr(duckdb, "connect", Connection)
    with pytest.raises(failure):
        query_with_rows([{"Employee.id": 1}], ["Employee.id"]).to_parquet(
            target, single_file=True
        )
    assert pl.read_parquet(target).item() == 42
    assert closed == [True]
    assert list(tmp_path.iterdir()) == [target]


def test_schema_hook_preserves_empty_types_and_rejects_lossy_cast(tmp_path):
    query = query_with_rows([], ["Employee.id"])
    query._parquet_schema = lambda: {"Employee.id": pl.Int64}
    query.to_parquet(tmp_path / "empty")
    assert pl.read_parquet(tmp_path / "empty" / "part-00000.parquet").schema == {
        "Employee.id": pl.Int64
    }
    query.iter_rows = lambda **kwargs: iter([{"Employee.id": 1.5}])
    with pytest.raises(ValueError, match="lose precision"):
        query.to_parquet(tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("single_file", [False, True])
def test_writer_keeps_only_a_bounded_number_of_rows_and_reads_one_part(
    tmp_path, monkeypatch, single_file
):
    import weakref

    alive = weakref.WeakSet()
    peak = []

    class Row(dict):
        __hash__ = object.__hash__
        __eq__ = object.__eq__

    def rows(**kwargs):
        for index in range(1000):
            row = Row({"Employee.id": None if index < 7 else index})
            alive.add(row)
            peak.append(len(alive))
            yield row

    query = query_with_rows([], ["Employee.id"])
    query.iter_rows = rows
    original_read = pl.read_parquet
    original_from_arrow = pl.from_arrow
    reads = []

    def from_arrow(table, **kwargs):
        reads.append(table.num_rows)
        assert table.num_rows <= 7
        return original_from_arrow(table, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Parquet data must be read through DuckDB")

    monkeypatch.setattr(pl, "from_arrow", from_arrow)
    monkeypatch.setattr(pl, "read_parquet", forbidden)
    monkeypatch.setattr(pl, "scan_parquet", forbidden)
    target = tmp_path / "result"
    query.to_parquet(target, batch_size=7, single_file=single_file)
    assert max(peak) <= 8
    assert reads == [7]
    monkeypatch.undo()
    result = original_read(target if single_file else target / "part-*.parquet")
    assert result.height == 1000
    assert result["Employee.id"].to_list() == [None] * 7 + list(range(7, 1000))


def test_existing_parallel_controls_reach_producer(tmp_path):
    from intermine314.query import ParallelOptions

    query = query_with_rows([], ["Employee.id"])
    seen = []
    query.iter_rows = lambda **kwargs: seen.append(kwargs) or iter([{"Employee.id": 2}])
    options = ParallelOptions(
        page_size=7, max_workers=2, ordered="unordered", inflight_limit=3
    )
    query.to_parquet(
        tmp_path / "result", start=11, size=1, batch_size=4, parallel_options=options
    )
    assert seen == [
        {"start": 11, "size": 1, "mode": "dict", "parallel_options": options}
    ]


@pytest.mark.parametrize("single_file", [False, True])
def test_interrupt_after_publication_rename_restores_old_output(
    tmp_path, monkeypatch, single_file
):
    from intermine314.export import parquet

    target = tmp_path / "result"
    query_with_rows([{"Employee.id": 42}], ["Employee.id"]).to_parquet(
        target, single_file=single_file
    )
    monkeypatch.setattr(parquet, "_exchange_directories", lambda *args: False)
    original = parquet.os.replace
    interrupted = []

    def replace(source, destination):
        original(source, destination)
        if source.name == "output" and not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt("signal after rename")

    monkeypatch.setattr(parquet.os, "replace", replace)
    with pytest.raises(KeyboardInterrupt):
        query_with_rows([{"Employee.id": 1}], ["Employee.id"]).to_parquet(
            target, single_file=single_file
        )
    assert (
        pl.read_parquet(target if single_file else target / "part-00000.parquet").item()
        == 42
    )
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("single_file", [False, True])
def test_large_signed_unsigned_and_128_bit_integers_remain_exact(tmp_path, single_file):
    rows = [{"Employee.id": value} for value in [-(2**63), 2**64 - 1, 2**80 + 1]]
    target = tmp_path / "result"
    query_with_rows(rows, ["Employee.id"]).to_parquet(
        target, batch_size=1, single_file=single_file
    )
    result = pl.read_parquet(target if single_file else target / "part-*.parquet")
    assert result.to_dicts() == rows


def test_invalid_target_closes_an_already_open_producer(tmp_path):
    closed = []

    def batches(**kwargs):
        try:
            while True:
                yield [{"Employee.id": 1}]
        finally:
            closed.append(True)

    producer = batches()
    next(producer)
    query = query_with_rows([], ["Employee.id"])
    query.iter_batches = lambda **kwargs: producer
    target = tmp_path / "existing-file"
    target.write_text("keep")
    with pytest.raises(ValueError, match="directory"):
        query.to_parquet(target)
    assert closed == [True]
    assert target.read_text() == "keep"


def test_float_first_batch_cannot_round_a_later_large_integer(tmp_path):
    rows = [{"Employee.id": 1.5}, {"Employee.id": 9007199254740993}]
    with pytest.raises((ValueError, TypeError), match="precision|type"):
        query_with_rows(rows, ["Employee.id"]).to_parquet(
            tmp_path / "result", batch_size=2
        )
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("single_file", [False, True])
@pytest.mark.parametrize("container", ["list", "struct", "array"])
@pytest.mark.parametrize("schema_hook", [False, True])
def test_nested_int128_exports_as_exact_numeric_values(
    tmp_path, single_file, container, schema_hook
):
    import duckdb

    value = 2**80 + 1
    wrapped = {"n": value} if container == "struct" else [value]
    dtype = {
        "list": pl.List(pl.Int128),
        "struct": pl.Struct({"n": pl.Int128}),
        "array": pl.Array(pl.Int128, 1),
    }[container]
    rows = [{"Employee.id": wrapped}]
    query = query_with_rows(rows, ["Employee.id"])
    if schema_hook:
        query._parquet_schema = lambda: {"Employee.id": dtype}
    target = tmp_path / "result"
    query.to_parquet(target, single_file=single_file)
    path = target if single_file else target / "part-00000.parquet"
    assert pl.read_parquet(path).to_dicts() == rows
    with duckdb.connect(":memory:") as con:
        result = con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchone()[0]
        assert result == wrapped
        number = result["n"] if container == "struct" else result[0]
        assert isinstance(number, Decimal)


@pytest.mark.parametrize("single_file", [False, True])
@pytest.mark.parametrize("container", ["list", "struct"])
def test_nested_int128_decimal_overflow_preserves_previous_output(
    tmp_path, single_file, container
):
    import duckdb

    target = tmp_path / "result"
    query_with_rows([{"Employee.id": 42}], ["Employee.id"]).to_parquet(
        target, single_file=single_file
    )
    number = 10**38
    wrapped = {"n": number} if container == "struct" else [number]
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        query_with_rows([{"Employee.id": wrapped}], ["Employee.id"]).to_parquet(
            target, single_file=single_file
        )
    path = target if single_file else target / "part-00000.parquet"
    with duckdb.connect(":memory:") as con:
        assert con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchone() == (
            42,
        )
    assert list(tmp_path.iterdir()) == [target]
