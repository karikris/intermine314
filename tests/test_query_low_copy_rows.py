from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from intermine314.query import builder as query_builder
from intermine314.service import session as session_module


def _json_stream_lines(rows):
    lines = [b'{"results":[']
    for idx, row in enumerate(rows):
        suffix = b"," if idx < len(rows) - 1 else b""
        lines.append(row + suffix)
    lines.append(b'],"wasSuccessful":true,"statusCode":200,"error":null}')
    return lines


class _LineConnection:
    def __init__(self, lines):
        self._iter = iter(lines)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def close(self):
        self.closed = True


class _Opener:
    def __init__(self, lines):
        self._lines = list(lines)
        self.calls = []
        self.connections = []

    def open(self, url, data=None, *_args, **_kwargs):
        self.calls.append((url, data))
        con = _LineConnection(self._lines)
        self.connections.append(con)
        return con


class _Service:
    def __init__(self, opener, *, version=8):
        self.version = int(version)
        self.root = "https://example.org/service"
        self.opener = opener


def test_result_iterator_dict_mode_schema_low_copy(monkeypatch):
    class _ExplodingResultRow:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("ResultRow should not be constructed in dict mode")

    monkeypatch.setattr(session_module, "ResultRow", _ExplodingResultRow)
    opener = _Opener(_json_stream_lines([b'["geneA",1]', b'["geneB",2]']))
    service = _Service(opener, version=8)
    it = session_module.ResultIterator(
        service,
        "/query/results",
        {"query": "xml"},
        "dict",
        ["Gene.symbol", "Gene.length"],
    )

    rows = list(it)

    assert rows == [
        {"Gene.symbol": "geneA", "Gene.length": 1},
        {"Gene.symbol": "geneB", "Gene.length": 2},
    ]


def test_result_iterator_list_mode_low_copy(monkeypatch):
    class _ExplodingResultRow:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("ResultRow should not be constructed in list mode")

    monkeypatch.setattr(session_module, "ResultRow", _ExplodingResultRow)
    opener = _Opener(_json_stream_lines([b'["geneA",1]']))
    service = _Service(opener, version=8)
    it = session_module.ResultIterator(
        service,
        "/query/results",
        {"query": "xml"},
        "list",
        ["Gene.symbol", "Gene.length"],
    )

    rows = list(it)
    assert rows == [["geneA", 1]]


def test_result_iterator_rr_mode_still_uses_resultrow(monkeypatch):
    class _CountingResultRow:
        created = 0

        def __init__(self, data, _views):
            type(self).created += 1
            self._data = data

        def __getitem__(self, key):
            if isinstance(key, int):
                return self._data[key]
            raise KeyError(key)

    monkeypatch.setattr(session_module, "ResultRow", _CountingResultRow)
    opener = _Opener(_json_stream_lines([b'["geneA",1]', b'["geneB",2]']))
    service = _Service(opener, version=8)
    it = session_module.ResultIterator(
        service,
        "/query/results",
        {"query": "xml"},
        "rr",
        ["Gene.symbol", "Gene.length"],
    )

    rows = list(it)
    assert len(rows) == 2
    assert _CountingResultRow.created >= 2


class _IterRowsHarness:
    def __init__(self):
        self.calls = []

    def _coerce_parallel_options(self, *, parallel_options=None):
        self.calls.append(("coerce", {"parallel_options": parallel_options}))
        return {"opts": True}

    def _iter_result_rows(self, **kwargs):
        self.calls.append(("rows", kwargs))
        row = kwargs["row"]
        if row == "dict":
            return iter([{"x": 1}, {"x": 2}])
        if row == "list":
            return iter([[1], [2]])
        return iter(["rr1", "rr2"])


def test_query_iter_rows_mode_dict():
    harness = _IterRowsHarness()
    rows = list(query_builder.Query.iter_rows(harness, mode="dict", start=3, size=2, parallel=True))

    assert rows == [{"x": 1}, {"x": 2}]
    mode_call = [payload for kind, payload in harness.calls if kind == "rows"][0]
    assert mode_call["row"] == "dict"
    assert mode_call["start"] == 3
    assert mode_call["size"] == 2
    assert mode_call["parallel"] is True


def test_query_iter_rows_invalid_mode():
    harness = _IterRowsHarness()
    with pytest.raises(ValueError, match="mode must be one of"):
        list(query_builder.Query.iter_rows(harness, mode="bad"))


class _IterBatchesHarness:
    def __init__(self):
        self.row_modes = []

    def iter_rows(self, **kwargs):
        self.row_modes.append(kwargs["mode"])
        for row in [{"a": 1}, {"a": 2}, {"a": 3}]:
            yield row


def test_iter_batches_uses_iter_rows_mode():
    harness = _IterBatchesHarness()
    batches = list(query_builder.Query.iter_batches(harness, batch_size=2, row_mode="dict"))

    assert harness.row_modes == ["dict"]
    assert batches == [[{"a": 1}, {"a": 2}], [{"a": 3}]]


class _FakeFrame:
    def __init__(self, rows, writes):
        self.rows = list(rows)
        self._writes = writes

    def write_parquet(self, path, compression):
        self._writes.append((path, compression, list(self.rows)))
        Path(path).write_bytes(b"part")


class _FakePolars:
    def __init__(self):
        self.writes = []

    def from_dicts(self, rows):
        return _FakeFrame(rows, self.writes)


class _FakePolarsInferAware:
    def __init__(self):
        self.writes = []
        self.infer_lengths = []

    def from_dicts(self, rows, *, infer_schema_length=100):
        self.infer_lengths.append(infer_schema_length)
        return _FakeFrame(rows, self.writes)


class _ParquetHarness:
    def __init__(self):
        self.iter_batches_calls = []

    def _coerce_parallel_options(self, *, parallel_options=None):
        _ = parallel_options
        return {"opts": True}

    def _iter_batches_kwargs(self, **kwargs):
        return query_builder.Query._iter_batches_kwargs(self, **kwargs)

    def iter_batches(self, **kwargs):
        self.iter_batches_calls.append(kwargs)
        yield [{"Gene.symbol": "geneA"}]


def test_to_parquet_uses_dict_row_mode_for_low_copy(monkeypatch):
    fake_polars = _FakePolars()
    harness = _ParquetHarness()
    monkeypatch.setattr(query_builder, "_require_polars", lambda _ctx: fake_polars)

    with TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "parts"
        query_builder.Query.to_parquet(harness, out_dir, single_file=False, batch_size=1)

    assert harness.iter_batches_calls
    call = harness.iter_batches_calls[0]
    assert call["row_mode"] == "dict"
    assert fake_polars.writes
    assert fake_polars.writes[0][2] == [{"Gene.symbol": "geneA"}]


def test_to_parquet_requests_full_batch_schema_inference_when_supported(monkeypatch):
    fake_polars = _FakePolarsInferAware()
    harness = _ParquetHarness()
    monkeypatch.setattr(query_builder, "_require_polars", lambda _ctx: fake_polars)

    with TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "parts"
        query_builder.Query.to_parquet(harness, out_dir, single_file=False, batch_size=1)

    assert fake_polars.infer_lengths == [None]


class _DataFrameHarness:
    def __init__(self, batches):
        self._batches = [list(batch) for batch in batches]

    def _coerce_parallel_options(self, *, parallel_options=None):
        _ = parallel_options
        return {"opts": True}

    def _iter_batches_kwargs(self, **kwargs):
        return query_builder.Query._iter_batches_kwargs(self, **kwargs)

    def iter_batches(self, **_kwargs):
        for batch in self._batches:
            yield batch


class _FakePolarsDataframePolicy:
    def __init__(self):
        self.concat_calls = []

    @staticmethod
    def from_dicts(rows):
        return {"rows": list(rows)}

    def concat(self, frames, *, how, rechunk):
        materialized = list(frames)
        self.concat_calls.append({"how": how, "rechunk": rechunk, "frames": materialized})
        return {"frames": materialized, "rechunk": rechunk}

    @staticmethod
    def DataFrame():
        return {"empty": True}


def test_query_dataframe_defaults_to_non_rechunked_concat(monkeypatch):
    fake_polars = _FakePolarsDataframePolicy()
    harness = _DataFrameHarness([[{"a": 1}], [{"a": 2}]])
    monkeypatch.setattr(query_builder, "_require_polars", lambda _ctx: fake_polars)

    result = query_builder.Query.dataframe(harness, batch_size=1)

    assert fake_polars.concat_calls
    call = fake_polars.concat_calls[0]
    assert call["how"] == "diagonal_relaxed"
    assert call["rechunk"] is False
    assert result["rechunk"] is False


def test_query_dataframe_allows_explicit_final_rechunk(monkeypatch):
    fake_polars = _FakePolarsDataframePolicy()
    harness = _DataFrameHarness([[{"a": 1}], [{"a": 2}]])
    monkeypatch.setattr(query_builder, "_require_polars", lambda _ctx: fake_polars)

    result = query_builder.Query.dataframe(harness, batch_size=1, final_rechunk=True)

    assert fake_polars.concat_calls
    call = fake_polars.concat_calls[0]
    assert call["rechunk"] is True
    assert result["rechunk"] is True


class _SummaryPath:
    class _End:
        def __init__(self, type_name):
            self.type_name = type_name

    def __init__(self, type_name):
        self.end = self._End(type_name)


class _SummaryModel:
    def __init__(self, type_name):
        self.type_name = type_name

    def make_path(self, *_args, **_kwargs):
        return _SummaryPath(self.type_name)


class _SummaryHarness:
    def __init__(self, type_name, rows):
        self.model = _SummaryModel(type_name)
        self._rows = list(rows)

    @staticmethod
    def prefix_path(path):
        return path

    @staticmethod
    def get_subclass_dict():
        return {}

    def results(self, *_args, **_kwargs):
        return iter(self._rows)


def test_summarise_normalizes_numeric_type_name_before_membership_check():
    harness = _SummaryHarness(
        " Integer ",
        [{"average": "1.5", "stdev": "0.5", "max": "2", "min": "1"}],
    )

    summary = query_builder.Query.summarise(harness, "Gene.length")

    assert summary == {"average": 1.5, "stdev": 0.5, "max": 2.0, "min": 1.0}


def test_summarise_keeps_non_numeric_counts_path():
    harness = _SummaryHarness(
        " java.lang.String ",
        [{"item": "Arabidopsis", "count": 3}, {"item": "Maize", "count": 1}],
    )

    summary = query_builder.Query.summarise(harness, "Gene.organism.name")

    assert summary == {"Arabidopsis": 3, "Maize": 1}
