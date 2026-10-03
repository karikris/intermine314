"""Offline checks for the owned benchmark's real storage and reference boundary."""
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from benchmarks import bench_fetch
from benchmarks import bench_storage_compare as storage

ROWS = [
    {"Gene.id": 2**53 + 1, "Gene.value": Decimal("1.234567890123456789"), "Gene.code": "0007"},
    {"Gene.id": 2**53 + 2, "Gene.value": None, "Gene.code": "0008"},
    {"Gene.id": 2**53 + 3, "Gene.value": Decimal("9.2"), "Gene.code": "0009"},
]


class Page:
    def __init__(self, rows, fail=False):
        self.rows = iter(rows)
        self.fail = fail
        self.closed = False
        self.read = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.fail and self.read == 1:
            raise OSError("connection interrupted after a row")
        self.read += 1
        return next(self.rows)

    def close(self):
        self.closed = True


class Query:
    views = list(ROWS[0])

    def __init__(self, fail_calls=()):
        self.pages = []
        self.calls = []
        self.fail_calls = fail_calls

    def results(self, *, row, start, size):
        assert row == "dict"
        self.calls.append((start, size))
        page = Page(ROWS[start:start + size], len(self.calls) in self.fail_calls)
        self.pages.append(page)
        # The returned iterable and its actual iterator need not be identical.
        return IterablePage(page)


class IterablePage:
    def __init__(self, page):
        self.page = page

    def __iter__(self):
        return self.page


def export(monkeypatch, tmp_path, query, *, retries=2):
    monkeypatch.setattr(storage, "get_legacy_service_class", lambda: object)
    monkeypatch.setattr(storage, "_configure_legacy_intermine_transport", lambda **kw: None)
    monkeypatch.setattr(storage, "make_query", lambda *a, **kw: query)
    monkeypatch.setattr(storage, "count_with_retry", lambda *a, **kw: (3, 0, 0, 0))
    monkeypatch.setattr(storage, "_retry_wait_seconds", lambda attempt: 0)
    monkeypatch.setattr(storage, "_retriable_exceptions_for_mode", lambda mode: (OSError,))
    return storage._legacy_export_parquet(
        mine_url="https://example.test/service", rows_target=3, page_size=2,
        parquet_path=tmp_path / "reference.parquet", query_root_class="Gene",
        query_views=list(ROWS[0]), query_joins=[], transport_mode="direct",
        tor_proxy_url_value=None, timeout_seconds=1, max_retries=retries,
    )


def test_reference_retry_commits_only_complete_pages_and_preserves_values(monkeypatch, tmp_path):
    query = Query(fail_calls=(1,))
    result = export(monkeypatch, tmp_path, query)
    frame = pl.read_parquet(result["path"])
    assert frame.to_dicts() == ROWS
    assert frame.schema["Gene.id"] == pl.Int64
    assert result["rows"] == 3
    assert result["retries"] == 1
    assert query.calls == [(0, 2), (0, 2), (2, 1)]
    assert all(page.closed for page in query.pages)
    assert [p.name for p in tmp_path.iterdir()] == ["reference.parquet"]


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_failed_reference_export_preserves_destination_and_closes_iterator(monkeypatch, tmp_path, failure):
    path = tmp_path / "reference.parquet"
    path.write_bytes(b"previous output")
    query = Query(fail_calls=(2, 3))
    if failure is KeyboardInterrupt:
        original = Page.__next__
        def interrupted(page):
            if page.fail:
                raise KeyboardInterrupt()
            return original(page)
        monkeypatch.setattr(Page, "__next__", interrupted)
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        export(monkeypatch, tmp_path, query)
    assert path.read_bytes() == b"previous output"
    assert all(page.closed for page in query.pages)
    assert list(tmp_path.iterdir()) == [path]


def test_sample_count_hash_and_bounded_arrow_polars_read(monkeypatch, tmp_path):
    path = tmp_path / "values.parquet"
    pl.DataFrame(ROWS).write_parquet(path)
    batches = []
    real_from_arrow = pl.from_arrow
    def from_arrow(value, **kwargs):
        frame = real_from_arrow(value, **kwargs)
        batches.append(frame.height)
        return frame
    monkeypatch.setattr(pl, "from_arrow", from_arrow)
    loaded = storage._load_parquet_polars(path, batch_size=2)
    assert loaded["rows"] == 3
    assert max(batches) <= 2
    assert loaded["peak_batch_memory_bytes"] > 0
    sample = storage._sample_parquet_rows(path, columns=list(ROWS[0]), sample_size=2)
    assert sample == ROWS[:2]
    assert storage._sha256_rows(sample, list(ROWS[0])) == storage._sha256_rows(ROWS[:2], list(ROWS[0]))
    assert storage._sample_parquet_rows(path, columns=list(ROWS[0]), sample_size=0) == []
    assert storage._load_modern_duckdb(path)["rows"] == 3


@pytest.mark.parametrize("module, name, payload", [
    (bench_fetch, "_run_legacy_mode_subprocess", {"run": {"rows": 3}}),
    (storage, "_run_legacy_storage_subprocess", {"legacy_export": {"status": "ok"}}),
])
def test_reference_uses_explicit_interpreter(monkeypatch, tmp_path, module, name, payload):
    import json
    selected = str(tmp_path / "reference-env/bin/python")
    monkeypatch.setenv("INTERMINE314_REFERENCE_PYTHON", selected)
    monkeypatch.setenv("PYTHONPATH", "/unrelated/site-packages")
    calls = []
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    kwargs = dict(mine_url="https://example.test/service", rows_target=3,
                  page_size=2, query_root_class="Gene", query_views=[], query_joins=[])
    if module is bench_fetch:
        kwargs["runtime_kwargs"] = {}
    else:
        kwargs.update(parquet_path=tmp_path / "reference.parquet", transport_mode="direct",
                      tor_proxy_url_value=None, timeout_seconds=1, max_retries=2)
    getattr(module, name)(**kwargs)
    assert calls[0][0][0] == selected
    assert "/unrelated/site-packages" not in calls[0][1]["env"]["PYTHONPATH"]
    assert calls[0][1]["env"]["PYTHONNOUSERSITE"] == "1"


def test_reference_fetch_requires_environment_selection(monkeypatch):
    monkeypatch.delenv("INTERMINE314_REFERENCE_PYTHON", raising=False)
    with pytest.raises(RuntimeError, match="INTERMINE314_REFERENCE_PYTHON"):
        bench_fetch._run_legacy_mode_subprocess(
            mine_url="https://example.test/service", rows_target=3, page_size=2,
            runtime_kwargs={}, query_root_class="Gene", query_views=[], query_joins=[],
        )


@pytest.mark.parametrize("values", [{"rows_target": -1}, {"page_size": 0}, {"max_retries": 0}])
def test_invalid_reference_sizes_fail_before_import_or_io(monkeypatch, tmp_path, values):
    def unexpected():
        pytest.fail("reference import must follow argument validation")
    monkeypatch.setattr(storage, "get_legacy_service_class", unexpected)
    kwargs = dict(mine_url="https://example.test/service", rows_target=3, page_size=2,
                  parquet_path=tmp_path / "reference.parquet", query_root_class="Gene",
                  query_views=[], query_joins=[], transport_mode="direct",
                  tor_proxy_url_value=None, timeout_seconds=1, max_retries=2)
    kwargs.update(values)
    with pytest.raises(ValueError):
        storage._legacy_export_parquet(**kwargs)


@pytest.mark.parametrize("reference_enabled", [True, False])
def test_storage_report_measures_actual_files_and_marks_optional_skip(monkeypatch, tmp_path, reference_enabled):
    def modern(**kwargs):
        # Native completion order can differ; sample comparison sorts in SQL.
        pl.DataFrame(list(reversed(ROWS))).write_parquet(kwargs["parquet_path"])
        return {"status": "ok", "seconds": 0.1, "columns": list(ROWS[0])}
    monkeypatch.setattr(storage, "_modern_export_parquet", modern)
    if reference_enabled:
        def reference(**kwargs):
            result = export(monkeypatch, tmp_path, Query())
            Path(result["path"]).rename(kwargs["parquet_path"])
            return {"legacy_export": result, "legacy_polars_load": storage._load_parquet_polars(kwargs["parquet_path"])}
        monkeypatch.setattr(storage, "_run_legacy_storage_subprocess", reference)
    else:
        monkeypatch.delenv("INTERMINE314_REFERENCE_PYTHON", raising=False)
    result = storage.run_storage_compare(
        mine_url="https://example.test/service", rows_target=3, page_size=2, workers=1,
        query_root_class="Gene", query_views=list(ROWS[0]), query_joins=[],
        transport_mode="direct", tor_proxy_url_value=None, output_dir=tmp_path,
        timeout_seconds=1, max_retries=2, repetitions=1,
    )
    assert result["schema_version"] == "parquet_storage_compare_v3"
    assert result["runs"][0]["modern_polars_load"]["rows"] == 3
    expected = True if reference_enabled else None
    assert result["parity"] == {"row_count_match_all": expected, "sample_hash_match_all": expected}
    assert not list(tmp_path.glob("*.csv"))
    if not reference_enabled:
        assert result["runs"][0]["legacy_export_parquet"]["status"] == "skipped"
        assert not Path(result["runs"][0]["artifacts"]["reference_parquet_path"]).exists()


@pytest.mark.parametrize("reference_enabled", [False, True])
def test_cli_reference_is_optional_and_reported(monkeypatch, capsys, reference_enabled):
    import json

    from benchmarks import benchmarks as cli
    if reference_enabled:
        monkeypatch.setenv("INTERMINE314_REFERENCE_PYTHON", "/reference/bin/python")
    else:
        monkeypatch.delenv("INTERMINE314_REFERENCE_PYTHON", raising=False)
    plans = []
    def fetch(**kwargs):
        plans.append(kwargs["phase_plan"])
        return {}
    monkeypatch.setattr(cli, "run_fetch_phase", fetch)
    assert cli.main(["--mine-url", "https://example.test/service", "--transport-modes", "direct",
                     "--workers", "1", "--no-storage-compare"]) == 0
    assert all(plan["include_legacy_baseline"] is reference_enabled for plan in plans)
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime"]["reference_baseline"] == ("enabled" if reference_enabled else "skipped")


def test_cli_explicit_reference_requires_selected_environment(monkeypatch):
    from benchmarks import benchmarks as cli
    monkeypatch.delenv("INTERMINE314_REFERENCE_PYTHON", raising=False)
    with pytest.raises(RuntimeError, match="INTERMINE314_REFERENCE_PYTHON"):
        cli.main(["--legacy-baseline"])


def test_benchmark_query_selects_only_requested_columns(native_service_factory):
    service = native_service_factory()
    query = bench_fetch.make_query(
        lambda *a, **kw: service, service.root, "Employee",
        ["Employee.name", "Employee.age"], [],
    )
    assert query.views == ["Employee.name", "Employee.age"]


@pytest.mark.parametrize("target_parts,sibling_parts", [
    (("run[1]", "modern.parquet"), ("run1", "modern.parquet")),
    (("run*", "modern.parquet"), ("run-other", "modern.parquet")),
    (("run", "modern[1].parquet"), ("run", "modern1.parquet")),
    (("run", "modern*.parquet"), ("run", "modern-other.parquet")),
])
def test_count_load_and_sample_read_only_literal_parquet_path(tmp_path, target_parts, sibling_parts):
    target = tmp_path.joinpath(*target_parts)
    sibling = tmp_path.joinpath(*sibling_parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    sibling.parent.mkdir(parents=True, exist_ok=True)
    expected = [{"id": 2**53 + 1, "code": "0007"}, {"id": 2**53 + 2, "code": "0008"}]
    pl.DataFrame(expected).write_parquet(target)
    pl.DataFrame({"id": list(range(9)), "code": ["sibling"] * 9}).write_parquet(sibling)

    loaded = storage._load_parquet_polars(target, batch_size=1)
    sample = storage._sample_parquet_rows(target, columns=["id", "code"])
    count = storage._load_modern_duckdb(target)

    assert loaded["rows"] == 2
    assert sample == expected
    assert count["rows"] == loaded["rows"] == len(sample)
