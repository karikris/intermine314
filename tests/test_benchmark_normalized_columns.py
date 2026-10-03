"""Benchmark exports/reporting must use columns resolved by real query factories."""
import json
from pathlib import Path

import polars as pl
import pytest

from benchmarks import bench_fetch
from benchmarks import bench_storage_compare as storage


@pytest.fixture
def normalized_benchmark(monkeypatch, native_service_factory, offline_session_factory):
    values = {
        "Employee.age": [30, 40], "Employee.end": [None, "done"],
        "Employee.fullTime": [True, False], "Employee.id": [101, 102],
        "Employee.name": ["Ada", "Bea"],
    }
    def setup(raw_views):
        # Resolve through an actual offline Service, including inherited fields.
        model_service = native_service_factory()
        query = bench_fetch.make_query(lambda *a, **kw: model_service, model_service.root,
                                       "Employee", raw_views, [])
        columns = list(query.views)
        rows = [{column: values[column][i] for column in columns} for i in range(2)]
        body = ('{"results":[\n' + ',\n'.join(json.dumps(list(row.values())) for row in rows)
                + '\n],"wasSuccessful":true,"statusCode":200}\n').encode()
        def service(root, **kwargs):
            return native_service_factory(session=offline_session_factory(rows=body), **kwargs)
        monkeypatch.setattr(bench_fetch, "NewService", service)
        monkeypatch.setattr(storage, "get_legacy_service_class", lambda: service)
        monkeypatch.setattr(storage, "_configure_legacy_intermine_transport", lambda **kw: None)
        monkeypatch.setattr(storage, "count_with_retry", lambda *a, **kw: (2, 0, 0, 0))
        monkeypatch.setattr(storage, "_retriable_exceptions_for_mode", lambda mode: (OSError,))
        return columns, rows
    return setup


def arguments(tmp_path, raw_views):
    return dict(mine_url="https://offline.example/service", rows_target=2, page_size=2,
                parquet_path=tmp_path / "export.parquet", query_root_class="Employee",
                query_views=raw_views, query_joins=[], transport_mode="direct",
                tor_proxy_url_value=None, timeout_seconds=1)


@pytest.mark.parametrize("raw_views", [["name"], ["Employee.*"]])
def test_reference_export_uses_resolved_query_columns(normalized_benchmark, tmp_path, raw_views):
    columns, rows = normalized_benchmark(raw_views)
    result = storage._legacy_export_parquet(**arguments(tmp_path, raw_views), max_retries=2)
    assert result["columns"] == columns
    frame = pl.read_parquet(result["path"])
    assert frame.columns == columns
    assert frame.to_dicts() == rows


@pytest.mark.parametrize("raw_views", [["name"], ["Employee.*"]])
def test_modern_export_reports_resolved_query_columns(normalized_benchmark, tmp_path, raw_views):
    columns, rows = normalized_benchmark(raw_views)
    result = storage._modern_export_parquet(**arguments(tmp_path, raw_views), workers=1)
    assert result["columns"] == columns
    frame = pl.read_parquet(result["path"])
    assert frame.columns == columns
    assert frame.to_dicts() == rows


@pytest.mark.parametrize("raw_views", [["name"], ["Employee.*"]])
@pytest.mark.parametrize("reference_enabled", [True, False])
def test_report_samples_resolved_columns_and_preserves_reference_skip(
    normalized_benchmark, monkeypatch, tmp_path, raw_views, reference_enabled,
):
    columns, _rows = normalized_benchmark(raw_views)
    if reference_enabled:
        def reference(**kwargs):
            result = storage._legacy_export_parquet(**kwargs)
            return {"legacy_export": result, "legacy_polars_load": storage._load_parquet_polars(Path(result["path"]))}
        monkeypatch.setattr(storage, "_run_legacy_storage_subprocess", reference)
    else:
        monkeypatch.delenv("INTERMINE314_REFERENCE_PYTHON", raising=False)
    kwargs = arguments(tmp_path, raw_views)
    kwargs.pop("parquet_path")
    result = storage.run_storage_compare(**kwargs, workers=1, output_dir=tmp_path,
                                         max_retries=2, repetitions=1)
    run = result["runs"][0]
    assert run["modern_export_parquet"]["columns"] == columns
    assert run["modern_polars_load"]["rows"] == 2
    expected = True if reference_enabled else None
    assert run["parity"]["columns_match"] is expected
    assert result["parity"]["row_count_match_all"] is expected
    assert result["parity"]["sample_hash_match_all"] is expected
    if reference_enabled:
        assert run["legacy_export_parquet"]["columns"] == columns
    else:
        assert run["legacy_export_parquet"]["status"] == "skipped"
        assert run["parity"]["sample_hash_reference"] is None


def test_report_does_not_project_reference_names_onto_different_native_columns(monkeypatch, tmp_path):
    def reference(**kwargs):
        pl.DataFrame({"Employee.name": ["Ada"]}).write_parquet(kwargs["parquet_path"])
        return {"legacy_export": {"status": "ok", "columns": ["Employee.name"]},
                "legacy_polars_load": {"status": "ok", "rows": 1}}
    def modern(**kwargs):
        pl.DataFrame({"Employee.other": ["Ada"]}).write_parquet(kwargs["parquet_path"])
        return {"status": "ok", "columns": ["Employee.other"]}
    monkeypatch.setattr(storage, "_run_legacy_storage_subprocess", reference)
    monkeypatch.setattr(storage, "_modern_export_parquet", modern)
    kwargs = arguments(tmp_path, ["name"])
    kwargs.pop("parquet_path")
    result = storage.run_storage_compare(**kwargs, workers=1, output_dir=tmp_path,
                                         max_retries=2, repetitions=1)
    assert result["runs"][0]["parity"]["columns_match"] is False
    assert result["parity"]["row_count_match_all"] is True
    assert result["parity"]["sample_hash_match_all"] is False
