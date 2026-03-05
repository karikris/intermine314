import json
import os
import subprocess
import sys
import inspect
from pathlib import Path

import intermine314.export.fetch as export_fetch
import intermine314.query.builder as query_builder
import intermine314.config.storage_policy as storage_policy
import intermine314.service.iterators as service_iterators


def test_storage_policy_is_single_sourced_for_query_and_export():
    assert query_builder._validate_parquet_compression is storage_policy.validate_parquet_compression
    assert export_fetch.validate_parquet_compression is storage_policy.validate_parquet_compression
    assert query_builder._validate_duckdb_identifier is storage_policy.validate_duckdb_identifier
    assert export_fetch.validate_duckdb_identifier is storage_policy.validate_duckdb_identifier


def test_runtime_import_does_not_pull_benchmark_only_dependencies():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    cmd = [
        sys.executable,
        "-c",
        (
            "import json,sys; import intermine314; "
            "mods=sorted(m for m in sys.modules if m=='intermine' or m.startswith('intermine.') "
            "or m=='pandas' or m.startswith('pandas.')); "
            "print(json.dumps(mods))"
        ),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads((proc.stdout or "[]").strip() or "[]")
    assert loaded == []


def test_fetch_from_mine_removes_policy_profiles_and_keeps_parallel_contract():
    params = list(inspect.signature(export_fetch.fetch_from_mine).parameters.keys())
    assert "workflow" not in params
    assert "production_profile" not in params
    assert "resource_profile" not in params
    assert "parallel_profile" not in params
    assert "large_query_mode" not in params

    for expected in [
        "mine_url",
        "root_class",
        "views",
        "page_size",
        "max_workers",
        "ordered",
        "prefetch",
        "inflight_limit",
        "max_inflight_bytes_estimate",
        "parquet_path",
        "duckdb_table",
        "managed",
    ]:
        assert expected in params


def test_legacy_object_row_modes_are_removed():
    assert "rr" not in query_builder.VALID_ITER_ROW_MODES
    assert "rr" not in query_builder.VALID_RESULT_ROW_MODES
    assert "list" not in service_iterators.ResultIterator.ROW_FORMATS
    assert "rr" not in service_iterators.ResultIterator.ROW_FORMATS
