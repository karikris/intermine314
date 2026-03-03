import json
import subprocess
import sys

import intermine314.config.storage_policy as storage_policy
import intermine314.export.fetch as export_fetch
import intermine314.query.builder as query_builder


def test_storage_policy_is_single_sourced_for_query_and_export():
    assert query_builder._validate_parquet_compression is storage_policy.validate_parquet_compression
    assert export_fetch.validate_parquet_compression is storage_policy.validate_parquet_compression
    assert query_builder._validate_duckdb_identifier is storage_policy.validate_duckdb_identifier
    assert export_fetch.validate_duckdb_identifier is storage_policy.validate_duckdb_identifier


def test_runtime_import_does_not_pull_benchmark_only_dependencies():
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
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads((proc.stdout or "[]").strip() or "[]")
    assert loaded == []
