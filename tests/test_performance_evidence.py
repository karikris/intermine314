"""Check the portable Parquet evidence contract, without running benchmarks."""

import importlib.util
import shutil
from pathlib import Path

import pyarrow.parquet as pq

EVIDENCE = Path(__file__).resolve().parents[1] / 'docs/analysis/performance-2026-10-13'


def test_catalog_relocation_uses_copied_parquet(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('evidence_catalog', EVIDENCE / 'catalog.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "relocated evidence's directory"
    shutil.copytree(EVIDENCE, target, ignore=shutil.ignore_patterns('__pycache__'))
    # Mutate only the relocated data so an absolute view cannot pass this test.
    trial_file = target / 'benchmark_validation.parquet'
    table = pq.read_table(trial_file)
    pq.write_table(table.slice(0, 1), trial_file)
    monkeypatch.chdir(tmp_path)
    with module.open_catalog(target) as con:
        assert con.sql('SELECT count(*) FROM benchmark_validation').fetchone() == (1,)
        assert con.sql('SELECT count(*) FROM opportunities').fetchone() == (25,)
        assert con.sql('SELECT count(*) FROM implementation_tasks').fetchone() == (32,)
        assert all(str(EVIDENCE) not in r[0] for r in con.sql('SELECT sql FROM duckdb_views() WHERE NOT internal').fetchall())


def test_ledger_links_accepted_phases_to_successful_ci():
    tasks = pq.read_table(EVIDENCE / 'implementation_tasks.parquet').to_pylist()
    assert {row['phase'] for row in tasks} == set(range(1, 11))
    phases = pq.read_table(EVIDENCE / 'phase_status.parquet').to_pylist()
    for row in phases:
        if row['status'] == 'accepted':
            assert row['push_status'] == 'verified' and row['ci_status'] == 'success'
            assert row['head_commit'] and row['ci_url']
    scope = pq.read_table(EVIDENCE / 'opportunity_dispositions.parquet').to_pylist()
    assert [row['opportunity_id'] for row in scope if row['disposition'] == 'excluded_by_user'] == ['P09']
    assert all(row['tasks'] for row in scope)
