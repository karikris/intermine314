"""Check maintained measurement portability, without rerunning benchmarks."""
import re
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.evidence import open_catalog

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/analysis/evidence'


def test_catalog_relocation_uses_copied_parquet(tmp_path, monkeypatch):
    target = tmp_path / "relocated evidence's directory"
    shutil.copytree(EVIDENCE, target)
    trial_file = target / 'storage/trials.parquet'
    table = pq.read_table(trial_file)
    pq.write_table(table.slice(0, 1), trial_file)
    monkeypatch.chdir(tmp_path)
    with open_catalog(target) as con:
        assert con.sql('SELECT count(*) FROM storage_trials').fetchone() == (1,)
        assert all(str(EVIDENCE) not in r[0] for r in con.sql('SELECT sql FROM duckdb_views() WHERE NOT internal').fetchall())
        for name, in con.sql('SHOW TABLES').fetchall():
            con.table(name).count('*').fetchone()
    assert not list(target.rglob('*.duckdb'))
    assert pq.read_table(EVIDENCE / 'storage/trials.parquet').num_rows == table.num_rows


def test_ambiguous_catalog_names_fail_instead_of_replacing_evidence(tmp_path):
    for name in ('a-b', 'a_b'):
        pq.write_table(pa.table({'value': [1]}), tmp_path / f'{name}.parquet')
    with pytest.raises(ValueError, match='Ambiguous evidence table name'):
        open_catalog(tmp_path)


def test_accepted_phase_receipts_link_exact_heads_to_successful_ci():
    rows = pq.read_table(ROOT / 'docs/analysis/cleanup/ci-receipts.parquet').to_pylist()
    assert rows
    for row in rows:
        assert row['workflow_conclusion'] == row['conclusion'] == 'success'
        assert re.fullmatch('[0-9a-f]{40}', row['head'])
        assert re.fullmatch(r'https://github.com/karikris/intermine314/actions/runs/\d+', row['url'])
        assert row['job']
