"""Validate the working-tree evidence without rerunning or rewriting raw trials."""

from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from catalog import open_catalog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {'href', 'src'} and value and not value.startswith(('http:', 'https:', '#')):
                self.paths.append(value)


def validate():
    checks = []

    def check(name, passed, detail):
        checks.append(dict(check=name, status='passed' if passed else 'failed', detail=detail))

    original = pq.read_table(HERE / 'baseline' / 'source_inventory.parquet').to_pylist()
    runtime = [r for r in original if r['runtime']]
    check('baseline_runtime_unchanged', all(
        hashlib.sha256((ROOT / row['file']).read_bytes()).hexdigest() == row['sha256']
        for row in runtime), f'{len(runtime)} baseline runtime files unchanged; benchmark tooling may change')
    original_artifacts = pq.read_table(HERE / 'baseline' / 'artifact_manifest.parquet').to_pylist()
    observations = {'benchmark_pilot.parquet', 'benchmark_runs.parquet', 'benchmark_validation.parquet',
                    'conversion_profile.parquet', 'memory_comparison.parquet'}
    selected_artifacts = [row for row in original_artifacts if row['file'] in observations]
    check('raw_observation_hashes', len(selected_artifacts) == len(observations) and all(
        hashlib.sha256((HERE / row['file']).read_bytes()).hexdigest() == row['sha256']
        for row in selected_artifacts), 'Original measurements remain byte-for-byte unchanged')
    with open_catalog() as con:
        selected = con.sql('SELECT count(*) FROM selected_measurements').fetchone()[0]
        raw = con.sql('SELECT count(*) FROM all_measurements').fetchone()[0]
        check('trials', selected == 482 and raw == 689, f'{selected} selected / {raw} raw historical trials')
        opportunities = con.sql('SELECT id, evidence_ids FROM opportunities').fetchall()
        source_ids = {r[0] for r in con.sql('SELECT id FROM sources').fetchall()}
        check('opportunities', {r[0] for r in opportunities} == {f'P{i:02}' for i in range(1, 27) if i != 9}, '25 included IDs; P09 excluded by user')
        check('sources', all(set(ids) <= source_ids for _, ids in opportunities), 'All active evidence references resolve')
        tasks = con.sql('SELECT task_id, acceptance FROM implementation_tasks').fetchall()
        check('tasks', len(tasks) == 32 and len({r[0] for r in tasks}) == 32 and all(r[1] for r in tasks), '32 unique task definitions with acceptance checks')
        dispositions = con.sql('SELECT opportunity_id, disposition, tasks FROM opportunity_dispositions').fetchall()
        task_ids = {row[0] for row in tasks}
        check('coverage', len(dispositions) == 26 and all(ids and set(ids) <= task_ids for _, _, ids in dispositions), 'All 26 original opportunities have explicit task/disposition links')
        check('excluded_scope', [row[:2] for row in dispositions if row[1] == 'excluded_by_user'] == [('P09', 'excluded_by_user')], 'Excluded item retained only as scope accounting')
        views = con.sql('SELECT view_name, sql FROM duckdb_views() WHERE NOT internal').fetchall()
        check('relative_views', all(str(HERE) not in sql for _, sql in views), 'Catalog contains relative Parquet paths; open through catalog.open_catalog')
        for name, _ in views:
            con.execute('SELECT * FROM "' + name.replace('"', '""') + '" LIMIT 1').fetchall()
        check('duckdb_views', True, f'{len(views)} views queryable')
    parser = Links()
    html = (HERE / 'report.html').read_text()
    parser.feed(html)
    check('local_links', all((HERE / path).is_file() for path in parser.paths), f'{len(parser.paths)} report links resolve')
    check('scope_prose', all(token not in html for token in ('PyO3', 'Rust +', 'TypeScript')), 'Custom accelerator and additional language recommendations removed')
    check('truthful_status', 'Git metadata is writable' in html and 'implementation ledger' in html, 'Report renders task/phase status and discloses unimplemented runtime controls')
    paths = sorted(HERE.glob('*.parquet'))
    for path in paths:
        pq.read_table(path)
    check('parquet_tables', True, f'{len(paths)} current Parquet tables readable')
    pq.write_table(pa.Table.from_pylist(checks), HERE / 'validation.parquet', compression='zstd')
    artifacts = []
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path != HERE / 'artifact_manifest.parquet':
            content = path.read_bytes()
            artifacts.append(dict(file=str(path.relative_to(HERE)), bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    pq.write_table(pa.Table.from_pylist(artifacts), HERE / 'artifact_manifest.parquet', compression='zstd')
    print(json.dumps(checks))
    return all(row['status'] == 'passed' for row in checks)


if __name__ == '__main__':
    raise SystemExit(0 if validate() else 1)
