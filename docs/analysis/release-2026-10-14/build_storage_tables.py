"""Derive documentation and conditional job estimates from measured Parquet."""

import json
import math
import os
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE / 'storage'
DOCUMENT = HERE.parents[1] / 'source/storage_performance.rst'


def table(title, headers, rows):
    lines = [f'.. list-table:: {title}', '   :header-rows: 1', '']
    for row in [headers, *rows]:
        lines.append('   * - ' + str(row[0]))
        lines.extend('     - ' + str(value) for value in row[1:])
    return '\n'.join(lines) + '\n\n'


def main():
    previous = Path.cwd()
    try:
        os.chdir(DATA)
        with duckdb.connect('analysis.duckdb') as con:
            timings = con.execute('SELECT * FROM timings ORDER BY fixture, rows, stage').fetchall()
            columns = [description[0] for description in con.description]
            stats = {(r['fixture'], r['rows'], r['stage']): r for r in map(lambda row: dict(zip(columns, row)), timings)}
            assert all(row['repetitions'] == 7 for row in stats.values())
            assert con.execute("SELECT count(*) FROM trials WHERE correctness != 'passed'").fetchone()[0] == 0
            storage = con.execute('SELECT fixture, rows, baseline_format, baseline_bytes, parquet_zstd_bytes FROM storage').fetchall()
            env = json.loads(con.execute('SELECT environment_json FROM environment').fetchone()[0])
            pq.write_table(pa.Table.from_pylist(list(stats.values())), DATA / 'timings.parquet', compression='zstd')
            cases = []
            fixture = 'diverse_identifiers_sequences'
            for rows in (100_000, 1_000_000, 10_000_000, 100_000_000):
                measured_rows = min(rows, 1_000_000)
                scale = rows / measured_rows
                stage = {name: stats[fixture, measured_rows, name]['median_s'] * scale
                         for name in ('persist_csv', 'persist_parquet', 'query_csv_python', 'query_csv_duckdb', 'query_parquet_duckdb')}
                for workflow in ('python', 'duckdb'):
                    saved_per_query = stage['query_csv_' + workflow] - stage['query_parquet_duckdb']
                    break_even = (max(0, math.ceil((stage['persist_parquet'] - stage['persist_csv']) / saved_per_query))
                                  if saved_per_query > 0 else None)
                    for queries in (1, 10, 100):
                        common_fetch_s = rows / 100_000
                        csv_s = common_fetch_s + stage['persist_csv'] + queries * stage['query_csv_' + workflow]
                        parquet_s = common_fetch_s + stage['persist_parquet'] + queries * stage['query_parquet_duckdb']
                        cases.append(dict(fixture=fixture, rows=rows, analyses=queries, csv_analysis=workflow,
                                          basis_rows=measured_rows, local_basis='measured medians' if scale == 1 else 'linear extrapolation',
                                          assumed_fetch_rows_per_s=100_000, common_fetch_s=common_fetch_s,
                                          csv_total_s=csv_s, parquet_total_s=parquet_s, saved_s=csv_s-parquet_s,
                                          saved_fraction=1-parquet_s/csv_s, break_even_queries=break_even))
            pq.write_table(pa.Table.from_pylist(cases), DATA / 'job_estimates.parquet', compression='zstd')
            con.execute("CREATE OR REPLACE VIEW job_estimates AS SELECT * FROM read_parquet('job_estimates.parquet')")
            con.execute("CREATE OR REPLACE VIEW storage_savings AS SELECT *, 1-parquet_zstd_bytes::DOUBLE/baseline_bytes AS saved_fraction FROM storage")
    finally:
        os.chdir(previous)

    doc = '''Storage savings and job-time estimates
======================================

For the diverse one-million-row fixture, Zstandard Parquet used **61.7% less
space than CSV** and **20.7% less than gzip CSV**. A selective DuckDB aggregation
was **6.6 times faster** over Parquet than typed CSV. These are local synthetic
measurements, not a universal speedup for replacing the InterMine client.

What is compared
----------------

The original InterMine client can stream CSV, TSV and JSON and materialize a
pandas dataframe. It does not define one mandatory on-disk storage format.
GitHits verified those formats and the dataframe implementation against
`InterMine Python client 1.13.0 <https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/results.py#L320-L323>`_
and its
`dataframe method <https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query.py#L1444-L1467>`_.

The baseline here is an **InterMine-style CSV workflow** using Python's CSV
writer, followed by either a Python row aggregation or DuckDB over typed CSV.
The comparison uses intermine314's actual typed Parquet writer and
``query_parquet()``. It does not execute the original client, benchmark pandas,
measure HTTP download speed or compare the client's default iteration cost.
Both writers start with the same already available Python rows. A raw server
CSV stream could avoid this decoding/re-encoding cost. Parallel HTTP gains from
other benchmarks cannot be applied directly: the original client can use one
streaming response rather than one request per page.

Measured storage
----------------

Both fixtures contain eight columns: integer ID, identifier, length, score,
name, nullable integer, description and chromosome. The repetitive fixture has
reused annotations; the diverse fixture has deterministic random identifiers,
names, scores and 80-base sequences (seed 314). They illustrate compressibility
differences and do not establish a typical distribution for a real mine.
Sizes below are decimal MB; gzip uses level 6 and Parquet uses Zstandard with
the package's default 122,880-row target row group. CSV/TSV include headers and
UTF-8 LF records; JSON is compact arrays/objects without an HTTP status envelope.
All encoded values were checked. Text does not retain the same type/null schema
as Parquet; the benchmark supplies CSV types explicitly and retains leading zeros.

'''
    labels = {'repetitive_annotations': 'Repetitive', 'diverse_identifiers_sequences': 'Diverse'}
    storage_rows = []
    for fixture, rows, name, before, after in storage:
        if rows == 1_000_000:
            storage_rows.append([labels[fixture], name, f'{before/1e6:.3f}', f'{after/1e6:.3f}', f'{100*(1-after/before):.1f}%'])
    doc += table('Measured storage: 1,000,000 rows', ['Fixture', 'Text format', 'Text MB', 'Parquet MB', 'Space saved'], storage_rows)
    doc += '''At 100,000 rows, diverse CSV occupied 17.527 MB, gzip CSV 8.496 MB and
Parquet 6.783 MB: savings of 61.3% and 20.2%, respectively. All 32 measured
format/size combinations are retained in ``storage/storage.parquet``.
The much higher repetitive-data savings should not be used as a general forecast.
For data resembling the diverse million-row fixture, a **conditional** 100 GB
CSV estimate is 38.3 GB of Parquet; 100 GB of gzip CSV is 79.3 GB of Parquet.
Those large sizes were not written or measured.

Measured local processing
-------------------------

Seven repetitions per case were run in randomized order after a warm-up.
The table shows median seconds and the observed minimum–maximum range for the
diverse fixture. Query work groups by chromosome and computes count/sum of
length, projecting two of eight columns. Each DuckDB operation includes a fresh
connection; Parquet includes package validation and Arrow/Polars result
conversion. Writes include conversion, staging, ordered compaction and final
publication for Parquet; CSV includes row serialization. Neither writer calls
``fsync``. Import time, fixture generation and correctness checks are untimed.

'''
    stage_labels = {'persist_csv': 'Write CSV', 'persist_parquet': 'Write package Parquet',
                    'query_csv_python': 'CSV → Python aggregate', 'query_csv_duckdb': 'CSV → DuckDB aggregate',
                    'query_csv_gzip_duckdb': 'gzip CSV → DuckDB aggregate', 'query_parquet_duckdb': 'Parquet → package DuckDB aggregate'}
    rows = []
    for stage, label in stage_labels.items():
        row = [label]
        for count in (100_000, 1_000_000):
            s = stats['diverse_identifiers_sequences', count, stage]
            row.append(f'{s["median_s"]:.4f} ({s["min_s"]:.4f}–{s["max_s"]:.4f})')
        rows.append(row)
    doc += table('Measured seconds: median (min–max)', ['Stage', '100,000 rows', '1,000,000 rows'], rows)
    doc += '''At 100,000 diverse rows, Parquet writing was slower. A single CSV/DuckDB
analysis plus writing remained about **0.021 seconds faster** than Parquet;
two analyses amortized that extra Parquet write cost. At one million rows,
both Parquet writing and selective analysis were faster. The one-million-row
Parquet query was 45.6 times faster than gzip CSV/DuckDB and 57.1 times faster
than Python CSV aggregation in this fixture. These large ratios describe a
narrow, warm-cache projection, not every query or an entire download job.
Repeated queries against a preloaded DuckDB table, full-column scans, joins,
real sequence distributions and cold or remote storage require other measurements.

Estimated complete jobs
-----------------------

The following totals are **estimates**, even at measured sizes. They assume
both clients deliver 100,000 rows/second including server work, transfer and
decode, then write once and run the indicated number of local analyses::

    CSV total     = rows / assumed_fetch_rate + CSV_write + K * CSV_query
    Parquet total = rows / assumed_fetch_rate + Parquet_write + K * Parquet_query
    time saved    = CSV total - Parquet total

The common fetch rate is an explicit scenario, not an observed server rate.
No parallel-download advantage is assumed. Real acquisition costs can differ
between CSV and JSON and between clients. The 100,000/1,000,000-row local stages
use measured medians. The 10/100-million-row stages scale the million-row
medians linearly and are **unmeasured extrapolations**; memory pressure, storage,
cache limits and row-group geometry can invalidate that scaling. All totals
are seconds, rounded only for display; negative savings mean Parquet is slower.

'''
    for workflow, label in [('duckdb', 'CSV with DuckDB: isolates much of the format effect'),
                            ('python', 'CSV with Python: also replaces the analysis engine')]:
        rows = []
        for case in cases:
            if case['csv_analysis'] == workflow and case['analyses'] in (1, 10):
                rows.append([f'{case["rows"]:,}', case['local_basis'], case['analyses'],
                             f'{case["csv_total_s"]:.3f}', f'{case["parquet_total_s"]:.3f}',
                             f'{case["saved_s"]:.3f} ({100*case["saved_fraction"]:.1f}%)'])
        doc += table(label, ['Rows', 'Local basis', 'Analyses', 'CSV total s', 'Parquet total s', 'Saved s (%)'], rows)
    doc += f'''At one million rows and ten analyses, the assumed network-inclusive saving
is about **1.35 seconds (10.6%)** against CSV/DuckDB or **10.47 seconds (48.0%)**
against CSV/Python. Slower server delivery increases both totals and reduces
the percentage saved. For an already downloaded dataset, omit the common fetch
term. Gzip initial-write cost was not timed, so no gzip total-job estimate is
claimed. The Parquet estimates also exclude migrating an existing CSV archive.

Reproduce and inspect
---------------------

Measured on {env['python'].split()[0]}, Polars {env['versions']['polars']},
PyArrow {env['versions']['pyarrow']} and DuckDB {env['versions']['duckdb']},
on Linux/WSL2 with an Intel Core Ultra 7 265K. The guest exposes **18 CPUs**,
not all 20 host cores. Budget: 16 active native CPU workers; Polars 12,
Arrow CPU 4/I/O 2, DuckDB SQL 16 and ordered COPY 4, admitted by stage.
Files were on ext4, with warm OS cache; this is not a bare-host disk benchmark.
The fixture is materialized in memory for equality checks, so this benchmark
does not measure the package's bounded-streaming memory use.

From the repository root, choose an empty output directory::

    .venv/bin/python -m benchmarks.runners.storage_comparison \\
        --output /tmp/fresh-format-evidence \\
        --scratch-root /path/on/data/disk --rows 100000 1000000 --repeats 7

Canonical observations, 168 trials, timing summaries, 24 job scenarios and
environment/source hashes are Parquet in
``docs/analysis/release-2026-10-14/storage/``. ``analysis.duckdb`` holds relative
views: open it with that directory as the working directory. For example::

    import duckdb
    with duckdb.connect("analysis.duckdb", read_only=True) as con:
        print(con.sql("SELECT * FROM storage_savings").fetchall())
        print(con.sql("SELECT * FROM timings").fetchall())
        print(con.sql("SELECT * FROM job_estimates").fetchall())

``docs/analysis/release-2026-10-14/build_storage_tables.py`` regenerates this
page and forecasts from the committed observations. Earlier performance reports
retain their original dependency versions and measurements. Their old
intermine314 baseline is not the original InterMine client.
'''
    DOCUMENT.write_text(doc)
    print(f'Wrote {DOCUMENT}, {len(cases)} conditional scenarios')


if __name__ == '__main__':
    main()
