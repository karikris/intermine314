"""Measure local InterMine-style text workflows against package Parquet.

No HTTP or original-client execution is timed. Canonical evidence is Parquet;
temporary text fixtures are deleted. Run from an editable checkout, for example:
python -m benchmarks.runners.storage_comparison --output /tmp/format-evidence \
    --scratch-root /path/on/data/disk --rows 100000 1000000 --repeats 7
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ['Record.id', 'Record.identifier', 'Record.length', 'Record.score',
           'Record.name', 'Record.optional', 'Record.description', 'Record.chromosome']
SQL = ('SELECT "Record.chromosome", count(*)::BIGINT AS n, '
       'sum("Record.length")::BIGINT AS total FROM results GROUP BY 1 ORDER BY 1')
CSV_TYPES = dict(zip(COLUMNS, ['BIGINT', 'VARCHAR', 'BIGINT', 'DOUBLE',
                             'VARCHAR', 'BIGINT', 'VARCHAR', 'VARCHAR']))


def fixture_rows(count, diverse):
    rng = random.Random(314)
    rows = []
    for i in range(count):
        values = [i, f'{i:012d}', i % 5000, (i % 1009) / 11,
                  f'gene_{i % 10000}', None if i % 7 == 0 else i % 31,
                  'Müller 🧬 ' + 'ACGT' * 12, f'chr{i % 10}']
        if diverse:
            values[1] = rng.randbytes(16).hex()
            values[4] = rng.randbytes(12).hex()
            values[6] = ''.join(rng.choices('ACGT', k=80))
            values[3] = rng.random() * 1000
        rows.append(dict(zip(COLUMNS, values)))
    return rows


def write_text(rows, target, delimiter=','):
    with target.open('w', encoding='utf8', newline='') as handle:
        writer = csv.writer(handle, delimiter=delimiter, lineterminator='\n')
        writer.writerow(COLUMNS)
        writer.writerows([row[name] for name in COLUMNS] for row in rows)


def write_json(rows, target, objects):
    with target.open('w', encoding='utf8') as handle:
        handle.write('[')
        for index, row in enumerate(rows):
            if index:
                handle.write(',')
            handle.write(json.dumps(row if objects else list(row.values()),
                                    ensure_ascii=False, separators=(',', ':')))
        handle.write(']')


def csv_python(path):
    values = {}
    with path.open(newline='', encoding='utf8') as handle:
        for row in csv.DictReader(handle):
            key = row['Record.chromosome']
            count, total = values.get(key, (0, 0))
            values[key] = count + 1, total + int(row['Record.length'])
    return sorted((key, *value) for key, value in values.items())


def source_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'src/intermine314').rglob('*')):
        if path.suffix in ('.py', '.toml'):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run(args):
    # Select the native pool before any analytical dependency import.
    os.environ['POLARS_MAX_THREADS'] = '12'
    import duckdb
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    from intermine314.export import ResourceProfile, ResourceRuntime, query_parquet
    from intermine314.export.model_schema import ModelSchema
    from intermine314.export.parquet import write_parquet_batches

    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Evidence output directory must be empty')
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    source_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    env = dict(measured_at=datetime.now(UTC).isoformat(), source_sha=source_sha,
               source_digest=source_digest(), benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               python=sys.version, platform=platform.platform(), cpu_count=os.cpu_count(),
               process_cpu_count=os.process_cpu_count(), gil_enabled=sys._is_gil_enabled(),
               versions={name: version(name) for name in ('polars', 'pyarrow', 'duckdb', 'intermine314')},
               scratch_root=str(args.scratch_root.resolve()), repeats=args.repeats,
               filesystem=subprocess.check_output(['df', '-T', str(args.scratch_root)], text=True),
               cpu_model=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                               if line.startswith('model name')), 'unknown'),
               scope='Synthetic local persistence and warm-cache aggregation; excludes HTTP, imports, fixture generation, fsync and original-client execution')
    storage, trials = [], []
    for count in args.rows:
        for fixture, diverse in [('repetitive_annotations', False), ('diverse_identifiers_sequences', True)]:
            rows = fixture_rows(count, diverse)
            expected = {}
            fingerprint = hashlib.sha256()
            for row in rows:
                key = row['Record.chromosome']
                n, total = expected.get(key, (0, 0))
                expected[key] = n + 1, total + row['Record.length']
                fingerprint.update(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode())
                fingerprint.update(b'\n')
            expected = sorted((key, *value) for key, value in expected.items())
            result_hash = hashlib.sha256(repr(expected).encode()).hexdigest()
            schema = ModelSchema(dict(zip(COLUMNS, [pl.Int64, pl.String, pl.Int64, pl.Float64,
                                                   pl.String, pl.Int64, pl.String, pl.String])), ())
            with TemporaryDirectory(prefix='intermine-format-', dir=args.scratch_root) as temp:
                folder = Path(temp)
                profile = ResourceProfile(cpu_budget=16, polars_threads=12, arrow_cpu_threads=4,
                                          arrow_io_threads=2, duckdb_threads=16, temp_dir=folder)
                with ResourceRuntime(profile) as runtime:
                    paths = {name: folder / ('data.' + name) for name in ('csv', 'tsv', 'json_rows', 'json_objects')}
                    target = folder / 'data.parquet'

                    def persist_parquet():
                        # Starts with the same Python rows as the CSV writer;
                        # includes typed Arrow conversion, staging and COPY.
                        with runtime.activate():
                            write_parquet_batches(
                                batches=(rows[start:start + 65536] for start in range(0, count, 65536)),
                                target=target, columns=COLUMNS, polars_module=pl, schema=schema,
                                compression='zstd', single_file=True, staging_dir=folder,
                            )

                    def csv_duckdb(path, sql=SQL, delimiter=','):
                        with runtime.native('duckdb_sql'), duckdb.connect(config=runtime.duckdb_config()) as con:
                            con.read_csv(str(path), header=True, columns=CSV_TYPES, delimiter=delimiter).create_view('results')
                            return con.execute(sql).fetchall()

                    write_text(rows, paths['csv'])
                    write_text(rows, paths['tsv'], '\t')
                    write_json(rows, paths['json_rows'], False)
                    write_json(rows, paths['json_objects'], True)
                    persist_parquet()
                    # Check every typed Parquet and text value, not only an aggregate.
                    assert pl.read_parquet(target).equals(pl.from_dicts(rows, schema=dict(schema)))
                    expected_rows = [tuple(row.values()) for row in rows]
                    for name in ('csv', 'tsv'):
                        assert csv_duckdb(paths[name], 'SELECT * FROM results', '\t' if name == 'tsv' else ',') == expected_rows
                    del expected_rows
                    for name in ('json_rows', 'json_objects'):
                        with paths[name].open(encoding='utf8') as handle:
                            decoded = json.load(handle)
                        if name == 'json_objects':
                            assert decoded == rows
                        else:
                            assert all(actual == list(original.values()) for actual, original in zip(decoded, rows, strict=True))
                        del decoded
                    for name, path in list(paths.items()):
                        compressed = folder / (path.name + '.gz')
                        with path.open('rb') as source, compressed.open('wb') as raw:
                            with gzip.GzipFile(fileobj=raw, mode='wb', filename='', compresslevel=6, mtime=0) as output:
                                shutil.copyfileobj(source, output)
                        with gzip.open(compressed, 'rb') as source, path.open('rb') as original:
                            assert hashlib.file_digest(source, 'sha256').digest() == hashlib.file_digest(original, 'sha256').digest()
                        paths[name + '.gz'] = compressed
                    for name, path in paths.items():
                        storage.append(dict(fixture=fixture, rows=count, columns=len(COLUMNS), baseline_format=name,
                                            baseline_bytes=path.stat().st_size, parquet_zstd_bytes=target.stat().st_size,
                                            input_sha256=fingerprint.hexdigest(), correctness='all encoded values checked',
                                            source_sha=source_sha))
                    cases = {
                        'persist_csv': lambda: write_text(rows, paths['csv']),
                        'persist_parquet': persist_parquet,
                        'query_csv_python': lambda: csv_python(paths['csv']),
                        'query_csv_duckdb': lambda: csv_duckdb(paths['csv']),
                        'query_csv_gzip_duckdb': lambda: csv_duckdb(paths['csv.gz']),
                        'query_parquet_duckdb': lambda: query_parquet(target, SQL, runtime=runtime).rows(),
                    }
                    for name, action in cases.items():
                        value = action()
                        if name.startswith('query'):
                            assert value == expected
                    randomizer = random.Random(314)
                    for repeat in range(args.repeats):
                        order = list(cases)
                        randomizer.shuffle(order)
                        for name in order:
                            gc.collect()
                            begin_cpu = time.process_time()
                            begin = time.perf_counter()
                            value = cases[name]()
                            wall = time.perf_counter() - begin
                            cpu = time.process_time() - begin_cpu
                            if name.startswith('query'):
                                assert value == expected
                            elif name == 'persist_parquet':
                                assert pl.read_parquet(target).equals(pl.from_dicts(rows, schema=dict(schema)))
                            else:
                                assert csv_duckdb(paths['csv']) == expected
                            trials.append(dict(fixture=fixture, rows=count, stage=name, repeat=repeat,
                                               wall_s=wall, cpu_s=cpu, cpu_budget=runtime.cpu_budget,
                                               polars_threads=pl.thread_pool_size(), duckdb_threads=runtime.duckdb_config()['threads'],
                                               arrow_cpu_threads=pa.cpu_count(), arrow_io_threads=pa.io_thread_count(),
                                               checksum=result_hash, correctness='passed', source_sha=source_sha))
                    env['effective_pools'] = runtime.diagnostics()
            rows.clear()
            gc.collect()
            print(f'Checked {fixture}: {count:,} rows, {args.repeats} repetitions', flush=True)
    for name, records in [('storage', storage), ('trials', trials)]:
        pq.write_table(pa.Table.from_pylist(records), args.output / f'{name}.parquet', compression='zstd')
    pq.write_table(pa.Table.from_pylist([dict(environment_json=json.dumps(env, sort_keys=True, default=str))]),
                   args.output / 'environment.parquet', compression='zstd')
    from benchmarks.evidence import open_catalog

    with open_catalog(args.output) as con:
        timings = con.sql("""SELECT fixture, rows, stage, count(*) AS repetitions,
                             median(wall_s) AS median_s, min(wall_s) AS min_s, max(wall_s) AS max_s,
                             quantile_cont(wall_s, 0.1) AS p10_s, quantile_cont(wall_s, 0.9) AS p90_s
                             FROM trials GROUP BY ALL""").to_arrow_table()
        pq.write_table(timings, args.output / 'timings.parquet', compression='zstd')
    print(f'Evidence written to {args.output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scratch-root', required=True, type=Path)
    parser.add_argument('--rows', nargs='+', type=int, default=[100_000, 1_000_000])
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if min(args.rows) <= 0 or args.repeats < 3 or len(set(args.rows)) != len(args.rows):
        parser.error('Choose distinct positive row counts and at least three repetitions')
    run(args)


if __name__ == '__main__':
    main()
