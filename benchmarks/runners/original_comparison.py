"""Installed original InterMine versus the current columnar pipeline.

Synthetic, deterministic loopback data; no public-mine throughput claims.
Each timed operation runs in a fresh isolated interpreter. Fixture generation,
imports, metadata discovery and full-value verification are outside timing.
RSS is the Linux process high-water mark, including imports/native allocations.
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import csv
import gc
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from array import array
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['id', 'identifier', 'length', 'score', 'name', 'optional', 'description', 'chromosome']
COLUMNS = ['Record.' + name for name in NAMES]
TYPES = ['BIGINT', 'VARCHAR', 'BIGINT', 'DOUBLE', 'VARCHAR', 'VARCHAR', 'VARCHAR', 'VARCHAR']
MODEL = ('<model name="benchmark" package="org.intermine.model.benchmark">'
         '<class name="Record" is-interface="false">'
         + ''.join(f'<attribute name="{name}" type="{kind}"/>'
                   for name, kind in zip(NAMES[1:], [
                       'java.lang.String', 'int', 'double', 'java.lang.String',
                       'java.lang.String', 'java.lang.String', 'java.lang.String'], strict=True))
         + '</class></model>').encode()
SQL = ('SELECT "Record.chromosome", count(*)::BIGINT, sum("Record.length")::BIGINT '
       'FROM results GROUP BY 1 ORDER BY 1')
CASES = ['original_stream_csv', 'original_pandas_csv', 'modern_parquet_1', 'modern_parquet_4',
         'read_csv_pandas', 'read_csv_python', 'read_csv_duckdb', 'read_parquet_duckdb']


def encoded(row):
    return json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode() + b'\n'


def source_digest(package_path):
    digest = hashlib.sha256()
    for path in sorted(package_path.rglob('*')):
        if path.suffix in ('.py', '.toml'):
            digest.update(str(path.relative_to(package_path)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_fixture(folder, count, diverse):
    rng = random.Random(314)
    offsets = array('Q', [0])
    digest = hashlib.sha256()
    expected = {}
    path = folder / 'wire.rows'
    with path.open('wb') as output:
        for i in range(count):
            row = [i, f'{i:012d}', i % 5000, (i % 1009) / 4,
                   f'gene_{i % 10000}', None if i % 7 == 0 else f'annotation{i % 17}',
                   'Müller 🧬 ' + 'ACGT' * 12, f'chr{i % 10}']
            if diverse:
                row[1] = rng.randbytes(16).hex()
                row[4] = rng.randbytes(12).hex()
                row[6] = ''.join(rng.choices('ACGT', k=80))
            line = encoded(row)
            digest.update(line)
            output.write(line[:-1] + b',\n')
            offsets.append(output.tell())
            n, total = expected.get(row[-1], (0, 0))
            expected[row[-1]] = (n + 1, total + row[2])
    return path, offsets, digest.hexdigest(), sorted([k, *v] for k, v in expected.items())


@contextmanager
def fixture_server(path, offsets):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def respond(self):
            route = urlsplit(self.path)
            body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            params = parse_qs(route.query)
            params.update(parse_qs(body.decode()))
            if route.path.endswith('/version/ws'):
                payload = b'30'
            elif route.path.endswith('/model'):
                payload = MODEL
            elif route.path.endswith('/query/results'):
                start = int(params.get('start', ['0'])[0])
                count = int(params.get('size', [str(len(offsets) - 1)])[0])
                stop = min(start + count, len(offsets) - 1)
                requests.append(dict(start=start, size=count, format=params.get('format')))
                if params.get('format') == ['count']:
                    payload = str(len(offsets) - 1).encode()
                else:
                    prefix = b'{"results":[\n'
                    suffix = b'\n],"wasSuccessful":true}\n'
                    length = offsets[stop] - offsets[start] - 2 if stop > start else 0
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(prefix) + length + len(suffix)))
                    self.end_headers()
                    self.wfile.write(prefix)
                    with path.open('rb') as source:
                        source.seek(offsets[start])
                        while length:
                            block = source.read(min(length, 65536))
                            self.wfile.write(block)
                            length -= len(block)
                    self.wfile.write(suffix)
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = respond
        do_POST = respond

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}/service', requests
        finally:
            server.shutdown()
            thread.join()


def worker(args):
    original = args.case.startswith('original_') or args.case in ('read_csv_pandas', 'read_csv_python')
    package = 'intermine' if original else 'intermine314'
    module = __import__(package)
    if not Path(module.__file__).is_relative_to(Path(sys.prefix)):
        raise RuntimeError('Benchmark must use an installed distribution')
    if not original and importlib.util.find_spec('intermine') is not None:
        raise RuntimeError('Original InterMine leaked into candidate environment')
    if args.case == 'read_csv_pandas':
        import pandas as pd

    service = runtime = None
    if args.case.startswith(('original_', 'modern_')):
        if original:
            # Original 1.13.0 imports the alias removed in Python 3.10.
            # This harness-only shim does not modify the installed distribution.
            collections.MutableMapping = collections.abc.MutableMapping
            from intermine.webservice import Service
        else:
            from intermine314.service import Service
        service = Service(args.url)
        query = service.new_query(root='Record')
        query.add_view(*COLUMNS)
        query.add_sort_order('Record.id')
    if not original:
        import duckdb
        import polars
        import pyarrow

        from intermine314.export import ResourceRuntime, query_parquet
        from intermine314.query import ParallelOptions

        runtime = ResourceRuntime('cpu16')
        runtime.__enter__()
        # Load native engines before timing, without materializing benchmark data.
        with runtime.activate():
            with duckdb.connect(config=runtime.duckdb_config()) as con:
                con.execute('SELECT 1').fetchall()
        assert polars.thread_pool_size() <= 12
        assert pyarrow.cpu_count() <= 4

    def operation():
        if args.case == 'original_stream_csv':
            with args.target.open('w', newline='', encoding='utf8') as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerow(COLUMNS)
                writer.writerows(query.results(row='list', size=args.count))
            return args.count
        if args.case == 'original_pandas_csv':
            frame = query.dataframe(size=args.count)
            frame.to_csv(args.target, index=False, lineterminator='\n')
            return len(frame)
        if args.case.startswith('modern_parquet_'):
            workers = int(args.case.rsplit('_', 1)[1])
            with runtime.activate():
                query.to_parquet(
                    args.target, size=args.count, single_file=True, batch_size=10000,
                    parallel_options=ParallelOptions(
                        max_workers=workers, page_size=10000, inflight_limit=workers,
                        max_inflight_bytes_estimate=32 * 1024**2,
                        ordered=True, pagination='offset'),
                )
            return args.count
        if args.case == 'read_csv_pandas':
            frame = pd.read_csv(args.input, usecols=[COLUMNS[2], COLUMNS[-1]])
            grouped = frame.groupby(COLUMNS[-1])[COLUMNS[2]].agg(['size', 'sum'])
            return [[str(k), int(row['size']), int(row['sum'])] for k, row in grouped.iterrows()]
        if args.case == 'read_csv_python':
            groups = {}
            with args.input.open(newline='', encoding='utf8') as source:
                for row in csv.DictReader(source):
                    key = row[COLUMNS[-1]]
                    n, total = groups.get(key, (0, 0))
                    groups[key] = n + 1, total + int(row[COLUMNS[2]])
            return sorted([k, *v] for k, v in groups.items())
        with runtime.activate():
            if args.case == 'read_parquet_duckdb':
                return query_parquet(args.input, SQL).rows()
            with duckdb.connect(config=runtime.duckdb_config()) as con:
                con.read_csv(str(args.input), header=True, columns=dict(zip(COLUMNS, TYPES, strict=True))).create_view('results')
                return con.execute(SQL).fetchall()

    gc.collect()
    baseline_rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    start_cpu, start = time.process_time(), time.perf_counter()
    result = operation()
    wall, cpu = time.perf_counter() - start, time.process_time() - start_cpu
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if service is not None and hasattr(service, 'close'):
        service.close()
    if runtime is not None:
        runtime.__exit__(None, None, None)
    print(json.dumps(dict(seconds=wall, cpu_seconds=cpu, baseline_rss_bytes=baseline_rss,
                          peak_rss_bytes=peak_rss, result=result,
                          package_version=importlib.metadata.version(package),
                          package_file=module.__file__, package_digest=source_digest(Path(module.__file__).parent),
                          python=sys.version,
                          dependencies={dist.metadata['Name']: dist.version
                                        for dist in importlib.metadata.distributions()})))


def fingerprint(con, path):
    if path.suffix == '.parquet':
        relation = con.read_parquet(str(path))
    else:
        relation = con.read_csv(str(path), header=True, columns=dict(zip(COLUMNS, TYPES, strict=True)))
    digest, count = hashlib.sha256(), 0
    with relation.to_arrow_reader(batch_size=10000) as reader:
        for batch in reader:
            for row in zip(*(column.to_pylist() for column in batch.columns), strict=True):
                digest.update(encoded(row))
                count += 1
    return digest.hexdigest(), count


def run(args):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Output directory must be empty')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    env = dict(measured_at=datetime.now(UTC).isoformat(), source_revision=revision,
               source_digest=source_digest(ROOT / 'src/intermine314'),
               benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                              if line.startswith('model name')), process_cpus=os.process_cpu_count(),
               platform=platform.platform(), repeats=args.repeats, cpu_budget=16,
               page_size=10000, batch_size=10000, queue_bytes=32 * 1024**2,
               polars_threads=12, arrow_cpu_threads=4, arrow_io_threads=2,
               reference_adjustment='Harness aliases collections.MutableMapping to collections.abc.MutableMapping before Service import; original wheel unchanged',
               scope='Synthetic loopback, no added latency, warm OS cache; imports/setup excluded; RSS includes native/import overhead; no fsync')
    pq.write_table(pa.Table.from_pylist([env]), args.output / 'environment.parquet')
    trials, storage = [], []
    for count in args.rows:
        for fixture in args.fixtures:
            with tempfile.TemporaryDirectory(prefix='intermine-comparison-') as tmp:
                folder = Path(tmp)
                path, offsets, expected_hash, expected = prepare_fixture(folder, count, fixture == 'diverse')
                targets = {case: folder / (case + ('.parquet' if 'parquet' in case else '.csv'))
                           for case in CASES[:4]}
                with fixture_server(path, offsets) as (url, requests), duckdb.connect(config={'threads': 16}) as con:
                    rng = random.Random(314)
                    for repeat in range(args.repeats):
                        # Read tests depend on completed exports; randomize within each stage.
                        exports, reads = CASES[:4].copy(), CASES[4:].copy()
                        rng.shuffle(exports)
                        rng.shuffle(reads)
                        for case in exports + reads:
                            original = case.startswith('original_') or case in ('read_csv_pandas', 'read_csv_python')
                            python = args.reference_python if original else args.candidate_python
                            target = targets.get(case, folder / 'unused')
                            source = targets['modern_parquet_4' if case == 'read_parquet_duckdb' else 'original_stream_csv']
                            command = [str(python.absolute()), '-I', str(Path(__file__).resolve()), '--worker',
                                       '--case', case, '--url', url, '--count', str(count),
                                       '--target', str(target), '--input', str(source)]
                            before = len(requests)
                            result = subprocess.run(command, cwd=folder, check=False, capture_output=True, text=True,
                                                    env={**os.environ, 'POLARS_MAX_THREADS': '12',
                                                         'ARROW_NUM_THREADS': '4', 'OMP_NUM_THREADS': '1',
                                                         'OPENBLAS_NUM_THREADS': '1', 'NO_PROXY': '127.0.0.1'})
                            if result.returncode:
                                raise RuntimeError(f'{case} failed:\n{result.stderr}')
                            value = json.loads(result.stdout)
                            if not original:
                                assert value['package_digest'] == env['source_digest'], 'Installed candidate differs from source'
                            if case in exports:
                                assert value['result'] == count
                                assert fingerprint(con, target) == (expected_hash, count), case
                            else:
                                assert value['result'] == expected, (case, value['result'], expected)
                            value['result'] = json.dumps(value['result'])
                            value['dependencies'] = json.dumps(value['dependencies'], sort_keys=True)
                            trials.append(dict(fixture=fixture, rows=count, case=case, repeat=repeat,
                                               requests=len(requests) - before,
                                               output_bytes=target.stat().st_size if target.exists() else None,
                                               full_value_sha256=expected_hash, **value))
                            print(f'{fixture} {count} {repeat + 1}/{args.repeats} {case}: {value["seconds"]:.3f}s', flush=True)
                            pq.write_table(pa.Table.from_pylist(trials), args.output / 'trials.parquet')
                    # gzip is a storage control, not an original-client performance claim.
                    compressed = folder / 'stream.csv.gz'
                    with targets['original_stream_csv'].open('rb') as source, compressed.open('wb') as raw:
                        with gzip.GzipFile(fileobj=raw, filename='', mode='wb', compresslevel=6, mtime=0) as output:
                            shutil.copyfileobj(source, output)
                    assert fingerprint(con, compressed) == (expected_hash, count)
                    for name, target in {**targets, 'original_csv_gzip6': compressed}.items():
                        storage.append(dict(fixture=fixture, rows=count, format=name, bytes=target.stat().st_size,
                                            full_value_sha256=expected_hash, verified_rows=count))
                    pq.write_table(pa.Table.from_pylist(storage), args.output / 'storage.parquet')
    with duckdb.connect() as con:
        summary = con.execute(
            'SELECT fixture,rows,"case",median(seconds) AS "seconds",min(seconds) min_seconds,'
            'max(seconds) max_seconds, median(peak_rss_bytes) peak_rss_bytes,'
            'median(baseline_rss_bytes) baseline_rss_bytes, median(requests) requests,'
            'median(output_bytes) output_bytes,count(*) repeats '
            'FROM read_parquet(?) GROUP BY ALL ORDER BY fixture,rows,"case"',
            [str(args.output / 'trials.parquet')]).to_arrow_table()
        pq.write_table(summary, args.output / 'summary.parquet')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--url')
    parser.add_argument('--count', type=int)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--reference-python', type=Path)
    parser.add_argument('--candidate-python', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rows', nargs='+', type=int, default=[100000, 1000000])
    parser.add_argument('--fixtures', nargs='+', choices=['repetitive', 'diverse'], default=['repetitive', 'diverse'])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    worker(args) if args.worker else run(args)


if __name__ == '__main__':
    main()
