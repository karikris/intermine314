"""Reproducible, offline stage benchmarks; measurements are canonical Parquet.

Run with the repository venv. Every case runs in a fresh process, with native
pools configured before import. No public InterMine server is load tested.
Synthetic fixtures deliberately do not certify all InterMine data semantics.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import gc
import gzip
import json
import multiprocessing
import os
import random
import resource
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COLUMNS = ["Record.id", "Record.identifier", "Record.length", "Record.score",
           "Record.name", "Record.optional", "Record.description", "Record.chromosome"]
ROWS = 50_000
SQL_ROWS = 4_000_000
WORKERS = [1, 2, 4, 8, 12, 16, 18, 20]


def cpu_checksum(n):
    """Pure Python task, intentionally no extension imports in subinterpreters."""
    total = 0
    for i in range(n):
        total += ((i * 17) % 10007) ^ (i & 255)
    return total


def decode_job(payload):
    from intermine314.util.json import json_loads

    return sum(json_loads(row)[0] for row in payload)


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    return 0


def io_bytes():
    data = {}
    for line in Path('/proc/self/io').read_text().splitlines():
        key, value = line.split(':')
        data[key] = int(value)
    return data


def measure(action, check, *, spec, repeat, rows, input_bytes=0):
    gc.collect()
    initial_rss = rss_bytes()
    peak = [initial_rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.005):
            peak[0] = max(peak[0], rss_bytes())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    io_before = io_bytes()
    cpu_before = time.process_time()
    started = time.perf_counter()
    value = action()
    wall = time.perf_counter() - started
    cpu = time.process_time() - cpu_before
    io_after = io_bytes()
    peak[0] = max(peak[0], rss_bytes())
    stop.set()
    sampler.join()
    checksum, output_bytes = check(value)
    record = dict(spec, repeat=repeat, rows=rows, input_bytes=input_bytes,
                  wall_s=wall, parent_cpu_s=cpu, parent_cpu_cores=cpu / wall,
                  rows_per_s=rows / wall, initial_rss_bytes=initial_rss,
                  sampled_peak_rss_bytes=peak[0], sampled_rss_delta_bytes=peak[0] - initial_rss,
                  lifetime_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  write_syscall_bytes=io_after['wchar'] - io_before['wchar'],
                  physical_write_bytes=io_after['write_bytes'] - io_before['write_bytes'],
                  output_bytes=output_bytes, checksum=str(checksum), correctness='passed',
                  pid=os.getpid(), cpu_count=os.cpu_count(), process_cpu_count=os.process_cpu_count(),
                  gil_enabled=sys._is_gil_enabled(), measured_at=time.time())
    return record


def rows_fixture(n=ROWS):
    return [dict(zip(COLUMNS, [i, f'{i:012d}', i % 5000, (i % 1009) / 11,
                             f'gene_{i % 10000}', None if i % 7 == 0 else i % 31,
                             'Müller 🧬 ' + ('ACGT' * 12), f'chr{i % 10}'])) for i in range(n)]


@contextmanager
def loopback(delay_ms, *, pages=32, page_size=1000):
    """Real Requests/urllib3/InterMine path; fixed latency, gzip, no WAN."""
    import orjson

    data = rows_fixture(pages * page_size)
    bodies = {}
    for start in range(0, len(data), page_size):
        rows = data[start:start + page_size]
        wire = b'{"results":[\n' + b',\n'.join(orjson.dumps(list(row.values())) for row in rows)
        wire += b'\n],"wasSuccessful":true,"statusCode":200}\n'
        bodies[start] = gzip.compress(wire)
    attrs = zip([c.split('.')[1] for c in COLUMNS],
                ['java.lang.Long', 'java.lang.String', 'java.lang.Integer', 'java.lang.Double',
                 'java.lang.String', 'java.lang.Integer', 'java.lang.String', 'java.lang.String'])
    model = ('<model name="records" package="org.example"><class name="Record">' +
             ''.join(f'<attribute name="{name}" type="{dtype}"/>' for name, dtype in attrs if name != 'id') +
             '</class></model>').encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            # Remove loopback Nagle/delayed-ACK artifacts from the configured
            # latency experiment; this is a fixture choice, not a client edit.
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, *args):
            pass

        def do_GET(self):
            body = b'30' if self.path.endswith('/version/ws') else model
            self.respond(body)

        def do_POST(self):
            params = parse_qs(self.rfile.read(int(self.headers['Content-Length'])).decode())
            time.sleep(delay_ms / 1000)
            self.respond(bodies[int(params['start'][0])], compressed=True)

        def respond(self, body, compressed=False):
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            if compressed:
                self.send_header('Content-Encoding', 'gzip')
            self.end_headers()
            self.wfile.write(body)

    class Server(ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = True

    server = Server(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/service', data, sum(map(len, bodies.values()))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def execute_case(spec, scratch, repeats):
    import duckdb
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    pa.set_cpu_count(spec.get('native_threads', 1))
    pa.set_io_thread_count(2)
    assert pl.thread_pool_size() == spec.get('native_threads', 1)
    results = []
    kind, variant, workers = spec['stage'], spec['variant'], spec['workers']

    def run(action, check, n, input_bytes=0):
        action() if spec.get('warmup', True) else None
        for repeat in range(repeats):
            results.append(measure(action, check, spec=spec, repeat=repeat, rows=n, input_bytes=input_bytes))

    if kind == 'http_fetch':
        from intermine314.query import ParallelOptions
        from intermine314.service import Service

        with loopback(spec['delay_ms']) as (url, fixture, wire_bytes):
            with Service(url, request_timeout=10) as service:
                query = service.select(*COLUMNS)
                _ = service.model, service.version
                options = ParallelOptions(max_workers=workers, page_size=1000, ordered=False,
                                          prefetch=workers, inflight_limit=workers,
                                          max_inflight_bytes_estimate=64 * 1024 * 1024)

                def action():
                    return sum(row[COLUMNS[0]] for row in query.run_parallel(size=len(fixture), parallel_options=options))

                def check(value):
                    assert value == len(fixture) * (len(fixture) - 1) // 2
                    return value, 0

                run(action, check, len(fixture), wire_bytes)
    elif kind == 'conversion':
        from intermine314.export.model_schema import ModelSchema, model_batch_frame
        from intermine314.export.parquet import _batch_frame

        fixture = rows_fixture()
        types = [pl.Int64, pl.String, pl.Int32, pl.Float64, pl.String, pl.Int32, pl.String, pl.String]
        schema = ModelSchema(dict(zip(COLUMNS, types)), ())
        arrow_schema = pa.schema(zip(COLUMNS, [pa.int64(), pa.large_string(), pa.int32(), pa.float64(),
                                              pa.large_string(), pa.int32(), pa.large_string(), pa.large_string()]))
        actions = {
            'package_inferred': lambda: _batch_frame(pl, fixture, COLUMNS),
            'package_model': lambda: model_batch_frame(pl, fixture, COLUMNS, schema),
            'polars_from_dicts_prototype': lambda: pl.from_dicts(fixture, schema=schema, strict=True),
            'arrow_from_pylist_prototype': lambda: pa.Table.from_pylist(fixture, schema=arrow_schema),
        }
        if variant == 'arrow_column_builders_prototype':
            values = [[row[c] for row in fixture] for c in COLUMNS]
            actions[variant] = lambda: pa.Table.from_arrays([pa.array(v, type=t) for v, t in zip(values, arrow_schema.types)], schema=arrow_schema)
        if variant == 'arrow_columns_from_rows_prototype':
            actions[variant] = lambda: pa.Table.from_arrays(
                [pa.array([row[c] for row in fixture], type=t) for c, t in zip(COLUMNS, arrow_schema.types)],
                schema=arrow_schema)

        def check(frame):
            if isinstance(frame, pa.Table):
                assert frame.num_rows == ROWS
                assert frame.slice(0, 64).to_pylist() == fixture[:64]
                assert frame.slice(ROWS - 64).to_pylist() == fixture[-64:]
                return ROWS, frame.nbytes
            assert frame.height == ROWS and frame.head(64).to_dicts() == fixture[:64]
            assert frame.tail(64).to_dicts() == fixture[-64:]
            return ROWS, frame.estimated_size()

        run(actions[variant], check, ROWS)
    elif kind == 'json_decode':
        import orjson

        payload = [orjson.dumps([i, f'{i:012d}', 'ACGT' * 16, i / 11]) for i in range(120_000)]
        chunks = [payload[i:i + 5000] for i in range(0, len(payload), 5000)]
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            action = (lambda: sum(map(decode_job, chunks))) if variant == 'serial' else (lambda: sum(pool.map(decode_job, chunks, buffersize=workers * 2)))

            def check(value):
                assert value == len(payload) * (len(payload) - 1) // 2
                return value, 0

            run(action, check, len(payload), sum(map(len, payload)))
    elif kind == 'python_cpu':
        # Native libraries imported in this coordinator do not enter the worker.
        n, jobs = 600_000, 32
        expected = cpu_checksum(n) * jobs
        cls = {'thread': futures.ThreadPoolExecutor, 'interpreter': futures.InterpreterPoolExecutor,
               'spawn_process': futures.ProcessPoolExecutor}.get(variant)
        kwargs = {'max_workers': workers}
        if variant == 'spawn_process':
            kwargs['mp_context'] = multiprocessing.get_context('spawn')
        pool = cls(**kwargs) if cls else None
        try:
            action = (lambda: sum(map(cpu_checksum, [n] * jobs))) if pool is None else (lambda: sum(pool.map(cpu_checksum, [n] * jobs, buffersize=workers * 2)))

            def check(value):
                assert value == expected
                return value, 0

            run(action, check, n * jobs)
        finally:
            if pool:
                pool.shutdown()
    elif kind in ('duckdb_query', 'duckdb_compact'):
        source = scratch / f'input-rg{spec["row_group_size"]}.parquet'
        con = duckdb.connect(config={'threads': workers, 'memory_limit': '2GB',
                                     'temp_directory': str(scratch / 'spill'),
                                     'preserve_insertion_order': spec.get('ordered', False)})
        try:
            if kind == 'duckdb_query':
                sql = ("SELECT bucket, count(*) n, sum(sqrt(metric)) s FROM read_parquet(?) "
                       "WHERE id % 3 != 0 GROUP BY bucket ORDER BY bucket")

                def action():
                    return con.execute(sql, [str(source)]).to_arrow_table()

                def check(table):
                    assert table.num_rows == 10_000
                    count = sum(table.column('n').to_pylist())
                    assert count == SQL_ROWS - (SQL_ROWS + 2) // 3
                    return count, table.nbytes

                run(action, check, SQL_ROWS, source.stat().st_size)
            else:
                target = scratch / f'compact-{os.getpid()}.parquet'
                sql = (f"COPY (SELECT * FROM read_parquet('{source}')) TO '{target}' "
                       f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {spec['output_row_group_size']})")

                def action():
                    target.unlink(missing_ok=True)
                    con.execute(sql)
                    return target

                def check(path):
                    assert pq.ParquetFile(path).metadata.num_rows == SQL_ROWS
                    return SQL_ROWS, path.stat().st_size

                run(action, check, SQL_ROWS, source.stat().st_size)
                target.unlink()
        finally:
            con.close()
    elif kind == 'polars_query':
        source = scratch / 'input-rg122880.parquet'

        def action():
            return (pl.scan_parquet(source).filter(pl.col('id') % 3 != 0)
                    .group_by('bucket').agg(pl.len().alias('n'), pl.col('metric').sqrt().sum().alias('s'))
                    .sort('bucket').collect(engine='streaming'))

        def check(frame):
            assert frame.height == 10_000 and frame['n'].sum() == SQL_ROWS - (SQL_ROWS + 2) // 3
            return frame['n'].sum(), frame.estimated_size()

        run(action, check, SQL_ROWS, source.stat().st_size)
    elif kind == 'export_writer':
        from intermine314.export.parquet import write_parquet_batches

        fixture = rows_fixture(200_000)
        schema = dict(zip(COLUMNS, [pl.Int64, pl.String, pl.Int64, pl.Float64,
                                   pl.String, pl.Int64, pl.String, pl.String]))
        target = scratch / f'export-{os.getpid()}'
        batch_size = spec['batch_size']

        def action():
            batches = (fixture[i:i + batch_size] for i in range(0, len(fixture), batch_size))
            return write_parquet_batches(batches=batches, target=target, columns=COLUMNS,
                                         polars_module=pl, compression='zstd', single_file=spec['single_file'],
                                         batch_size=batch_size, schema=schema)

        def check(path):
            files = [Path(path)] if Path(path).is_file() else list(Path(path).glob('*.parquet'))
            assert sum(pq.ParquetFile(p).metadata.num_rows for p in files) == len(fixture)
            return len(fixture), sum(p.stat().st_size for p in files)

        run(action, check, len(fixture))
    return results


def generate_inputs(scratch):
    import duckdb

    con = duckdb.connect(config={'threads': 8, 'memory_limit': '2GB'})
    try:
        for size in (10_000, 122_880, 500_000):
            target = scratch / f'input-rg{size}.parquet'
            con.execute(f"COPY (SELECT i::BIGINT AS id, (i % 10000)::INTEGER AS bucket, "
                        f"((i*7919) % 100003)::DOUBLE AS metric, 'gene_' || (i % 100000)::VARCHAR AS gene_name "
                        f"FROM range({SQL_ROWS}) t(i)) TO '{target}' "
                        f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {size})")
    finally:
        con.close()


def scenarios():
    specs = []

    def add(stage, variant, workers, **kwargs):
        specs.append(dict(stage=stage, variant=variant, workers=workers, native_threads=1, **kwargs))

    for delay in (5, 30):
        for w in WORKERS:
            add('http_fetch', 'gzip_loopback', w, delay_ms=delay)
    for v in ('package_inferred', 'package_model', 'polars_from_dicts_prototype',
              'arrow_from_pylist_prototype', 'arrow_column_builders_prototype',
              'arrow_columns_from_rows_prototype'):
        add('conversion', v, 1)
    add('json_decode', 'serial', 1)
    for w in (1, 2, 4, 8, 16):
        add('json_decode', 'thread', w)
    add('python_cpu', 'serial', 1)
    for v in ('thread', 'interpreter', 'spawn_process'):
        for w in (2, 4, 8, 16):
            add('python_cpu', v, w)
    for rg in (10_000, 122_880, 500_000):
        for w in (1, 4, 8, 12, 16, 18, 20):
            add('duckdb_query', 'parquet_aggregation', w, row_group_size=rg)
    for ordered in (True, False):
        for w in (1, 4, 8, 16):
            add('duckdb_compact', 'copy_zstd', w, row_group_size=10_000,
                output_row_group_size=122_880, ordered=ordered)
    # Match the current final-copy geometry separately from the proposed geometry.
    add('duckdb_compact', 'current_copy_geometry', 1, row_group_size=10_000,
        output_row_group_size=10_000, ordered=True)
    for w in (1, 4, 8, 12, 16, 18, 20):
        specs.append(dict(stage='polars_query', variant='streaming_aggregation', workers=w, native_threads=w))
    for batch in (10_000, 65_536):
        for single in (True, False):
            add('export_writer', 'package_writer', 1, batch_size=batch, single_file=single)
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', help='internal serialized case specification')
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--stages', help='optional comma-separated stages')
    parser.add_argument('--output', type=Path, default=HERE / 'benchmark_runs.parquet')
    args = parser.parse_args()
    if args.case:
        # JSON is only ephemeral subprocess control; all durable records are Parquet.
        print(json.dumps(execute_case(json.loads(args.case), args.scratch, args.repeats)))
        return
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    allowed = set(args.stages.split(',')) if args.stages else None
    specs = [s for s in scenarios() if allowed is None or s['stage'] in allowed]
    random.Random(265).shuffle(specs)
    records = []
    with TemporaryDirectory(prefix='intermine314-performance-', dir='/tmp') as directory:
        scratch = Path(directory)
        if any(s['stage'] in ('duckdb_query', 'duckdb_compact', 'polars_query') for s in specs):
            generate_inputs(scratch)
        for index, spec in enumerate(specs, 1):
            env = dict(os.environ, POLARS_MAX_THREADS=str(spec['native_threads']),
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONPATH=str(ROOT / 'src'))
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--case', json.dumps(spec),
                                     '--scratch', str(scratch), '--repeats', str(args.repeats)],
                                    env=env, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(f'Case failed: {spec}\n{result.stderr}')
            records.extend(json.loads(result.stdout))
            keys = sorted({key for record in records for key in record})
            table = pa.Table.from_pylist([{key: record.get(key) for key in keys} for record in records])
            pq.write_table(table, args.output, compression='zstd')
            print(f'{index}/{len(specs)} {spec["stage"]} {spec["variant"]} workers={spec["workers"]}', flush=True)
    con = duckdb.connect()
    con.read_parquet(str(args.output)).create_view('runs')
    print(con.execute('SELECT stage, count(*) trials, round(min(wall_s),4) best_s, round(max(wall_s),4) worst_s FROM runs GROUP BY stage ORDER BY stage').fetchall())
    con.close()


if __name__ == '__main__':
    main()
