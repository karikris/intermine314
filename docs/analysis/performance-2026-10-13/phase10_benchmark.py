"""Correctness-checked final stage sweeps; seven trials per configuration.

Run with the repository venv. Fresh case processes configure pools before imports.
The baseline source is extracted from Git, never changed. Only loopback traffic.
Results are Parquet, with DuckDB summaries. No portable performance assertions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from run_benchmarks import COLUMNS, cpu_checksum, loopback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE = '6a85ed73c6e14a8c54b31a6282df39b36aef9dee'
MEMORY = 4 * 1024**3
SQL_ROWS = 4_000_000
CPU_ITERATIONS = 4_000_000


def transform(source, destination):
    """Importable, path-only pure Python coarse callback, native threads one."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = pq.read_table(source)
    value = cpu_checksum(CPU_ITERATIONS)
    pq.write_table(data.append_column('checksum', pa.array([value] * data.num_rows)), destination)


def tree_rss(pid):
    # Sum of resident sets includes shared pages more than once; not unique PSS.
    try:
        status = Path(f'/proc/{pid}/status').read_text()
        rss = next(int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith('VmRSS:'))
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        return rss + sum(tree_rss(int(child)) for child in children)
    except (OSError, StopIteration):
        return 0


def case(spec, scratch, repeats):
    import duckdb
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    workers, stage = spec['workers'], spec['stage']
    current = spec['implementation'] == 'current'
    if current:
        from intermine314.export import ResourceProfile, ResourceRuntime

        runtime = ResourceRuntime(ResourceProfile(
            cpu_budget=max(16, workers), max_workers=workers, per_host_limit=workers,
            duckdb_threads=workers if stage in {'sql', 'ordered_copy', 'unordered_copy'} else 4,
            polars_threads=workers if stage == 'polars' else 4,
            arrow_cpu_threads=4, arrow_io_threads=2, memory_budget_bytes=MEMORY,
        ))
    records = []

    def run(action, check, rows):
        check(action())
        for repeat in range(repeats):
            peak = [tree_rss(os.getpid())]
            initial = peak[0]
            stop = threading.Event()

            def sample():
                while not stop.wait(.01):
                    peak[0] = max(peak[0], tree_rss(os.getpid()))

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu = time.process_time()
            start = time.perf_counter()
            try:
                result = action()
                elapsed = time.perf_counter() - start
                parent_cpu = time.process_time() - cpu
            finally:
                peak[0] = max(peak[0], tree_rss(os.getpid()))
                stop.set()
                sampler.join()
            child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            checksum, output_bytes = check(result)
            child_cpu = (child_after.ru_utime + child_after.ru_stime -
                         child_before.ru_utime - child_before.ru_stime)
            records.append(dict(spec, repeat=repeat, rows=rows, wall_s=elapsed,
                                parent_cpu_s=parent_cpu, child_cpu_s=child_cpu,
                                initial_tree_rss_bytes=initial, peak_tree_rss_bytes=peak[0],
                                output_bytes=output_bytes, result_sha256=checksum,
                                correctness='passed', memory_budget_bytes=MEMORY,
                                cpu_visible=os.process_cpu_count(), python=sys.version.split()[0],
                                source_revision=BASELINE if not current else subprocess.check_output(
                                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                memory_scope='10ms process-tree RSS sum; shared pages counted per process',
                                measured_at=time.time()))

    def frame_check(frame, expected):
        assert frame.columns == expected.columns
        assert frame.rows() == expected.rows(), 'Full output values differ'
        # Decimal/engine widths are checked separately by compatibility tests.
        return hashlib.sha256(repr(frame.rows()).encode()).hexdigest()

    output = scratch / 'output.parquet'
    if stage == 'http_pipeline':
        from intermine314.query import ParallelOptions
        from intermine314.service import Service

        with loopback(spec['delay_ms']) as (url, fixture, _):
            with Service(url, request_timeout=10) as service:
                query = service.select(*COLUMNS)
                _ = service.model, service.version
                options = ParallelOptions(max_workers=workers, page_size=1000, ordered=True,
                                          prefetch=workers, inflight_limit=workers,
                                          max_inflight_bytes_estimate=64 * 1024**2)
                expected = pl.from_dicts(fixture)

                def action():
                    query.to_parquet(output, size=len(fixture), batch_size=1000, single_file=True,
                                     parallel_options=options, **({'runtime': runtime} if current else {}))
                    return output

                run(action, lambda path: (frame_check(pl.read_parquet(path), expected), path.stat().st_size), len(fixture))
    elif stage in {'sql', 'polars'}:
        from intermine314.export import query_parquet

        source = scratch / 'sql.parquet'
        sql = 'SELECT bucket, count(*)::BIGINT AS n, sum(metric)::BIGINT AS total FROM results GROUP BY bucket ORDER BY bucket'
        with duckdb.connect(config={'threads': 1}) as con:
            expected = pl.from_arrow(con.execute(sql.replace('results', 'read_parquet(?)'), [str(source)]).to_arrow_table())
        if stage == 'sql':
            def action():
                return query_parquet(source, sql, runtime=runtime)
        else:
            def action():
                with runtime.native('polars'):
                    return pl.scan_parquet(source).group_by('bucket').agg(
                        pl.len().cast(pl.Int64).alias('n'), pl.col('metric').sum().alias('total')
                    ).sort('bucket').collect(engine='streaming')
        run(action, lambda frame: (frame_check(frame, expected), frame.estimated_size()), SQL_ROWS)
    elif stage in {'ordered_copy', 'unordered_copy'}:
        from intermine314.export import ParquetOptions
        from intermine314.export.parquet import write_parquet_batches

        fixture = pl.read_parquet(scratch / 'sql.parquet').head(1_000_000)
        batches = list(fixture.iter_slices(65536))
        ordered = stage == 'ordered_copy'

        def action():
            with runtime.activate():
                write_parquet_batches(batches=batches, target=output, columns=fixture.columns,
                                      polars_module=pl, compression='zstd', single_file=True,
                                      schema=dict(fixture.schema), parquet_options=ParquetOptions(preserve_order=ordered))
            return output

        def check(path):
            frame = pl.read_parquet(path)
            if not ordered:
                frame = frame.sort('id')
            return frame_check(frame, fixture), path.stat().st_size

        run(action, check, fixture.height)
    elif stage == 'spawn_transform':
        from intermine314.export import transform_parquet

        source = scratch / 'transform.parquet'
        pq.write_table(pa.table({'id': range(1024)}), source)
        checksum = cpu_checksum(CPU_ITERATIONS)
        expected = pl.DataFrame({'id': list(range(1024)) * 16, 'checksum': [checksum] * (1024 * 16)})

        def action():
            return transform_parquet([source] * 16, scratch / 'transformed',
                                     'phase10_benchmark:transform', max_workers=workers,
                                     inflight_limit=workers, runtime=runtime)

        def check(path):
            parts = sorted(Path(path).glob('*.parquet'))
            frame = pl.concat([pl.read_parquet(part) for part in parts])
            return frame_check(frame, expected), sum(part.stat().st_size for part in parts)

        run(action, check, 16 * CPU_ITERATIONS)
    if current:
        runtime.close()
    return records


def summarize():
    import duckdb

    with duckdb.connect() as con:
        con.read_parquet(str(HERE / 'phase10_trials.parquet')).create_view('trials')
        summary = con.sql('''SELECT stage, delay_ms, implementation, workers, count(*) AS repetitions,
            median(wall_s) AS median_s, min(wall_s) AS min_s, max(wall_s) AS max_s,
            stddev_samp(wall_s)/avg(wall_s) AS cv,
            median(rows/wall_s) AS units_per_second,
            median((parent_cpu_s+child_cpu_s)/wall_s) AS cpu_cores,
            max(peak_tree_rss_bytes) AS peak_tree_rss_bytes,
            max(memory_budget_bytes) AS memory_budget_bytes,
            count(DISTINCT result_sha256)=1 AS stable_values
            FROM trials GROUP BY ALL''').to_arrow_table()
        con.register('summary', summary)
        con.execute('COPY summary TO ? (FORMAT PARQUET, COMPRESSION ZSTD)', [str(HERE / 'phase10_summary.parquet')])
        con.execute('''COPY (WITH eligible AS (SELECT *, min(median_s) OVER(PARTITION BY stage,delay_ms) AS best_s
            FROM summary WHERE implementation='current' AND peak_tree_rss_bytes<=memory_budget_bytes),
            chosen AS (SELECT *, row_number() OVER(PARTITION BY stage,delay_ms ORDER BY workers) AS choice
            FROM eligible WHERE median_s<=best_s*1.05)
            SELECT *, 'Smallest within 5% of best median, process-tree RSS sum <=4GiB; seven trials, no CI/p95 claim' AS rule
            FROM chosen WHERE choice=1) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)''', [str(HERE / 'phase10_worker_choices.parquet')])
        print(con.sql('SELECT stage,delay_ms,workers,round(median_s,4),round(peak_tree_rss_bytes/1048576,1) FROM summary ORDER BY stage,delay_ms,workers').fetchall())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case')
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--baseline-src', type=Path)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.case:
        print(json.dumps(case(json.loads(args.case), args.scratch, args.repeats)))
        return
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    specs = []
    for delay, counts in [(5, [1, 2, 4, 8]), (30, [1, 4, 8, 12, 16, 18])]:
        specs.extend(dict(stage='http_pipeline', delay_ms=delay, workers=w, implementation='current') for w in counts)
        specs.append(dict(stage='http_pipeline', delay_ms=delay, workers=4, implementation='baseline'))
    for stage, counts in [('sql', [4, 8, 12, 16, 18]), ('polars', [4, 8, 12, 16, 18]),
                          ('ordered_copy', [1, 2, 4, 8]), ('unordered_copy', [4, 8, 12, 16, 18]),
                          ('spawn_transform', [1, 2, 4, 8, 12, 16])]:
        specs.extend(dict(stage=stage, delay_ms=0, workers=w, implementation='current') for w in counts)
    random.Random(314265).shuffle(specs)
    if args.baseline_src is None:
        parser.error('--baseline-src must point to src extracted from baseline Git revision')
    records = pq.read_table(HERE / 'phase10_trials.parquet').to_pylist() if args.resume else []
    with TemporaryDirectory(prefix='intermine314-final-bench-') as folder:
        scratch = Path(folder)
        with duckdb.connect(config={'threads': 4}) as con:
            con.execute(f'''COPY (SELECT i::BIGINT id,(i%1000)::INTEGER bucket,
                ((i*7919)%100003)::BIGINT metric FROM range({SQL_ROWS}) t(i)) TO ?
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)''', [str(scratch / 'sql.parquet')])
        for spec in specs:
            if sum(all(row[key] == value for key, value in spec.items()) for row in records) == args.repeats:
                continue
            env = dict(os.environ, POLARS_MAX_THREADS=str(spec['workers'] if spec['stage'] == 'polars' else 4),
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONPATH=str(args.baseline_src if spec['implementation'] == 'baseline' else ROOT / 'src'))
            result = subprocess.run([sys.executable, str(__file__), '--case', json.dumps(spec),
                                     '--scratch', folder, '--repeats', str(args.repeats)],
                                    env=env, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f'{spec}: {result.stderr}')
            records.extend(json.loads(result.stdout))
            pq.write_table(pa.Table.from_pylist(records), HERE / 'phase10_trials.parquet', compression='zstd')
            print(f'{len(records)} trials: {spec}', flush=True)
    summarize()


if __name__ == '__main__':
    main()
