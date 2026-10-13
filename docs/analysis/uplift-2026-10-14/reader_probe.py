"""Evaluate native read_all; also expose its lost batch cancellation boundary."""
import random
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def native(reader):
    with reader._lock:
        try:
            with reader._runtime.native('duckdb_sql', threads=reader._threads):
                return reader._reader.read_all()
        finally:
            reader.close()


def main():
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    from intermine314.export import ResourceRuntime, query_parquet_reader
    from intermine314.export.query import ParquetReader

    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    rows, rng = [], random.Random(314)
    with TemporaryDirectory() as temporary, ResourceRuntime('cpu16') as runtime:
        path = Path(temporary) / 'data.parquet'
        pl.DataFrame({'id': range(100000), 'text': ['gene'] * 100000}).write_parquet(path)
        for repetition in range(21):
            variants = ['current', 'native']
            rng.shuffle(variants)
            tables = []
            for variant in variants:
                with query_parquet_reader(path, batch_size=128, runtime=runtime) as reader:
                    started = time.perf_counter_ns()
                    table = reader.read_all() if variant == 'current' else native(reader)
                    elapsed = time.perf_counter_ns() - started
                    assert reader.closed
                    tables.append(table)
                    rows.append(dict(variant=variant, repetition=repetition, ns=elapsed, rows=table.num_rows))
            assert tables[0].equals(tables[1])
    pq.write_table(pa.Table.from_pylist(rows), Path(__file__).with_name('reader_trials.parquet'), compression='zstd')

    cancellation = []
    for variant in ('current', 'native'):
        entered, release = threading.Event(), threading.Event()
        runtime = ResourceRuntime('cpu2')
        outcomes = []

        class Source:
            schema = pa.schema([('id', pa.int64())])
            calls = 0

            def __next__(self):
                if self.calls == 20:
                    raise StopIteration
                self.calls += 1
                if self.calls == 1:
                    entered.set()
                    assert release.wait(5)
                return pa.record_batch([[self.calls]], schema=self.schema)

            def read_all(self):
                batches = []
                while True:
                    try:
                        batches.append(next(self))
                    except StopIteration:
                        return pa.Table.from_batches(batches, schema=self.schema)

        source = Source()
        reader = runtime.own(ParquetReader(source, ExitStack(), runtime, 1))

        def consume():
            try:
                result = reader.read_all() if variant == 'current' else native(reader)
                outcomes.append(str(result.num_rows))
            except RuntimeError as error:
                outcomes.append(type(error).__name__)

        worker = threading.Thread(target=consume)
        worker.start()
        assert entered.wait(5)
        closer = threading.Thread(target=runtime.close)
        closer.start()
        assert runtime._closing.wait(5)
        release.set()
        worker.join(5)
        closer.join(5)
        assert not worker.is_alive() and not closer.is_alive() and reader.closed
        cancellation.append(dict(variant=variant, consumed_batches=source.calls, outcome=outcomes[0]))
    assert cancellation[0]['consumed_batches'] == 1 and cancellation[1]['consumed_batches'] == 20
    pq.write_table(pa.Table.from_pylist(cancellation), Path(__file__).with_name('reader_cancellation.parquet'), compression='zstd')
    import statistics

    print({variant: statistics.median(row['ns'] for row in rows if row['variant'] == variant) / 1e6 for variant in ('current', 'native')})
    print(cancellation)


if __name__ == '__main__':
    main()
