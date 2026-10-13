import pytest

from intermine314.query import ParallelOptions
from intermine314.query.inflight import BoundedInflightQueue
from intermine314.query.parallel_offset import PageBatch, run_parallel_pages
from tests.fixtures.compatibility import FixtureSession


def test_page_views_keep_rows_and_reservations_until_consumed(monkeypatch):
    from intermine314.query import parallel_offset

    queue = BoundedInflightQueue(inflight_limit=1, max_inflight_bytes_estimate=4096)
    monkeypatch.setattr(parallel_offset, 'BoundedInflightQueue', lambda **kwargs: queue)

    class Pages:
        def results(self, start=0, size=0, **kwargs):
            return iter([{'value': i} for i in range(start, start + size)])

    pages = run_parallel_pages(Pages(), size=4, page_size=4, max_workers=1,
                               inflight_limit=1, thread_name_prefix='page-views')
    page = next(pages)
    batches = list(page.batches(3))
    assert [len(batch) for batch in batches] == [3, 1]
    assert all(batch.page.rows is page.rows for batch in batches)
    assert [row['value'] for batch in batches for row in batch] == [0, 1, 2, 3]
    assert queue.stats_fields()['emitting_pages'] == 1
    pages.close()
    assert queue.stats_fields()['outstanding_pages'] == 0


def test_builtin_export_consumes_pages_without_public_row_reassembly(native_service_factory, tmp_path, monkeypatch):
    from intermine314.query import builder

    wire = b'{"results":[\n["Ada"],\n["Bob"]\n],"wasSuccessful":true}\n'
    session = FixtureSession.service(rows=wire)
    query = native_service_factory(session=session).select('Employee.name')
    captured = []
    writer = builder.write_parquet_batches

    def write(**kwargs):
        original = kwargs['batches']

        def observe():
            try:
                for batch in original:
                    assert isinstance(batch, PageBatch)
                    captured.append((batch.page.index, len(batch)))
                    yield batch
            finally:
                original.close()

        kwargs['batches'] = observe()
        return writer(**kwargs)

    monkeypatch.setattr(builder, 'write_parquet_batches', write)
    query.to_parquet(tmp_path / 'result', size=4, batch_size=1,
                     parallel_options=ParallelOptions(page_size=2, max_workers=2, ordered=True))
    assert captured == [(0, 1), (0, 1), (1, 1), (1, 1)]


@pytest.mark.parametrize('footer', [b'false', b'true,broken'])
def test_failed_page_never_yields_a_batch(native_service_factory, footer):
    session = FixtureSession.service(rows=b'{"results":[\n["Ada"]\n],"wasSuccessful":' + footer + b'}\n')
    query = native_service_factory(session=session).select('Employee.name')
    batches = query._export_batches(start=0, size=1, batch_size=1,
                                    parallel_options=ParallelOptions(page_size=1, max_workers=1))
    with pytest.raises(RuntimeError, match='fetch failed'):
        next(batches)


@pytest.mark.parametrize('fail_write', [False, True])
def test_export_accounts_native_buffers_and_releases_on_failure(native_service_factory, tmp_path, monkeypatch, fail_write):
    import polars as pl

    from intermine314.query import parallel_offset

    captured = []

    def queue(**kwargs):
        item = BoundedInflightQueue(**kwargs)
        captured.append(item)
        return item

    monkeypatch.setattr(parallel_offset, 'BoundedInflightQueue', queue)
    session = FixtureSession.service(rows=b'{"results":[\n["Ada"]\n],"wasSuccessful":true}\n')
    query = native_service_factory(session=session).select('Employee.name')
    if fail_write:
        def fail(*args, **kwargs):
            raise OSError('write failed')

        monkeypatch.setattr(pl.DataFrame, 'write_parquet', fail)
        with pytest.raises(OSError, match='write failed'):
            query.to_parquet(tmp_path / 'result', size=1, parallel_options=ParallelOptions(page_size=1))
    else:
        query.to_parquet(tmp_path / 'result', size=1, parallel_options=ParallelOptions(page_size=1))
    stats = captured[0].stats_fields()
    assert stats['peak_native_batch_bytes'] > 0
    assert stats['native_batch_bytes'] == stats['estimated_inflight_bytes'] == stats['outstanding_pages'] == 0
