"""Profile-specific presentation over one typed Parquet/DuckDB pipeline."""

import pytest

from intermine314.util import deps


def test_optional_pandas_loader_is_lazy_and_missing_error_names_extra(monkeypatch):
    imports = []

    def missing(name):
        imports.append(name)
        raise ImportError(name)

    monkeypatch.setattr(deps, 'import_module', missing)
    deps._optional_module.cache_clear()
    try:
        with pytest.raises(ImportError, match=r'intermine314\[pandas\]'):
            deps.require_pandas('Query.dataframe()')
        assert imports == ['pandas']
    finally:
        deps._optional_module.cache_clear()


@pytest.mark.parametrize('profile,backend,expected', [
    ('native', None, 'polars'), ('legacy', None, 'pandas'),
    ('native', 'pandas', 'pandas'), ('legacy', 'polars', 'polars'),
])
@pytest.mark.parametrize('template', [False, True])
def test_profile_defaults_and_explicit_backend_share_remote_pipeline(profile, backend, expected, template, native_service_factory):
    import polars as pl

    from intermine314.query import Template
    from tests.fixtures.compatibility import FixtureSession

    pd = pytest.importorskip('pandas') if expected == 'pandas' else None
    rows = b'{"results":[\n["Ada",42],\n["Bob",null]\n],"wasSuccessful":true}\n'
    session = FixtureSession.service(rows=rows)
    session.routes['POST', '/service/template/results'] = rows
    service = native_service_factory(session=session, compatibility=profile)
    if template:
        query = Template(service.model, service=service, root='Employee', compatibility=profile)
        query.name = 'Names'
        query.add_view('Employee.name', 'Employee.age')
        query.add_constraint('name', '=', 'old')
    else:
        query = service.select('Employee.name', 'Employee.age')
    frame = query.dataframe(size=2, backend=backend, **({'A': 'new'} if template else {}))
    if expected == 'pandas':
        assert isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.RangeIndex)
        assert frame.columns.tolist() == ['Employee.name', 'Employee.age']
        assert frame['Employee.age'][0] == 42 and pd.isna(frame['Employee.age'][1])
        assert all(isinstance(dtype, pd.ArrowDtype) for dtype in frame.dtypes)
    else:
        assert isinstance(frame, pl.DataFrame)
        assert frame.to_dicts() == [{'Employee.name': 'Ada', 'Employee.age': 42}, {'Employee.name': 'Bob', 'Employee.age': None}]
    posts = [request for request in session.requests if request.method == 'POST']
    assert len(posts) == 1
    assert posts[0].path.endswith('/template/results' if template else '/query/results')
    assert all(response.closed for response in session.responses)
    if template:
        assert query.get_constraint('A').value == 'old'


@pytest.mark.parametrize('backend', ['invalid', [], 1, 'PANDAS'])
def test_invalid_backend_precedes_read_and_output(backend, tmp_path):
    from intermine314.query import Query

    target = tmp_path / 'existing.parquet'
    target.write_bytes(b'preserve')
    with pytest.raises(ValueError, match='backend'):
        Query().dataframe(backend=backend, csv_input=tmp_path / 'absent.csv', parquet_path=target)
    assert target.read_bytes() == b'preserve'


def test_missing_pandas_precedes_requests_reads_and_writes(monkeypatch, tmp_path, native_service_factory):
    from intermine314.query import Query

    monkeypatch.setattr(deps, '_optional_module', lambda name: None)
    from tests.fixtures.compatibility import FixtureSession

    session = FixtureSession.service()
    service = native_service_factory(session=session, compatibility='legacy')
    query = service.select('Employee.name')
    before = len(session.requests)
    target = tmp_path / 'existing.parquet'
    target.write_bytes(b'preserve')
    for call in (lambda: query.dataframe(size=1, parquet_path=target),
                 lambda: Query().dataframe(backend='pandas', csv_input=tmp_path / 'absent.csv', parquet_path=target)):
        with pytest.raises(ImportError, match=r'intermine314\[pandas\]'):
            call()
        assert target.read_bytes() == b'preserve'
    assert len(session.requests) == before


@pytest.mark.parametrize('empty', [False, True])
def test_pandas_arrow_types_preserve_exact_values_and_detach_from_storage(empty, tmp_path):
    from decimal import Decimal
    from io import StringIO

    import polars as pl

    from intermine314.query import Query

    pd = pytest.importorskip('pandas')
    source = 'id,amount,stamp\n' + ('' if empty else '9007199254740993,1234567890.123456789,2026-10-20T01:02:03Z\n')
    frame = Query().dataframe(backend='pandas', csv_input=StringIO(source),
                             csv_options={'schema_overrides': {'id': pl.Int64, 'amount': pl.Decimal(30, 9),
                                                               'stamp': pl.Datetime('us', 'UTC')}})
    assert frame.columns.tolist() == ['id', 'amount', 'stamp'] and len(frame) == int(not empty)
    assert all(isinstance(dtype, pd.ArrowDtype) for dtype in frame.dtypes)
    assert str(frame.dtypes['id']) == 'int64[pyarrow]'
    if not empty:
        assert frame['id'][0] == 9007199254740993
        assert frame['amount'][0] == Decimal('1234567890.123456789')
        assert frame['stamp'][0] == pd.Timestamp('2026-10-20T01:02:03Z')


def test_pandas_conversion_failure_releases_storage_and_native_admission(monkeypatch):
    from contextlib import contextmanager
    from io import StringIO

    from intermine314.export import ResourceRuntime
    from intermine314.query import Query

    pd = pytest.importorskip('pandas')
    query = Query()
    paths, stages = [], []
    write = query.to_parquet

    def tracked(path, **options):
        paths.append(path)
        return write(path, **options)

    with ResourceRuntime() as runtime:
        native = runtime.native

        @contextmanager
        def admission(stage, **options):
            stages.append(stage)
            with native(stage, **options):
                yield

        def fail(dtype):
            assert runtime.diagnostics()['active_native_reservations'] > 0
            raise RuntimeError('output conversion failed')

        monkeypatch.setattr(query, 'to_parquet', tracked)
        monkeypatch.setattr(runtime, 'native', admission)
        monkeypatch.setattr(pd, 'ArrowDtype', fail)
        with pytest.raises(RuntimeError, match='output conversion failed'):
            query.dataframe(backend='pandas', csv_input=StringIO('id\n1\n'), runtime=runtime)
        assert stages[-2:] == ['polars', 'arrow_cpu']
        assert runtime.diagnostics()['active_native_reservations'] == 0
    assert paths and not paths[0].parent.exists()
