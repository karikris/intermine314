"""Managed model and lossless wire-to-analytics integration contracts."""
from datetime import date
from decimal import Decimal

import polars as pl
import pytest

from intermine314.export import query_parquet
from intermine314.model import Model, ModelError
from intermine314.query import Query
from intermine314.query.builder import ConstraintError, QueryError
from tests.fixtures.compatibility import FixtureSession

TYPED_MODEL = b'''<model name="typed" package="org.example">
<class name="Record"><attribute name="identifier" type="java.lang.String"/>
<attribute name="amount" type="java.math.BigDecimal"/>
<attribute name="longValue" type="java.lang.Long"/>
<attribute name="active" type="java.lang.Boolean"/>
<attribute name="day" type="java.util.Date"/>
<attribute name="byteValue" type="byte"/><attribute name="shortValue" type="short"/>
<attribute name="intValue" type="int"/><attribute name="floatValue" type="float"/>
<attribute name="doubleValue" type="double"/></class></model>'''
COLUMNS = ['Record.' + n for n in (
    'identifier', 'amount', 'longValue', 'active', 'day', 'byteValue',
    'shortValue', 'intValue', 'floatValue', 'doubleValue',
)]


def wire(rows, version):
    if version < 8:
        rows = ['[' + ','.join('{"value":' + v + '}' for v in row) + ']' for row in rows]
    else:
        rows = ['[' + ','.join(row) + ']' for row in rows]
    return ('{"results":[\n' + ',\n'.join(rows) + ('\n' if rows else '')
            + '],"wasSuccessful":true,"statusCode":200}\n').encode()


def typed_session(version=8, rows=None):
    session = FixtureSession.service(version=version, rows=wire(rows or [], version))
    session.routes['GET', '/service/model'] = TYPED_MODEL
    return session


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_managed_model_shared_cache_root_and_clone(profile, native_service_factory):
    service = native_service_factory(compatibility=profile)
    query = service.select('Employee.name')
    model = service.model
    assert isinstance(model, Model)
    assert query.model is model is service.model
    assert service._resolve_model_name() == 'testmodel'
    assert sum(r.path.endswith('/model') for r in service.opener._session.requests) == 1
    assert all(r.closed for r in service.opener._session.responses)
    if profile == 'legacy':
        assert query.root is model.get_class('Employee')
        assert query.rootClass is query.root
    else:
        assert query.root == 'Employee'
        assert not hasattr(query, 'rootClass')
    clone = query.clone()
    assert clone.root is query.root
    assert clone.model is model and clone.service is service
    assert clone.to_spec().root_class == 'Employee'
    assert clone.to_spec().compatibility == profile


def test_model_wildcards_prefetch_and_validation(model_xml):
    model = Model(model_xml)
    query = Query(model, root='Employee')
    assert query.root.name == 'Employee'
    query.prefetch_depth = 2
    query.prefetch_id_only = True
    query.select('*')
    assert query.views == ['Employee.age', 'Employee.end', 'Employee.fullTime',
                           'Employee.id', 'Employee.name', 'Employee.address.id',
                           'Employee.department.id', 'Employee.departmentThatRejectedMe.id',
                           'Employee.simpleObjects.name']
    assert [j.path for j in query.joins] == [
        'Employee.address', 'Employee.department', 'Employee.departmentThatRejectedMe',
        'Employee.simpleObjects']
    assert all(j.style == 'OUTER' for j in query.joins)
    with pytest.raises(ModelError):
        Query(model).select('Employee.missing')
    with pytest.raises(ConstraintError, match='attribute'):
        Query(model).select('Employee.department')
    with pytest.raises(QueryError, match='reference'):
        Query(model, root='Employee').add_join('name')
    native = Query(model, root='Employee', compatibility='native', validate=False)
    assert native.root == 'Employee'
    native.select('doesNotExist')
    assert native.views == ['Employee.doesNotExist']


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('version', [7, 8])
@pytest.mark.parametrize('single', [False, True])
def test_wire_exact_schema_across_null_first_and_scale_changes(
    tmp_path, native_service_factory, profile, version, single,
):
    rows = [['null'] * len(COLUMNS),
            ['"0007"', '9007199254740993.12', '9007199254740993', 'true', '"2026-01-02"',
             '-128', '32767', '2147483647', '1.5', '2.25'],
            ['"0002"', '-0.00000001', '-9007199254740993', 'false', '"2026-02-03"',
             '127', '-32768', '-2147483648', '0.5', '0.25']]
    session = typed_session(version, rows)
    service = native_service_factory(session=session, compatibility=profile)
    query = service.select(*COLUMNS)
    ordinary = list(query.rows())
    assert isinstance(ordinary[1]['Record.amount'], float)
    target = tmp_path / ('rows.parquet' if single else 'parts')
    query.to_parquet(target, batch_size=1, single_file=single, size=3)
    result = query_parquet(target)
    assert result.schema == dict(zip(COLUMNS, [pl.String, pl.Decimal(38, 8), pl.Int64,
        pl.Boolean, pl.Date, pl.Int8, pl.Int16, pl.Int32, pl.Float32, pl.Float64]))
    assert result.rows() == [(None,) * len(COLUMNS),
        ('0007', Decimal('9007199254740993.12'), 9007199254740993, True, date(2026, 1, 2),
         -128, 32767, 2147483647, 1.5, 2.25),
        ('0002', Decimal('-0.00000001'), -9007199254740993, False, date(2026, 2, 3),
         127, -32768, -2147483648, 0.5, 0.25)]
    assert query.dataframe(size=3).rows() == result.rows()
    assert isinstance(list(query.rows())[1]['Record.amount'], float)
    assert all(response.closed for response in session.responses)
    assert session.close_calls == 0


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_empty_model_schema_and_dataframe(tmp_path, native_service_factory, profile):
    query = native_service_factory(session=typed_session(), compatibility=profile).select(*COLUMNS)
    result = query.dataframe(size=0, parquet_path=tmp_path / 'empty.parquet')
    assert result.height == 0
    assert result.schema['Record.amount'] == pl.Decimal(38, 0)
    assert result.schema['Record.longValue'] == pl.Int64
    assert result.schema['Record.day'] == pl.Date


class PagedSession(FixtureSession):
    """Real managed HTTP requests with independent responses for each page."""

    def __init__(self, rows, version=8, model=TYPED_MODEL):
        super().__init__(typed_session(version).routes)
        self.routes['GET', '/service/model'] = model
        self.rows = rows
        self.version = version

    def _capture(self, method, url, data=None, headers=None, **options):
        from urllib.parse import parse_qs

        result = super()._capture(method, url, data, headers, **options)
        if self.requests[-1].path != '/service/query/results':
            return result
        params = parse_qs(data.decode() if isinstance(data, bytes) else data)
        if params.get('format') == ['count']:
            return str(len(self.rows)).encode()
        start = int(params['start'][0])
        size = int(params.get('size', [len(self.rows)])[0])
        return wire(self.rows[start:start + size], self.version)


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('version', [7, 8])
def test_parallel_decimal_precision_isolation_and_closure(
    tmp_path, native_service_factory, profile, version,
):
    from intermine314.query import ParallelOptions

    rows = [['9007199254740993.12'], ['0.000000001'], ['null'], ['-123.00001']]
    session = PagedSession(rows, version)
    query = native_service_factory(session=session, compatibility=profile).select('Record.amount')
    spec = query.to_spec()
    path = query.to_parquet(tmp_path / 'parallel', batch_size=1,
        parallel_options=ParallelOptions(page_size=1, max_workers=2, inflight_limit=2))
    assert query_parquet(path).to_series().to_list() == [
        Decimal('9007199254740993.12'), Decimal('0.000000001'), None, Decimal('-123.00001')]
    assert query.to_spec() == spec
    assert spec.decimal_paths == ()
    assert isinstance(next(iter(query.rows()))['Record.amount'], float)
    assert all(response.closed for response in session.responses)
    assert session.close_calls == 0


@pytest.mark.parametrize('single', [True, False])
@pytest.mark.parametrize('bad', ['1e38', '0.000000000000000000000000000000000000001',
                                '99999999999999999999999999999999999999'])
def test_decimal_failure_preserves_previous_output_and_closes_responses(
    tmp_path, native_service_factory, single, bad,
):
    # The third value either exceeds 38 digits or makes prior high-scale parts
    # incompatible with the final precision. Publication must remain untouched.
    session = PagedSession([['1.25'], ['0.0001'], [bad]])
    query = native_service_factory(session=session).select('Record.amount')
    target = tmp_path / ('previous.parquet' if single else 'previous')
    old = target if single else target / 'part-00000.parquet'
    if not single:
        target.mkdir()
    pl.DataFrame({'previous': ['kept']}).write_parquet(old)
    before = old.read_bytes()
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        query.to_parquet(target, batch_size=1, single_file=single)
    assert old.read_bytes() == before
    assert all(response.closed for response in session.responses)
    assert list(tmp_path.iterdir()) == [target]


def test_default_root_export_uses_same_selection_without_mutation(
    tmp_path, native_service_factory,
):
    simple_model = b'''<model name="default" package="org.example"><class name="Record">
        <attribute name="name" type="String"/></class></model>'''
    session = PagedSession([['42', '"0007"']], model=simple_model)
    service = native_service_factory(session=session, compatibility='legacy')
    query = Query(service.model, service, root='Record')
    assert query.views == []
    result = query.dataframe(parquet_path=tmp_path / 'default.parquet')
    assert result.schema == {'Record.id': pl.Int32, 'Record.name': pl.String}
    assert result.rows() == [(42, '0007')]
    assert query.views == []
    assert query.root is service.model.get_class('Record')


@pytest.mark.parametrize('xml', [b'<model name="only"/>', b'<model broken'])
def test_native_invalid_model_fallback_and_strict_property(
    xml, native_service_factory, monkeypatch,
):
    from intermine314.model import ModelParseError

    session = FixtureSession.service()
    session.routes['GET', '/service/model'] = xml
    original = Model.parse_model

    def parse_closed(self, source):
        assert all(response.closed for response in session.responses)
        return original(self, source)

    monkeypatch.setattr(Model, 'parse_model', parse_closed)
    service = native_service_factory(session=session)
    query = service.select('Missing.unknown')
    assert query.root == 'Missing'
    assert not isinstance(query.model, Model)
    with pytest.raises(ModelParseError):
        service.model
    service.select('Missing.other')
    assert sum(r.path.endswith('/model') for r in session.requests) == 1
    assert all(r.closed for r in session.responses)


def test_native_unavailable_model_only_requests_once(native_service_factory):
    session = FixtureSession.service()
    session.routes['GET', '/service/model'] = RuntimeError('offline')
    service = native_service_factory(session=session)
    assert service.select('Anything.name').model is None
    assert sum(r.path.endswith('/model') for r in session.requests) == 1


def test_strict_constraints_subclasses_and_nested_schema(model_xml):
    model = Model(model_xml)
    query = Query(model, root='Employee')
    query.add_constraint(path='department.manager', subclass='CEO')
    query.select('department.manager.salary', 'name', 'age')
    assert query._parquet_schema() == {
        'Employee.department.manager.salary': pl.Int32,
        'Employee.name': pl.String, 'Employee.age': pl.Int32,
    }
    query.select('department.manager.*')
    assert 'Employee.department.manager.salary' in query.views
    with pytest.raises(ModelError):
        Query(model, root='Employee').add_constraint('missing', '=', 1)
    with pytest.raises(ConstraintError, match='attribute'):
        Query(model, root='Employee').add_constraint('department', '=', 1)
    with pytest.raises(ConstraintError, match='subclass'):
        Query(model, root='Employee').add_constraint(path='department', subclass='Address')


def test_unknown_types_and_invalid_values_are_explicit(tmp_path, native_service_factory):
    session = PagedSession([['"12"']])
    query = native_service_factory(session=session).select('Record.longValue')
    with pytest.raises((TypeError, ValueError, pl.exceptions.PolarsError)):
        query.to_parquet(tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
    assert all(r.closed for r in session.responses)
    query.model.get_class('Record').get_field('longValue').type_name = 'Unsupported'
    with pytest.raises(ValueError, match='Unsupported model type'):
        query._parquet_schema()


def test_parallel_pages_close_iterators_when_page_limit_stops_early():
    from intermine314.query import ParallelOptions

    closed = []

    class Rows:
        def __iter__(self):
            return self

        def __next__(self):
            return {'Record.amount': Decimal('1.25')}

        def close(self):
            closed.append(True)

    query = Query().select('Record.amount')
    query.results = lambda **kwargs: Rows()
    result = list(query.run_parallel(size=2, parallel_options=ParallelOptions(page_size=1, max_workers=2)))
    assert len(result) == 2
    assert len(closed) == 2


@pytest.mark.parametrize('field,value', [('day', '123'), ('floatValue', '1e100')])
def test_model_rejects_calendar_number_and_float_width_overflow(
    tmp_path, native_service_factory, field, value,
):
    session = PagedSession([[value]])
    query = native_service_factory(session=session).select('Record.' + field)
    with pytest.raises((TypeError, ValueError, pl.exceptions.PolarsError)):
        query.to_parquet(tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
    assert all(r.closed for r in session.responses)


def test_clone_preserves_query_prefetch_overrides(model_xml):
    query = Query(Model(model_xml), root='Employee')
    query.prefetch_depth = 2
    query.prefetch_id_only = True
    clone = query.clone()
    assert clone.prefetch_depth == 2
    assert clone.prefetch_id_only is True


def test_native_unverified_unknown_path_cannot_disable_known_decimal_schema(native_service_factory):
    query = native_service_factory(session=typed_session()).select('Record.amount', 'Record.unknown')
    # Native construction remains permissive; exports require a complete schema
    # when a full Model is present, rather than silently decoding known decimals as floats.
    assert query.views == ['Record.amount', 'Record.unknown']
    with pytest.raises(ModelError, match='unknown'):
        query._parquet_schema()


def test_exact_decoder_does_not_hide_float_overflow(native_service_factory):
    from dataclasses import replace

    session = PagedSession([['1.25', '1e400']])
    service = native_service_factory(session=session)
    query = service.select('Record.amount', 'Record.doubleValue')
    spec = replace(query.to_spec(), decimal_paths=('Record.amount',))
    with pytest.raises(ValueError, match='overflows'):
        list(service.execute(spec).results())
    assert all(response.closed for response in session.responses)


@pytest.mark.parametrize('version', [7, 8])
@pytest.mark.parametrize('single', [True, False])
@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('large,fraction', [
    ('12345678901234567890123456789012345678', '0.001'),
    ('-1234567890123456789012345678901234567', '0.01'),
    ('1', '0.00000000000000000000000000000000000001'),
])
def test_same_batch_decimal_scale_overflow_rolls_back_without_nulling(
    tmp_path, native_service_factory, version, single, profile, large, fraction,
):
    session = PagedSession([['null'], [large], [fraction]], version)
    query = native_service_factory(session=session, compatibility=profile).select('Record.amount')
    target = tmp_path / ('previous.parquet' if single else 'previous')
    old = target if single else target / 'part-00000.parquet'
    if not single:
        target.mkdir()
    pl.DataFrame({'previous': ['kept']}).write_parquet(old)
    before = old.read_bytes()
    with pytest.raises(ValueError, match='precision 38'):
        query.to_parquet(target, batch_size=3, single_file=single)
    assert old.read_bytes() == before
    assert all(response.closed for response in session.responses)
    assert session.close_calls == 0
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('version', [7, 8])
@pytest.mark.parametrize('single', [True, False])
def test_same_batch_decimal_boundary_values_are_exact(
    tmp_path, native_service_factory, version, single,
):
    values = ['null', '12345678901234567890123456789012345.123', '-0.001', '0.00']
    session = PagedSession([[value] for value in values], version)
    query = native_service_factory(session=session).select('Record.amount')
    target = tmp_path / ('boundary.parquet' if single else 'boundary')
    query.to_parquet(target, batch_size=4, single_file=single)
    frame = query_parquet(target)
    assert frame.schema == {'Record.amount': pl.Decimal(38, 3)}
    assert frame.to_series().to_list() == [None, *(Decimal(value) for value in values[1:])]
    assert all(response.closed for response in session.responses)
