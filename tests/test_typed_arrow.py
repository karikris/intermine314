from datetime import date, datetime
from decimal import Decimal

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from intermine314.export.arrow import typed_batch
from intermine314.export.model_schema import ModelSchema
from intermine314.export.parquet import _fallback_batch_frame


@pytest.mark.parametrize('values', [
    [None, None], [None, '0007', 'Müller 🧬'], [b'a', None, b'\x00'],
    [True, None, False], [None, -(2**63), 2**63 - 1], [2**63, 2**64 - 1],
    [-(2**64), 2**64], [2**100, None], [1.25, float('nan'), float('inf'), None],
    [Decimal('123456789012345678901234567.123'), None, Decimal('0.0000001')],
    [date(2026, 1, 2), None], [datetime(2026, 1, 2, 3, 4, 5, 123456), None],
])
def test_flat_arrow_values_and_dtypes_match_legacy_inference(values):
    rows = [{'x': value} for value in values]
    expected = _fallback_batch_frame(pl, rows, ['x'])
    result = typed_batch(pl, rows, ['x'])
    assert result is not None
    assert_frame_equal(pl.from_arrow(result), expected)


@pytest.mark.parametrize('values', [[1.0, 2], [1, 2.0], [True, 1], [[1, 2], None], [{'a': 1}, {'a': 2}]])
def test_distinct_mixed_or_nested_semantics_use_the_existing_fallback(values):
    assert typed_batch(pl, [{'x': value} for value in values], ['x']) is None


@pytest.mark.parametrize('dtype,values', [
    (pl.Int8, [-128, None, 127]), (pl.Int16, [-32768, 32767]),
    (pl.Int32, [-(2**31), 2**31 - 1]), (pl.Int64, [-(2**63), 2**63 - 1]),
    (pl.Float32, [16777217, None, 1.25]), (pl.Float64, [2**53 + 1, 1.25]),
    (pl.Float32, [float('nan'), float('inf'), -float('inf')]),
    (pl.Boolean, [True, None]), (pl.String, ['0001', None]),
])
def test_declared_primitive_widths_match_polars(dtype, values):
    schema = ModelSchema({'x': dtype}, ())
    result = typed_batch(pl, [{'x': value} for value in values], ['x'], schema)
    expected = pl.DataFrame([pl.Series('x', values, dtype=dtype, strict=True)])
    assert_frame_equal(pl.from_arrow(result), expected)


@pytest.mark.parametrize('dtype,values', [
    (pl.Int8, [128]), (pl.Int32, [2**31]), (pl.Int64, [2**63]),
    (pl.Float32, [float(2**128 - 2**103)]), (pl.Float32, [1e100]),
])
def test_typed_overflow_cannot_silently_truncate_or_insert_null(dtype, values):
    with pytest.raises((ValueError, OverflowError)):
        typed_batch(pl, [{'x': value} for value in values], ['x'], ModelSchema({'x': dtype}, ()))


def test_integer_float_input_is_never_truncated_by_arrow_safe_flag():
    schema = ModelSchema({'x': pl.Int32}, ())
    with pytest.raises((ValueError, TypeError, pl.exceptions.PolarsError)):
        typed_batch(pl, [{'x': 1.5}], ['x'], schema)


def test_large_integer_float32_avoids_double_rounding():
    values = [2**64 + 2**40 + 1]
    schema = ModelSchema({'x': pl.Float32}, ())
    result = typed_batch(pl, [{'x': value} for value in values], ['x'], schema)
    expected = pl.DataFrame([pl.Series('x', values, dtype=pl.Float32, strict=True)])
    assert_frame_equal(pl.from_arrow(result), expected)


def test_decimal_scale_overflow_and_dates_are_validated():
    schema = ModelSchema({'x': pl.Decimal(38, 0)}, ('x',))
    with pytest.raises(ValueError, match='precision 38'):
        typed_batch(pl, [{'x': Decimal('9' * 38)}, {'x': Decimal('0.01')}], ['x'], schema)
    schema = ModelSchema({'x': pl.Date}, ())
    result = typed_batch(pl, [{'x': '2026-01-02'}, {'x': None}], ['x'], schema)
    assert result.to_pylist() == [{'x': date(2026, 1, 2)}, {'x': None}]
    with pytest.raises(ValueError, match='calendar date'):
        typed_batch(pl, [{'x': 20260102}], ['x'], schema)


def test_conversion_plan_snapshots_schema_and_refines_decimal_scale():
    from intermine314.export.arrow import _ConversionPlan

    schema = ModelSchema({'x': pl.Decimal(38, 0)}, ('x',))
    plan = _ConversionPlan(pl, ['x'], schema)
    schema['x'] = pl.String
    first = plan.convert([{'x': Decimal('1.25')}])
    second = plan.convert([{'x': Decimal('1.0001')}])
    assert first.schema.field('x').type.scale == 2
    assert second.schema.field('x').type.scale == 4
    assert second.to_pylist() == [{'x': Decimal('1.0001')}]


@pytest.mark.parametrize('values', [[1, None, 2**63 - 1], [Decimal('1.25'), None], ['0001', 'Å']])
def test_flat_column_batches_match_dictionary_conversion(values):
    from intermine314.query._columnar import ColumnBatch

    columns = ColumnBatch([[value] for value in values], ['x'])
    expected = typed_batch(pl, [{'x': value} for value in values], ['x'])
    assert typed_batch(pl, columns, ['x']).equals(expected)
    with pytest.raises(ValueError, match='selected columns'):
        typed_batch(pl, ColumnBatch([[1, 2]], ['x']), ['x'])


@pytest.mark.parametrize('version', [7, 8])
@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_builtin_exports_use_flat_batches_after_footer_validation(native_service_factory, version, profile, tmp_path):
    import json

    from intermine314.query import ParallelOptions
    from intermine314.query._columnar import ColumnBatch
    from tests.fixtures.compatibility import FixtureSession

    values = ['Ada', 41, True]
    row = values if version >= 8 else [{'value': value} for value in values]
    body = ('{"results":[\n' + json.dumps(row) + '\n],"wasSuccessful":true}\n').encode()
    session = FixtureSession.service(version=version, rows=body)
    service = native_service_factory(session=session, compatibility=profile)
    query = service.select('Employee.name', 'Employee.age', 'Employee.fullTime')
    options = ParallelOptions(page_size=1, max_workers=1)
    batches = query._export_batches(start=0, size=1, batch_size=1, parallel_options=options)
    batch = next(batches)
    assert isinstance(batch, ColumnBatch)
    assert batch.column_values() == tuple((value,) for value in values)
    assert session.responses[-1].closed
    queue = batch.source.page.queue
    assert queue.stats_fields()['emitting_pages'] == 1
    batch.observe_native(4096)
    assert queue.stats_fields()['estimated_inflight_bytes'] >= 4096
    batches.close()
    assert queue.stats_fields()['outstanding_pages'] == 0
    path = query.to_parquet(tmp_path / 'data', size=1, parallel_options=options)
    assert pl.read_parquet(str(path) + '/part-*.parquet').rows() == [tuple(values)]
