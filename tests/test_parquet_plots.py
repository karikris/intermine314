from decimal import Decimal

import polars as pl
import pytest

from intermine314.bar_chart import plot_parquet
from intermine314.export import query_parquet


@pytest.fixture
def pyplot():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot

    yield pyplot
    pyplot.close('all')


def test_plot_aggregates_in_duckdb_and_only_materializes_bounded_summary(tmp_path, pyplot, monkeypatch):
    source = tmp_path / 'input.parquet'
    pl.DataFrame({'group': [str(i % 100) for i in range(100000)], 'amount': list(range(100000))}).write_parquet(source)
    summary = tmp_path / 'summary.parquet'
    original = pl.from_arrow
    observed = []
    def converted(table, *args, **kwargs):
        observed.append(table.num_rows)
        assert table.num_rows <= 5
        return original(table, *args, **kwargs)
    monkeypatch.setattr(pl, 'from_arrow', converted)
    def forbidden(*args, **kwargs):
        pytest.fail('Plot should aggregate upstream, not collect the full source')
    monkeypatch.setattr(pl, 'read_parquet', forbidden)
    ax = plot_parquet(source, 'group', 'amount', aggregation='sum', limit=5, summary_path=summary, show=False)
    assert observed and len(ax.patches) == 5
    frame = query_parquet(summary)
    assert frame['category'].to_list() == ['99', '98', '97', '96', '95']
    assert [bar.get_height() for bar in ax.patches] == [float(value) for value in frame['value']]
    assert frame['value'].to_list() == [sum(range(i, 100000, 100)) for i in (99, 98, 97, 96, 95)]
    assert pyplot.gcf().axes == [ax]


def test_plot_summary_retains_decimal_precision_and_literal_column_names(tmp_path, pyplot):
    source = tmp_path / 'literal[*].parquet'
    category, value = 'name"quoted', 'Gene.amount'
    amount = Decimal('12345678901234567890.1234')
    pl.DataFrame({category: ['a', 'a', 'b'], value: [amount, amount, None]},
                 schema={category: pl.String, value: pl.Decimal(30, 4)}).write_parquet(source)
    summary = tmp_path / 'summary.parquet'
    ax = plot_parquet(source, category, value, aggregation='sum', summary_path=summary, show=False)
    assert query_parquet(summary)['value'].to_list() == [amount * 2]
    assert len(ax.patches) == 1
    assert ax.texts[0].get_text() == str(amount * 2)


def test_count_plot_orders_ties_and_cleans_temporary_parquet(tmp_path, pyplot):
    from intermine314.export import ResourceProfile

    source = tmp_path / 'input.parquet'
    pl.DataFrame({'category': ['b', 'a', 'b', 'a', 'c']}).write_parquet(source)
    ax = plot_parquet(source, 'category', show=False, resource_profile=ResourceProfile(temp_dir=str(tmp_path)))
    assert [label.get_text() for label in ax.get_xticklabels()] == ['a', 'b', 'c']
    assert [bar.get_height() for bar in ax.patches] == [2, 2, 1]
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize('kwargs', [{'aggregation': 'unsafe SQL'}, {'aggregation': 'sum'},
                                   {'limit': 0}, {'category': ''}])
def test_invalid_plot_options_before_figure_or_output(tmp_path, pyplot, kwargs):
    options = dict(category='group', summary_path=tmp_path / 'summary.parquet', show=False)
    options.update(kwargs)
    with pytest.raises(ValueError):
        plot_parquet(tmp_path / 'absent.parquet', **options)
    assert not pyplot.get_fignums() and list(tmp_path.iterdir()) == []


def test_plot_rejects_output_collision_and_nonnumeric_aggregate(tmp_path, pyplot):
    source = tmp_path / 'input.parquet'
    pl.DataFrame({'group': ['a'], 'value': ['nonnumeric']}).write_parquet(source)
    old = source.read_bytes()
    with pytest.raises(ValueError, match='same file'):
        plot_parquet(source, 'group', summary_path=source, show=False)
    assert source.read_bytes() == old
    with pytest.raises(ValueError, match='numeric'):
        plot_parquet(source, 'group', 'value', aggregation='min', show=False)
    assert not pyplot.get_fignums()


def test_package_draw_calls_are_serialized_for_noninteractive_backend(tmp_path, pyplot, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    source = tmp_path / 'input.parquet'
    pl.DataFrame({'category': ['a', 'b']}).write_parquet(source)
    active = peak = 0
    guard = threading.Lock()
    def show():
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(.02)
        finally:
            with guard:
                active -= 1
    monkeypatch.setattr(pyplot, 'show', show)
    with ThreadPoolExecutor(2) as pool:
        axes = list(pool.map(lambda _: plot_parquet(source, 'category'), range(4)))
    assert len({id(ax) for ax in axes}) == 4 and peak == 1
