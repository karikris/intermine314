"""Historical bar charts with lazy Polars and optional Matplotlib.

Adapted from InterMine 1.13.0 under LICENSE-BSD and NOTICE. This module owns
account configuration independently of query_manager; plots write no files.
"""

from __future__ import annotations

import json
import warnings
from contextlib import closing
from itertools import islice
from math import isfinite
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from xml.etree import ElementTree

from intermine314._helper_state import _HelperState

__all__ = ['save_mine_and_token', 'plot_go_vs_p', 'plot_go_vs_count',
           'get_query', 'query_to_barchart_log']

_state = None


def save_mine_and_token(m, t, *, registry=None, service=None, opener=None, **registry_options):
    """Validate an account, retaining configured transport and borrowed clients."""
    global mine, token, _state
    _HelperState.validate_options(registry, service, opener, registry_options)
    mine, token = m, t
    old, _state = _state, None
    if old is not None:
        old.close()
    candidate = _HelperState(m, t)
    phase = 'mine'
    try:
        candidate.connect(registry=registry, service=service, opener=opener,
                          registry_options=registry_options)
        phase = 'token'
        json.loads(candidate.read('/user/queries'))['queries'].keys()
    except BaseException as error:
        try:
            candidate.close()
        except BaseException:
            pass
        if isinstance(error, Exception):
            return 'An exception of type ' + type(error).__name__ + ' occurred. Check ' + phase
        raise
    _state = candidate
    return None


def _configured():
    if _state is None:
        raise RuntimeError('Call save_mine_and_token successfully before using plotting helpers')
    return _state


def _plot_dependencies():
    try:
        from matplotlib import pyplot
    except ImportError as error:
        raise ImportError('matplotlib is required for bar charts. Install with: '
                          'pip install "intermine314[plots]"') from error
    from intermine314.util.deps import require_polars

    return require_polars('bar charts'), pyplot


class _AccountOpener:
    """Forward calls unchanged except for the selected account URL parameter."""

    def __init__(self, state):
        self.state = state

    def open(self, url, *args, **kwargs):
        parts = urlsplit(url)
        parameters = [(name, value) for name, value in parse_qsl(parts.query, keep_blank_values=True)
                      if name != 'token']
        parameters.append(('token', self.state.token))
        return self.state.opener.open(urlunsplit(parts._replace(query=urlencode(parameters))),
                                      *args, **kwargs)


class _ListAccount:
    """Small local service view for existing ListManager and List enrichment.

    No HTTP client is created. The owning helper state keeps the configured
    transport alive; the per-call manager and streamed enrichment stay local.
    """

    def __init__(self, state):
        from intermine314.service.service import Service

        self.root = state.root
        self.opener = _AccountOpener(state)
        self.LIST_PATH = Service.LIST_PATH
        self.LIST_ENRICHMENT_PATH = Service.LIST_ENRICHMENT_PATH
        self.compatibility = (state.service.compatibility if state.service is not None
                              else state.registry.compatibility)
        self.version = (state.service.version if state.service is not None
                        else int(state.read('/version/ws')))


def _go_rows(list_name):
    from intermine314.lists.listmanager import ListManager

    account = _ListAccount(_configured())
    manager = ListManager(account)
    item = manager.get_list(list_name)
    if item is None:
        raise ValueError('No such list available')
    with closing(item.calculate_enrichment(widget='go_enrichment_for_gene')) as stream:
        rows = [dict(row) for row in islice(stream, 5)]
    if not rows:
        raise ValueError('No enrichment data available')
    return rows


def _draw(pyplot, x, y, annotations, *, title, xlabel, ylabel, rotation):
    _, ax = pyplot.subplots()
    # Matplotlib's internal affine arithmetic warns on valid log(0)=-inf and
    # log(negative)=nan. Retain the actual values in patches and annotations.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered',
                                category=RuntimeWarning, module=r'matplotlib\..*')
        bars = ax.bar(range(len(x)), y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x)), labels=x, rotation=rotation)
    for bar, label in zip(bars, annotations, strict=True):
        finite = isfinite(bar.get_height())
        bar.set_visible(finite)
        ax.annotate(str(label), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                    visible=finite)
    ax.margins(y=0.1)
    pyplot.show()


def _plot_go(list_name, *, p_value):
    pl, pyplot = _plot_dependencies()
    rows = _go_rows(list_name)
    field = 'p-value' if p_value else 'matches'
    annotation = 'matches' if p_value else 'populationAnnotationCount'
    try:
        data = pl.DataFrame(rows).lazy().select(
            pl.col('identifier').cast(pl.String).alias('x'),
            pl.col(field).cast(pl.Float64).alias('y'),
            pl.col(annotation).alias('label'),
        ).collect()
        if data.null_count().row(0) != (0, 0, 0):
            raise ValueError('null enrichment fields')
    except (pl.exceptions.PolarsError, TypeError, ValueError) as error:
        raise ValueError('Malformed enrichment data') from error
    _draw(pyplot, data['x'].to_list(), data['y'].to_list(), data['label'].to_list(),
          title=('GO Term vs p-value (Label: Gene count)' if p_value
                 else 'GO Term vs Count (Label: Annotation)'),
          xlabel='GO Term', ylabel='p_value' if p_value else 'Number of Genes',
          rotation='horizontal')


def plot_go_vs_p(list_name):
    """Show the first five GO p-values annotated with matching gene counts."""
    _plot_go(list_name, p_value=True)


def plot_go_vs_count(list_name):
    """Show the first five matching gene counts annotated with population counts."""
    _plot_go(list_name, p_value=False)


def get_query(xml):
    """Return raw tab-split rows and a final empty-string newline sentinel.

    Unlike the source, a final row without a newline is also split correctly.
    Account credentials are included so private query results remain accessible.
    """
    lines = _configured().read('/query/results', {'query': xml}).split('\n')
    return [line.split('\t') if index < len(lines) - 1 or line else ''
            for index, line in enumerate(lines)]


def query_to_barchart_log(xml, resp):
    """Show column two against column three; exactly 'true' logs and rounds y.

    Empty data, short rows, nonnumeric values and missing XML views raise
    ValueError before figure creation. Natural log preserves -inf/nan for
    zero/negative values, as in the original NumPy expression.
    """
    pl, pyplot = _plot_dependencies()
    try:
        views = ElementTree.fromstring(xml).attrib['view'].split()
        if len(views) < 3:
            raise ValueError('three views required')
    except (ElementTree.ParseError, KeyError, TypeError, ValueError) as error:
        raise ValueError('Malformed query XML: three views required') from error
    rows = get_query(xml)
    if rows and rows[-1] == '':
        rows = rows[:-1]
    if not rows or any(not isinstance(row, list) or len(row) < 3 for row in rows):
        raise ValueError('Empty or malformed query data: three columns required')
    try:
        data = pl.DataFrame({'x': [row[1] for row in rows],
                             'y': [row[2] for row in rows]}).lazy()
        value = pl.col('y').str.strip_chars().cast(pl.Float64)
        if resp == 'true':
            value = value.log().round(2)
        data = data.with_columns(value).collect()
    except (pl.exceptions.PolarsError, TypeError, ValueError) as error:
        raise ValueError('Malformed query data: numeric third column required') from error
    values = data['y'].to_list()
    _draw(pyplot, data['x'].to_list(), values, values, title=rows[0][0],
          xlabel=views[1], ylabel='log(' + views[2] + ')' if resp == 'true' else views[2],
          rotation='vertical')
