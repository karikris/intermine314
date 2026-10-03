"""Pinned plotting behavior through strict offline managed transports and Agg."""

import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from urllib.parse import parse_qs, urlsplit

import pytest

from intermine314.webservice import Registry, Service
from tests.fixtures.compatibility import FixtureOpener, fixture_bytes
from tests.test_lists_enrichment import payload
from tests.test_query_manager_helpers import (
    OPTIONS,
    TOKEN,
    assert_closed,
    helper_session,
)

XML = '<query view="Gene.symbol Gene.label Gene.value"/>'
ROWS = [dict(identifier=f'GO:{i}', description='term', **{'p-value': i / 100},
             matches=i, populationAnnotationCount=i * 10) for i in range(1, 7)]


def plot_session(body=None):
    session = helper_session()
    session.routes.update({
        ('GET', '/custom/service/query/results'): fixture_bytes('plot-query-results.txt') if body is None else body,
        ('GET', '/custom/service/lists'): json.dumps({'wasSuccessful': True, 'lists': [
            dict(name='private+&人', title='private', type='Gene', size=6)]}).encode(),
        ('POST', '/custom/service/list/enrichment'): fixture_bytes('plot-enrichment.json'),
    })
    return session


@pytest.fixture
def bc():
    module = importlib.import_module('intermine314.bar_chart')
    yield module
    if module._state is not None:
        module._state.close()
    module._state = None
    for name in ('mine', 'token'):
        module.__dict__.pop(name, None)


@pytest.fixture
def plt(monkeypatch):
    matplotlib = pytest.importorskip('matplotlib', reason='optional plots extra is not installed')

    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot

    calls = []
    monkeypatch.setattr(pyplot, 'show', lambda: calls.append(True))
    pyplot.show_calls = calls
    yield pyplot
    pyplot.close('all')


def configure(bc, session, **options):
    assert bc.save_mine_and_token('OfflineMine', TOKEN, session=session, **options) is None


def test_five_historical_plotting_helpers_are_available():
    assert importlib.util.find_spec('intermine314.bar_chart') is not None


@pytest.mark.parametrize('wire,rows', [
    (' gene \t α \t1\n', [[' gene ', ' α ', '1'], '']),
    ('gene\ta\t1', [['gene', 'a', '1']]),
    ('gene\ta\t1\ngene\tb\t2', [['gene', 'a', '1'], ['gene', 'b', '2']]),
    ('', ['']), ('\n', [[''], '']), ('gene\t\t1\n', [['gene', '', '1'], '']),
    ('gene\ta\t1\r\n', [['gene', 'a', '1\r'], '']),
])
def test_raw_query_tsv_preserves_whitespace_sentinel_and_repairs_final_row(bc, wire, rows):
    session = plot_session(wire.encode())
    configure(bc, session, **OPTIONS)
    assert bc.get_query(XML) == rows
    call = session.requests[-1]
    assert parse_qs(urlsplit(call.url).query) == {'query': [XML], 'token': [TOKEN]}
    assert call.method == 'GET' and call.data is None
    assert call.options['verify'] == '/custom/ca.pem'
    assert call.options['timeout'] == (10.0, 13.0)
    assert call.headers['User-Agent'] == 'saved-query-client'
    assert 'Authorization' not in call.headers
    assert_closed(session)


@pytest.mark.parametrize('resp,y,label', [('true', [0, 0.69], 'log(Gene.value)'),
                                         ('false', [1, 2], 'Gene.value'),
                                         ('TRUE', [1, 2], 'Gene.value')])
def test_query_figure_original_columns_labels_annotations_show_and_none(bc, plt, resp, y, label):
    session = plot_session(b'gene\ta\t1\textra\ngene\tb\t2\textra')
    configure(bc, session)
    assert bc.query_to_barchart_log(XML, resp) is None
    ax = plt.gcf().axes[0]
    assert [bar.get_height() for bar in ax.patches] == y
    assert ax.get_title() == 'gene'
    assert ax.get_xlabel() == 'Gene.label' and ax.get_ylabel() == label
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ['a', 'b']
    assert all(tick.get_rotation() == 90 for tick in ax.get_xticklabels())
    assert [text.get_text() for text in ax.texts] == [str(float(value)) for value in y]
    assert ax.margins()[1] == 0.1 and plt.show_calls == [True]
    plt.gcf().canvas.draw()
    assert_closed(session)


def test_query_log_zero_negative_follow_natural_log_nonfinite_semantics(bc, plt):
    session = plot_session(b'gene\tpositive\t1\ngene\tzero\t0\ngene\tnegative\t-1\n')
    configure(bc, session)
    assert bc.query_to_barchart_log(XML, 'true') is None
    ax = plt.gcf().axes[0]
    heights = [bar.get_height() for bar in ax.patches]
    assert heights[:2] == [0, -math.inf] and math.isnan(heights[2])
    assert [text.get_text() for text in ax.texts] == ['0.0', '-inf', 'nan']
    plt.gcf().canvas.draw()


def test_query_numeric_whitespace_and_crlf_match_source_float_parsing(bc, plt):
    session = plot_session(b'gene\t a \t 1 \r\ngene\t b \t +2 \r\n')
    configure(bc, session)
    assert bc.query_to_barchart_log(XML, 'true') is None
    ax = plt.gcf().axes[0]
    assert [bar.get_height() for bar in ax.patches] == [0.0, 0.69]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [' a ', ' b ']
    plt.gcf().canvas.draw()


@pytest.mark.parametrize('body', [b'', b'\n', b'gene\ta\n', b'gene\ta\tx\n',
                                  b'gene\ta\t1\n\ngene\tb\t2\n'])
def test_empty_or_malformed_query_data_rejects_before_figure(bc, plt, body):
    session = plot_session(body)
    configure(bc, session)
    with pytest.raises(ValueError, match='query'):
        bc.query_to_barchart_log(XML, 'true')
    assert plt.get_fignums() == [] and plt.show_calls == []
    assert_closed(session)


@pytest.mark.parametrize('xml', ['broken', '<query/>', '<query view="Gene.symbol Gene.label"/>'])
def test_malformed_query_xml_rejects_before_query_fetch(bc, plt, xml):
    session = plot_session()
    configure(bc, session)
    before = len(session.requests)
    with pytest.raises(ValueError, match='query'):
        bc.query_to_barchart_log(xml, 'true')
    assert len(session.requests) == before and plt.get_fignums() == []


@pytest.mark.parametrize('name,title,ylabel,field,annotation', [
    ('plot_go_vs_p', 'GO Term vs p-value (Label: Gene count)', 'p_value', 'p-value', 'matches'),
    ('plot_go_vs_count', 'GO Term vs Count (Label: Annotation)', 'Number of Genes', 'matches', 'populationAnnotationCount'),
])
@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_go_first_five_real_stream_and_figure_selected_account(bc, plt, monkeypatch, name, title, ylabel, field, annotation, profile):
    from intermine314.service.session import JSONIterator

    consumed = []
    original = JSONIterator.__next__

    def observed_next(stream):
        row = original(stream)
        consumed.append(row.identifier)
        return row

    monkeypatch.setattr(JSONIterator, '__next__', observed_next)
    session = plot_session()
    configure(bc, session, compatibility=profile, **OPTIONS)
    assert getattr(bc, name)('private+&人') is None
    assert consumed == [row['identifier'] for row in ROWS[:5]]
    ax = plt.gcf().axes[0]
    assert ax.get_title() == title and ax.get_xlabel() == 'GO Term' and ax.get_ylabel() == ylabel
    assert [bar.get_height() for bar in ax.patches] == [row[field] for row in ROWS[:5]]
    assert [text.get_text() for text in ax.texts] == [str(row[annotation]) for row in ROWS[:5]]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == consumed
    assert all(tick.get_rotation() == 0 for tick in ax.get_xticklabels())
    assert ax.margins()[1] == 0.1 and plt.show_calls == [True]
    for call in session.requests[-2:]:
        assert parse_qs(urlsplit(call.url).query) == {'token': [TOKEN]}
        assert 'Authorization' not in call.headers
        assert call.options['verify'] == '/custom/ca.pem'
        assert call.options['timeout'] == (10.0, 13.0)
        assert call.headers['User-Agent'] == 'saved-query-client'
    assert parse_qs(session.requests[-1].data.decode(), keep_blank_values=True) == {
        'list': ['private+&人'], 'widget': ['go_enrichment_for_gene'],
        'correction': ['Holm-Bonferroni'], 'maxp': ['0.05'], 'filter': ['']}
    plt.gcf().canvas.draw()
    assert_closed(session)
    assert session.close_calls == 0


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_borrowed_service_identity_cache_and_configuration_preserved(bc, plt, profile):
    session = plot_session()
    with Registry(session=session, compatibility=profile) as registry, Service(
        'https://data.example/custom', token='other secret', session=session,
        compatibility=profile, **OPTIONS,
    ) as service:
        original = service.opener.token, service.opener.auth_header, service.compatibility
        assert bc.save_mine_and_token('OfflineMine', TOKEN, registry=registry, service=service) is None
        assert bc.plot_go_vs_count('private+&人') is None
        assert (service.opener.token, service.opener.auth_header, service.compatibility) == original
        assert service._list_manager is None
        assert parse_qs(urlsplit(session.requests[-1].url).query) == {'token': [TOKEN]}
        assert 'Authorization' not in session.requests[-1].headers
        bc._state.close()
        assert not service._closed and not registry._closed and session.close_calls == 0


def test_custom_opener_is_used_for_list_and_enrichment_without_new_service(bc, plt):
    session = plot_session()
    opener = FixtureOpener({key: value for key, value in session.routes.items() if key[1].startswith('/custom/')})
    with Registry(session=session) as registry:
        assert bc.save_mine_and_token('OfflineMine', TOKEN, registry=registry, opener=opener) is None
        assert bc._state.service is None
        assert bc.plot_go_vs_p('private+&人') is None
        assert [call.path for call in opener.requests] == ['/custom/service/user/queries',
            '/custom/service/version/ws', '/custom/service/lists', '/custom/service/list/enrichment']
        assert all(parse_qs(urlsplit(call.url).query)['token'] == [TOKEN] for call in opener.requests)
        assert all(connection.close_calls == 1 for connection in opener.connections)
        bc._state.close()
        assert opener.close_calls == 0 and not registry._closed


@pytest.mark.parametrize('kind', ['empty', 'bad-row', 'missing-list'])
def test_go_empty_or_malformed_data_and_missing_list_close_without_figure(bc, plt, kind):
    session = plot_session()
    if kind == 'missing-list':
        session.routes[('GET', '/custom/service/lists')] = b'{"wasSuccessful":true,"lists":[]}'
    else:
        session.routes[('POST', '/custom/service/list/enrichment')] = payload([] if kind == 'empty' else [{}])
    configure(bc, session)
    with pytest.raises(ValueError, match='list|enrichment'):
        bc.plot_go_vs_p('private+&人')
    assert plt.get_fignums() == [] and plt.show_calls == []
    assert_closed(session)


def test_configuration_source_errors_independent_query_manager_and_replacement(bc, caplog, capsys):
    from intermine314 import query_manager as qm

    other = helper_session()
    assert qm.save_mine_and_token('OfflineMine', 'other token', session=other) is None
    try:
        session = plot_session()
        configure(bc, session)
        previous = bc._state
        bad = plot_session()
        bad.routes[('GET', '/custom/service/user/queries')] = b'{}'
        assert bc.save_mine_and_token('OfflineMine', TOKEN, session=bad) == 'An exception of type KeyError occurred. Check token'
        assert bc._state is None and previous.service._closed and previous.registry._closed
        assert qm.get_all_query_names() == 'Zulu, a/+&人'
        with pytest.raises(RuntimeError, match='save_mine_and_token'):
            bc.get_query(XML)
        bad.routes[('GET', '/service/instances/OfflineMine')] = b'{}'
        assert bc.save_mine_and_token('OfflineMine', TOKEN, session=bad) == 'An exception of type KeyError occurred. Check mine'
        assert TOKEN not in caplog.text + capsys.readouterr().out
        assert_closed(session)
        assert_closed(bad)
    finally:
        qm._state.close()
        qm._state = None


def run_script(script):
    result = subprocess.run([sys.executable, '-c', script], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_import_configuration_and_raw_query_keep_analytics_lazy():
    run_script('''
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'numpy', 'matplotlib', 'lxml', 'intermine'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
from intermine314 import bar_chart as bc
assert 'intermine314.service.service' not in sys.modules
from tests.test_query_manager_helpers import helper_session
session = helper_session()
session.routes[('GET', '/custom/service/query/results')] = b'gene\\ta\\t1\\n'
assert bc.save_mine_and_token('OfflineMine', 'secret', session=session) is None
assert bc.get_query('<query/>') == [['gene', 'a', '1'], '']
bc._state.close()
''')


def test_missing_optional_plots_dependency_is_actionable_before_network():
    run_script('''
import importlib.abc
import sys
class BlockPlots(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'matplotlib':
            raise ModuleNotFoundError('no matplotlib', name='matplotlib')
sys.meta_path.insert(0, BlockPlots())
from intermine314 import bar_chart as bc
for method, args in [(bc.plot_go_vs_p, ('private',)), (bc.plot_go_vs_count, ('private',)),
                     (bc.query_to_barchart_log, ('<query/>', 'true'))]:
    try:
        method(*args)
    except ImportError as error:
        assert 'intermine314[plots]' in str(error) and 'matplotlib' in str(error)
    else:
        raise AssertionError('missing matplotlib accepted')
assert bc._state is None
''')


@pytest.mark.parametrize('identity', ['token', 'basic'])
def test_borrowed_authenticated_custom_opener_clones_identity_and_keeps_config(bc, plt, identity):
    from intermine314.results import InterMineURLOpener

    session = plot_session()
    auth = {'token': 'other secret'} if identity == 'token' else {'credentials': ('user', 'password')}
    with Registry(session=session) as registry, InterMineURLOpener(
        session=session, **auth, request_timeout=13, verify_tls='/custom/ca.pem',
        user_agent='plot-opener', proxy_url='socks5h://127.0.0.1:9050', tor_mode=True,
    ) as opener:
        original = opener.token, opener.auth_header, opener.using_authentication
        assert bc.save_mine_and_token('OfflineMine', TOKEN, registry=registry, opener=opener) is None
        assert bc.plot_go_vs_p('private+&人') is None
        assert (opener.token, opener.auth_header, opener.using_authentication) == original
        for call in session.requests[-3:]:
            assert parse_qs(urlsplit(call.url).query)['token'] == [TOKEN]
            assert 'Authorization' not in call.headers
            assert call.options['verify'] == '/custom/ca.pem'
            assert call.options['timeout'] == (10, 13)
            assert call.headers['User-Agent'] == 'plot-opener'
        bc._state.close()
        assert opener._session is session and session.close_calls == 0
        assert_closed(session)


@pytest.mark.parametrize('operation', ['query', 'go'])
@pytest.mark.parametrize('failure', [OSError('read failure'), KeyboardInterrupt('interrupted')])
def test_raw_read_or_enrichment_iteration_failure_closes_response(bc, plt, monkeypatch, operation, failure):
    from intermine314.service import session as transport

    session = plot_session()
    configure(bc, session)
    if operation == 'query':
        monkeypatch.setattr(transport._ResponseStreamAdapter, 'read', lambda *args, **kwargs: (_ for _ in ()).throw(failure))
    else:
        monkeypatch.setattr(transport._ResponseStreamAdapter, '__next__', lambda stream: (_ for _ in ()).throw(failure))
    with pytest.raises(type(failure)):
        bc.get_query(XML) if operation == 'query' else bc.plot_go_vs_count('private+&人')
    assert_closed(session)
    assert session.close_calls == 0 and bc._state is not None
    assert plt.get_fignums() == []


def test_owned_sessions_reconfiguration_failure_and_success_close_exactly_once(bc, monkeypatch):
    from intermine314.service import session as transport

    old, new, bad = plot_session(), plot_session(), plot_session()
    sessions = iter([old, new, bad])
    monkeypatch.setattr(transport, 'build_session', lambda **kwargs: next(sessions))
    assert bc.save_mine_and_token('OfflineMine', TOKEN) is None
    previous = bc._state
    assert bc.save_mine_and_token('OfflineMine', TOKEN) is None
    assert previous.service._closed and previous.registry._closed and old.close_calls == 1
    bc.mine, bc.token = 'changed', 'changed'
    assert bc.get_query(XML) == [['gene', 'a', '1'], ['gene', 'b', '2'], '']
    assert parse_qs(urlsplit(new.requests[-1].url).query)['token'] == [TOKEN]
    bad.routes[('GET', '/custom/service/user/queries')] = b'{}'
    assert bc.save_mine_and_token('OfflineMine', TOKEN) == 'An exception of type KeyError occurred. Check token'
    assert new.close_calls == bad.close_calls == 1 and bc._state is None
    for session in (old, new, bad):
        assert_closed(session)
