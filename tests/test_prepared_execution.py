from dataclasses import FrozenInstanceError
from urllib.parse import parse_qs

import pytest

from intermine314.query import ParallelOptions, Query
from intermine314.query import executor as execution_module
from tests.fixtures.compatibility import FixtureSession


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_parallel_snapshot_encodes_once_and_isolates_later_mutations(native_service_factory, monkeypatch, profile):
    session = FixtureSession.service(rows=b'{"results":[\n["Ada"]\n],"wasSuccessful":true}\n')
    service = native_service_factory(session=session, compatibility=profile)
    query = service.select('Employee.name')
    query.add_constraint('Employee.name', '=', 'before')
    before = query.to_xml()
    calls = []
    encode = execution_module.query_spec_to_xml

    def tracked(spec):
        calls.append(spec)
        return encode(spec)

    monkeypatch.setattr(execution_module, 'query_spec_to_xml', tracked)
    rows = query.run_parallel(size=3, parallel_options=ParallelOptions(page_size=1, max_workers=2, ordered=True))
    query.get_constraint('A').value = 'after'
    query.add_view('Employee.age')
    assert list(rows) == [{'Employee.name': 'Ada'}] * 3
    params = [parse_qs(request.data.decode()) for request in session.requests if request.path.endswith('/query/results')]
    assert len(calls) == 1 and len(params) == 3
    assert all(param['query'] == [before] for param in params)
    assert sorted(int(param['start'][0]) for param in params) == [0, 1, 2]


def test_prepared_execution_does_not_retain_mutable_constraints(native_service_factory):
    service = native_service_factory()
    query = service.select('Employee.name')
    query.add_constraint('Employee.name', '=', 'before')
    spec = query.to_spec()
    executor = service.execute(spec)
    prepared = executor.prepare()
    before = prepared.encoded
    spec.constraints[0].value = 'after'
    assert prepared.encoded == before
    assert b'before' in prepared.encoded and b'after' not in prepared.encoded
    assert executor.spec is spec
    with pytest.raises(FrozenInstanceError):
        prepared.encoded = b''


def test_custom_result_provider_remains_in_control():
    query = Query().select('Gene.id')
    calls = []

    def results(**kwargs):
        calls.append(kwargs['start'])
        return iter([{'Gene.id': kwargs['start']}])

    query.results = results
    assert list(query.run_parallel(size=3, parallel_options=ParallelOptions(page_size=1, max_workers=2, ordered=True))) == [
        {'Gene.id': i} for i in range(3)
    ]
    assert sorted(calls) == [0, 1, 2]


def test_prepared_parameters_reencode_mutations_and_deleted_none():
    from urllib.parse import urlencode

    from intermine314.service.session import encode_dict

    fields = (('query', 'Å & before'), ('optional', None))
    encoded = urlencode(encode_dict(dict(fields)), True).encode()
    params = execution_module._PreparedParameters(fields, encoded)
    params['start'] = 2
    for mutate in (lambda: None, lambda: params.update(query='after'),
                   lambda: params.pop('optional'), lambda: params.update(extra=['a', 'b'])):
        mutate()
        assert parse_qs(params.encoded().decode(), keep_blank_values=True) == parse_qs(
            urlencode(encode_dict(params), True), keep_blank_values=True)


@pytest.mark.parametrize('query_type', ['query', 'template'])
def test_custom_wire_overrides_survive_serial_parallel_and_export(tmp_path, query_type):
    from io import BytesIO
    from types import SimpleNamespace

    import pyarrow.parquet as pq

    from intermine314.query import Template
    from intermine314.query.executor import QueryExecutor

    calls = []

    class Opener:
        def open(self, url, data=None):
            calls.append(('count', url, parse_qs(data.decode())))
            return BytesIO(b'3')

    def results(path, params, row, views, cld):
        calls.append(('results', path, dict(params)))
        return iter([{'Gene.id': params['start']}])

    service = SimpleNamespace(root='https://offline.invalid', QUERY_PATH='/query/results',
                              TEMPLATEQUERY_PATH='/template/results', prefetch_depth=1,
                              prefetch_id_only=False, opener=Opener(), get_results=results)
    service.execute = lambda spec: QueryExecutor(service, spec)
    base = Query if query_type == 'query' else Template

    class CustomQuery(base):
        def to_query_params(self):
            return {'marker': 'custom'}

        def get_results_path(self):
            return '/custom/results'

    query = CustomQuery(service=service, root='Gene').select('id')
    before = query.to_xml()
    assert list(query.results()) == [{'Gene.id': 0}]
    assert query.count() == 3
    options = ParallelOptions(page_size=1, max_workers=2, ordered=True)
    expected = [{'Gene.id': i} for i in range(3)]
    assert list(query.run_parallel(size=3, parallel_options=options)) == expected
    path = tmp_path / 'custom.parquet'
    query.to_parquet(path, size=3, parallel_options=options)
    assert pq.read_table(path).to_pylist() == expected
    assert all(path.endswith('/custom/results') for _, path, _ in calls)
    assert all(params['marker'] in ('custom', ['custom']) for _, _, params in calls)
    assert query.to_xml() == before
