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
