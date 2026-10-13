import json
import threading
from io import BytesIO
from urllib.parse import parse_qs
from xml.etree import ElementTree

import pytest

from intermine314.export.runtime import _CURRENT
from intermine314.model import Model
from intermine314.results import ResultObject, prefetch_objects
from intermine314.service.service import Service
from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession, fixture_bytes


def objects_payload(rows, successful=True):
    return ('{"results":[\n' + ',\n'.join(json.dumps(row) for row in rows) +
            '\n],"wasSuccessful":' + str(successful).lower() + ',"error":"failed"}').encode()


class ObjectSession(FixtureSession):
    def __init__(self, *, successful=True):
        super().__init__(FixtureSession.service().routes)
        self.successful = successful
        self.queries = []
    def request(self, method, url, data=None, **kwargs):
        if method == 'POST':
            form = parse_qs(data.decode())
            assert 'size' not in form and form['format'] == ['jsonobjects']
            xml = ElementTree.fromstring(form['query'][0])
            identifiers = [int(value.text) for value in xml.findall('constraint/value')]
            self.queries.append((xml, identifiers))
            self.routes['POST', '/service/query/results'] = objects_payload([
                {'objectId': identifier, 'class': 'Department', 'name': f'D{identifier}',
                 'company': {'objectId': 100, 'name': 'Company'},
                 'employees': [{'objectId': i, 'name': f'E{i}'} for i in range(20)]}
                for identifier in reversed(identifiers)], self.successful)
        return super().request(method, url, data=data, **kwargs)


def test_prefetch_batches_preserve_order_cardinality_collections_and_identity():
    session = ObjectSession()
    with Service(SERVICE_ROOT, session=session) as service:
        original = [ResultObject({'objectId': i}, service.model.get_class('Department')) for i in range(1, 6)]
        original.insert(1, original[0])
        output = list(prefetch_objects(original, 'name', 'company', 'employees', batch_size=2, max_workers=2))
        assert all(before is after for before, after in zip(original, output))
        assert len(output) == len(original) and output[0] is output[1]
        assert [obj.name for obj in output] == ['D1', 'D1', 'D2', 'D3', 'D4', 'D5']
        assert len(session.queries) == 3
        for obj in output:
            assert obj.company is obj.company
            assert len(obj.employees) == 20 and obj.employees[19].name == 'E19'
        assert len(session.queries) == 3
        for xml, identifiers in session.queries:
            assert len(identifiers) <= 2
            assert xml.find('constraint').get('op') == 'ONE OF'
            assert {join.get('path') for join in xml.findall('join')} == {'Department.company', 'Department.employees'}
        assert all(response.closed for response in session.responses)


def test_prefetch_skips_loaded_selected_and_idless_fields():
    session = ObjectSession()
    with Service(SERVICE_ROOT, session=session) as service:
        cld = service.model.get_class('Department')
        objects = [ResultObject({'objectId': 1, 'name': None, 'company': None}, cld),
                   ResultObject({'objectId': 2}, cld, ['Department.name', 'Department.company.name']),
                   ResultObject({}, cld)]
        assert list(prefetch_objects(objects, 'name', 'company')) == objects
        assert all(obj.name is None and obj.company is None for obj in objects)
        assert not session.queries


def test_prefetch_failed_footer_leaves_window_unchanged_and_retryable():
    session = ObjectSession(successful=False)
    with Service(SERVICE_ROOT, session=session) as service:
        objects = [ResultObject({'objectId': i}, service.model.get_class('Department')) for i in range(1, 5)]
        from intermine314.service.errors import WebserviceError
        with pytest.raises(WebserviceError):
            list(prefetch_objects(objects, 'name', batch_size=2, max_workers=2))
        assert all(obj._attr_cache == {} and 'name' not in obj._data for obj in objects)
        assert all(response.closed for response in session.responses)
        session.successful = True
        assert [obj.name for obj in prefetch_objects(objects, 'name')] == ['D1', 'D2', 'D3', 'D4']


def test_prefetch_groups_exact_models_and_services():
    first, second = ObjectSession(), ObjectSession()
    with Service(SERVICE_ROOT, session=first) as a, Service(SERVICE_ROOT, session=second) as b:
        model = Model(BytesIO(fixture_bytes('model.xml')), service=a)
        objects = [ResultObject({'objectId': 1}, a.model.get_class('Department')),
                   ResultObject({'objectId': 2}, model.get_class('Department')),
                   ResultObject({'objectId': 3}, b.model.get_class('Department'))]
        assert [obj.name for obj in prefetch_objects(objects, 'name')] == ['D1', 'D2', 'D3']
        assert len(first.queries) == 2 and len(second.queries) == 1
        assert objects[1]._cld.model is model


def test_prefetch_bounds_input_window_and_does_not_hold_context_over_yield():
    session = ObjectSession()
    with Service(SERVICE_ROOT, session=session) as service:
        consumed = []
        def source():
            for i in range(100):
                consumed.append(i)
                yield ResultObject({'objectId': i}, service.model.get_class('Department'))
        iterator = prefetch_objects(source(), 'name', batch_size=2, max_workers=2)
        assert next(iterator).name == 'D0'
        assert consumed == [0, 1, 2, 3]
        assert _CURRENT.get() is None
        iterator.close()
        assert all(response.closed for response in session.responses)


def test_prefetch_uses_multiple_workers_for_independent_queries(monkeypatch):
    import intermine314._prefetch as implementation

    original = implementation._fetch
    barrier = threading.Barrier(2)
    threads = set()
    def fetch(request):
        threads.add(threading.get_ident())
        barrier.wait(2)
        return original(request)
    monkeypatch.setattr(implementation, '_fetch', fetch)
    session = ObjectSession()
    with Service(SERVICE_ROOT, session=session) as service:
        objects = [ResultObject({'objectId': i}, service.model.get_class('Department')) for i in range(4)]
        assert len(list(prefetch_objects(objects, 'name', batch_size=2, max_workers=2))) == 4
    assert len(threads) == 2


@pytest.mark.parametrize('options', [{'batch_size': 0}, {'batch_size': True}, {'max_workers': -1}])
def test_invalid_prefetch_options_fail_before_consuming_input(options):
    def source():
        pytest.fail('Invalid options must precede input consumption')
        yield
    with pytest.raises((ValueError, TypeError)):
        prefetch_objects(source(), 'name', **options)
