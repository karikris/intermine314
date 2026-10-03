"""Executable factory contracts adapted from the pinned 1.13.0 client."""
from io import StringIO
from itertools import permutations
from urllib.parse import parse_qs

import pytest

from intermine314.model import Column, Field, Model, ModelError
from intermine314.query import Query
from intermine314.query.builder import ConstraintError
from intermine314.service.service import Service
from tests.fixtures.compatibility import FixtureSession

XML = '<query model="testmodel" view="Employee.name"><constraint path="Employee.age" op="&gt;" value="20" code="A"/></query>'
EMPLOYEE = ['Employee.age', 'Employee.end', 'Employee.fullTime', 'Employee.id', 'Employee.name']


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('alias', ['select', 'new_query', 'query'])
def test_factory_descriptor_single_forms(native_service_factory, profile, alias):
    service = native_service_factory(compatibility=profile)
    factory = getattr(service, alias)
    employee = service.model.get_class('Employee')
    assert factory(employee).views == EMPLOYEE
    assert factory(employee.get_field('age')).views == ['Employee.age']
    # Inherited fields keep their declaration when selected on their own.
    assert factory(employee.get_field('name')).views == ['Employable.name']
    assert factory(employee.get_field('department')).views == ['Employee.department.id', 'Employee.department.name']
    assert factory(service.model.get_class('Department').get_field('employees')).views == [
        path.replace('Employee.', 'Department.employees.') for path in EMPLOYEE]
    assert factory(Field('age', 'int', employee)).views == ['Employee.age']
    assert factory(service.model.Employee.name).views == ['Employee.name']
    assert factory(service.model.Employee.department).views == ['Employee.department.id', 'Employee.department.name']
    assert factory(service.model.make_path('Employee.department')).views == ['Employee.department.id', 'Employee.department.name']
    assert Service.select is Service.new_query is Service.query


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_factory_mixed_forms_inherited_fields_and_no_discard(native_service_factory, profile):
    service = native_service_factory(compatibility=profile)
    employee = service.model.get_class('Employee')
    selected = service.select('name', [employee.get_field('age'), (service.model.Employee.department.name,)])
    assert selected.views == ['Employee.name', 'Employee.age', 'Employee.department.name']
    assert service.select(employee.get_field('name'), employee.get_field('age')).views == ['Employee.name', 'Employee.age']
    assert service.select(employee, employee.get_field('age')).views == EMPLOYEE + ['Employee.age']
    assert service.select(employee.get_field('department'), 'age').views == [
        'Employee.department.id', 'Employee.department.name', 'Employee.age']
    assert service.select(service.model.get_class('Manager'), employee.get_field('name')).views[-1] == 'Manager.name'
    assert service.select('name age', root=employee).views == ['Employee.name', 'Employee.age']


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_factory_reference_strings_keep_single_and_multi_defaults(native_service_factory, profile):
    service = native_service_factory(compatibility=profile)
    assert service.select('Employee.department').views == ['Employee.department.id', 'Employee.department.name']
    assert service.select('Employee.name').views == ['Employee.name']
    assert service.select('Employee').views == EMPLOYEE
    assert service.select('Employee.*').views == EMPLOYEE
    # Multiple strings keep existing profile-specific validation.
    if profile == 'legacy':
        with pytest.raises(ConstraintError, match='attribute'):
            service.select('Employee.department', 'Employee.age')
    else:
        assert service.select('Employee.department', 'Employee.age').views == ['Employee.department', 'Employee.age']


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_factory_column_refinement_and_column_select_clone_remain_independent(native_service_factory, profile):
    service = native_service_factory(compatibility=profile)
    employees = service.model.Department.employees
    employees < service.model.Manager
    query = service.select(employees, 'Department.name')
    assert 'Department.employees.title' in query.views
    assert query.views[-1] == 'Department.name'
    assert query.get_subclass_dict() == {'Department.employees': 'Manager'}
    assert query.do_verification is (profile == 'legacy')
    assert service.select(employees.title).views == ['Department.employees.title']
    query.verify()
    source = Query(service.model, service, root='Department', compatibility=profile).select('name').where(name='staff')
    column = Column('Department.employees', service.model, query=source)
    column < service.model.Manager
    refined = column.select('title')
    assert refined.views == ['Department.employees.title']
    assert refined.get_constraint('A').value == 'staff'
    assert source.views == ['Department.name'] and source.get_subclass_dict() == {}
    clone = refined.clone()
    clone.get_constraint('A').value = 'other'
    assert refined.get_constraint('A').value == 'staff'


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('alias', ['select', 'new_query', 'query'])
def test_factory_xml_is_executable_shared_and_borrowed(native_service_factory, offline_session_factory, profile, alias):
    session = offline_session_factory(rows=b'{"results":[\n["Ada"]\n],"wasSuccessful":true}')
    service = native_service_factory(session=session, compatibility=profile, prefetch_depth=2, prefetch_id_only=True)
    stream = StringIO(XML)
    query = getattr(service, alias)(xml=stream, root=service.model.get_class('Employee'))
    assert not stream.closed
    assert query.compatibility == profile and query.service is service and query.model is service.model
    assert query.prefetch_depth == 2 and query.prefetch_id_only
    assert query.root == ('Employee' if profile == 'native' else service.model.get_class('Employee'))
    assert [row for row in query.rows(row='dict')] == [{'Employee.name': 'Ada'}]
    payload = parse_qs(session.requests[-1].data.decode())
    assert payload['query'] == [query.to_xml()]
    assert sum(request.path.endswith('/model') for request in session.requests) == 1
    assert all(response.closed for response in session.responses)


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_factory_invalid_mixed_roots_and_xml_conflicts(native_service_factory, profile):
    service = native_service_factory(compatibility=profile)
    employee, department = service.model.get_class('Employee'), service.model.get_class('Department')
    for args in [(employee.get_field('age'), department.get_field('name')),
                 (employee, service.model.Department.name), (employee, 'Department.name')]:
        with pytest.raises(ModelError, match='[Rr]oot|compatible'):
            service.select(*args)
    with pytest.raises(ModelError):
        service.select(employee, 'unknown')
    with pytest.raises(TypeError, match='xml.*columns|columns.*xml'):
        service.select('Employee.name', xml=XML)
    with pytest.raises(TypeError, match='unexpected|Unsupported'):
        service.select(xml=XML, typo=True)
    with pytest.raises(TypeError, match='unexpected|Unsupported'):
        service.select('Employee.name', typo=True)


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_factory_prefetch_cache_tls_and_tor(native_service_factory, offline_session_factory, profile):
    session = offline_session_factory(routes={('GET', '/saved.xml'): XML.encode()})
    service = native_service_factory(session=session, compatibility=profile, prefetch_depth=2,
                                     prefetch_id_only=True, verify_tls='/ca/offline.pem',
                                     tor=True, proxy_url='socks5h://127.0.0.1:9050')
    selected = service.select(service.model.get_class('Employee').get_field('department'))
    assert 'Employee.department.employees.id' in selected.views
    assert any(join.path == 'Employee.department.employees' and join.style == 'OUTER' for join in selected.joins)
    assert service.select(xml='https://offline.example/saved.xml').service is service
    service.select(service.model.Employee.age)
    assert sum(request.path.endswith('/model') for request in session.requests) == 1
    assert all(request.options['verify'] == '/ca/offline.pem' for request in session.requests)
    assert all(response.closed for response in session.responses)
    service.close()
    assert session.close_calls == 0


def test_factory_native_stub_fallback_and_strict_legacy(native_service_factory, offline_session_factory):
    session = offline_session_factory(routes={('GET', '/service/model'): b'<model name="stub"/>'})
    service = native_service_factory(session=session)
    assert service.select('Missing').views == ['Missing.*']
    assert service.query('Missing.name').views == ['Missing.name']
    assert service.new_query(xml='<query view="Missing.name"/>').views == ['Missing.name']
    assert sum(request.path.endswith('/model') for request in session.requests) == 1
    assert not isinstance(service.select().model, Model)
    strict = native_service_factory(session=offline_session_factory(routes={('GET', '/service/model'): b'<model name="stub"/>'}), compatibility='legacy')
    with pytest.raises(ModelError):
        strict.select('Missing.name')


def test_factory_native_unknown_strings_stay_unverified(native_service_factory):
    service = native_service_factory()
    assert service.select('Employee.unknown').views == ['Employee.unknown']
    assert service.select('Missing.unknown').views == ['Missing.unknown']
    with pytest.raises(ModelError):
        service.select('Missing')
    assert service.select('Employee.name Employee.age').views == ['Employee.name', 'Employee.age']
    assert service.select('Employee.name', 'unknown').views == ['Employee.name', 'Employee.unknown']
    assert service.select().views == []


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_factory_and_standalone_where_keep_all_keyword_field_names(native_service_factory, offline_session_factory, profile):
    names = ['path', 'op', 'value', 'code', 'xml', 'root']
    model_xml = '<model name="keywords" package="test"><class name="Record">' + ''.join(
        f'<attribute name="{name}" type="String"/>' for name in names) + '</class></model>'
    service = native_service_factory(session=offline_session_factory(routes={('GET', '/service/model'): model_xml.encode()}), compatibility=profile)
    values = dict.fromkeys(names, 'data')
    for query in (service.select(service.model.get_class('Record')), Query(Model(model_xml), root='Record', compatibility=profile)):
        filtered = query.where(**values)
        assert {constraint.path: constraint.value for constraint in filtered.constraints} == {
            'Record.' + name: 'data' for name in names}
        assert query.constraints == []


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_factory_descriptor_export_retains_typed_schema(native_service_factory, offline_session_factory, tmp_path, profile):
    from decimal import Decimal

    import polars as pl

    class CountRowsSession(FixtureSession):
        def _capture(self, method, url, data=None, headers=None, **options):
            result = super()._capture(method, url, data, headers, **options)
            params = parse_qs(data.decode() if isinstance(data, bytes) else data or '')
            return b'1' if params.get('format') == ['count'] else result

    model_xml = b'<model name="export" package="test"><class name="Record"><attribute name="name" type="String"/><attribute name="amount" type="java.math.BigDecimal"/></class></model>'
    session = CountRowsSession(offline_session_factory(rows=b'{"results":[\n["0007","123456789.1234"]\n],"wasSuccessful":true}',
                                                      routes={('GET', '/service/model'): model_xml}).routes)
    service = native_service_factory(session=session, compatibility=profile)
    query = service.select(service.model.get_class('Record').get_field('name'), service.model.Record.amount)
    original = query.to_xml()
    frame = query.dataframe(parquet_path=tmp_path / 'selected.parquet')
    assert frame['Record.name'].dtype == pl.String
    assert isinstance(frame['Record.amount'].dtype, pl.Decimal)
    assert frame.rows() == [('0007', Decimal('123456789.1234'))]
    assert query.to_xml() == original
    assert sum(request.path.endswith('/model') for request in session.requests) == 1
    assert all(response.closed for response in session.responses)


def test_factory_non_column_path_object_and_conflicting_refinements(legacy_service_factory):
    class ReferencePath:
        def __str__(self):
            return 'Employee.department'

    service = legacy_service_factory()
    assert service.select(ReferencePath()).views == ['Employee.department.id', 'Employee.department.name']
    manager = service.model.Department.employees
    ceo = service.model.Department.employees
    manager < service.model.Manager
    ceo < service.model.CEO
    with pytest.raises(ModelError, match='Conflicting subclass'):
        service.select(manager, ceo)


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('subclass,error', [('Department', ConstraintError), ('Missing', ModelError)])
@pytest.mark.parametrize('selection', ['reference', 'attribute'])
def test_factory_rejects_invalid_column_refinement_before_expansion_or_export(
    native_service_factory, offline_session_factory, tmp_path, monkeypatch, profile, subclass, error, selection,
):
    session = offline_session_factory()
    service = native_service_factory(session=session, compatibility=profile)
    employees = service.model.Department.employees
    name = employees.name
    if subclass == 'Missing':
        employees._subclasses[str(employees)] = subclass
    else:
        employees < service.model.Department

    def reject_expansion(*args, **kwargs):
        pytest.fail('Invalid refinement reached wildcard expansion')

    monkeypatch.setattr(Query, '_expand_wildcard', reject_expansion)
    with pytest.raises(error):
        service.select(employees if selection == 'reference' else name).dataframe(
            parquet_path=tmp_path / 'invalid.parquet')
    assert not list(tmp_path.iterdir())
    assert not any(request.path == '/service/query/results' for request in session.requests)


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('names', list(permutations(['name', 'seniority', 'title'])))
def test_factory_multiple_inheritance_root_is_independent_of_field_order(native_service_factory, profile, names):
    service = native_service_factory(compatibility=profile)
    manager = service.model.get_class('Manager')
    query = service.select(*(manager.get_field(name) for name in names), manager.get_field(names[0]))
    assert query.views == ['Manager.' + name for name in (*names, names[0])]
    assert query.to_spec().root_class == 'Manager'
    assert query.do_verification is (profile == 'legacy')
    query.verify()


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('names', [('name', 'seniority'), ('seniority', 'name')])
def test_factory_multiple_inheritance_requires_selected_or_explicit_compatible_root(native_service_factory, profile, names):
    service = native_service_factory(compatibility=profile)
    manager = service.model.get_class('Manager')
    fields = [manager.get_field(name) for name in names]
    # Neither declaring class covers both fields. Do not invent a Manager root
    # merely because that unselected subclass exists elsewhere in the model.
    with pytest.raises(ModelError, match='[Rr]oot|compatible'):
        service.select(*fields)
    assert service.select(*fields, root=manager).views == ['Manager.' + name for name in names]
    with pytest.raises(ModelError, match='[Rr]oot|compatible'):
        service.select(*fields, root=service.model.get_class('Employee'))
