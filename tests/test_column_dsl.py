"""Real model/query integration for the original Column expression DSL."""
from itertools import product
from types import SimpleNamespace
from urllib.parse import parse_qs
from xml.etree import ElementTree as ET

import pytest

from intermine314.model import CodelessNode, Column, ConstraintNode, Model, ModelError
from intermine314.query import Query
from intermine314.query.builder import ConstraintError
from intermine314.query.constraints import LogicGroup
from tests.fixtures.compatibility import FixtureSession


def evaluate(node, values):
    if isinstance(node, LogicGroup):
        left, right = evaluate(node.left, values), evaluate(node.right, values)
        return left and right if node.op == 'AND' else left or right
    return values[node.code]


def assert_bound_logic(query, node=None):
    node = query.get_logic() if node is None else node
    if isinstance(node, LogicGroup):
        assert_bound_logic(query, node.left)
        assert_bound_logic(query, node.right)
    else:
        assert node is query.get_constraint(node.code)


def test_legacy_column_binding_aliases_and_native_strings(legacy_service_factory, native_service_factory):
    service = legacy_service_factory()
    query = service.select('Employee.name')
    column = query.c('department')
    assert isinstance(column, Column)
    assert str(column.name) == 'Employee.department.name'
    assert column._model is service.model and column._query is query
    assert column.name._query is query
    assert Query.c is Query.column and Query.filter is Query.where
    assert Column.filter is Column.where
    assert native_service_factory().select('Employee.name').c('age') == 'Employee.age'


def test_where_tree_keeps_existing_or_and_binds_actual_reserved_codes(model_xml):
    query = Query(Model(model_xml), root='Employee').select('name')
    query.add_constraint('age', '>', 10, code='X')
    query.add_constraint('age', '<', 90, code='Z')
    query.add_constraint('name', 'NOT LIKE', 'excluded%', code='B')
    query.set_logic('(X OR Z) AND B')
    original_xml = query.to_xml()
    name, age = query.column('name'), query.column('age')
    refined = query.filter((name == 'Alice') | (age >= 30), fullTime=True)
    assert refined.get_constraint('A').value == 'Alice'
    assert refined.get_constraint('C').value == 30
    assert refined.get_constraint('D').value is True
    assert set(refined.get_logic().get_codes()) == {'X', 'Z', 'B', 'A', 'C', 'D'}
    for flags in product([False, True], repeat=6):
        values = dict(zip(['X', 'Z', 'B', 'A', 'C', 'D'], flags))
        assert evaluate(refined.get_logic(), values) == (
            (values['X'] or values['Z']) and values['B']
            and (values['A'] or values['C']) and values['D'])
    assert query.to_xml() == original_xml
    assert_bound_logic(refined)
    xml = ET.fromstring(refined.to_xml())
    assert xml.attrib['constraintLogic'] == str(refined.get_logic())
    assert {c.attrib['code'] for c in xml.findall('constraint')} == set(refined.constraint_dict)
    clone = refined.clone()
    assert_bound_logic(clone)
    assert clone.get_logic() is not refined.get_logic()
    clone.get_constraint('A').value = 'Changed'
    assert refined.get_constraint('A').value == 'Alice'


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_tree_codes_continue_beyond_z_and_repeated_nodes_are_distinct(model_xml, profile):
    model = Model(model_xml)
    query = Query(model, root='Employee', compatibility=profile)
    for value in range(26):
        query.add_constraint('age', '>=', value)
    node = model.Employee.name == 'Alice'
    refined = query.where(node | node)
    assert refined.get_logic().get_codes()[-2:] == ['AA', 'AB']
    assert refined.get_constraint('AA') is not refined.get_constraint('AB')
    assert_bound_logic(refined)
    assert '(AA or AB)' in str(refined.get_logic())
    assert len(query.constraints) == 26


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('form', ['node', 'tuple', 'triple', 'kwargs', 'mixed', 'unary'])
def test_where_call_forms_clone_and_extend_explicit_logic(model_xml, profile, form):
    model = Model(model_xml)
    query = Query(model, root='Employee', compatibility=profile).select('name')
    first = query.add_constraint('name', 'LIKE', 'A%')
    second = query.add_constraint('name', 'LIKE', 'B%')
    query.set_logic(first | second)
    age = model.Employee.age
    calls = {
        'node': lambda: query.where(age > 20),
        'tuple': lambda: query.where((age, '>', 20)),
        'triple': lambda: query.where(age, '>', 20),
        'kwargs': lambda: query.where(ConstraintNode(path=age, op='>', value=20)),
        'mixed': lambda: query.where(age, op='>', value=20),
        'unary': lambda: query.where(age, 'IS NOT NULL'),
    }
    refined = calls[form]()
    assert str(refined.get_logic()) == '(A or B) and C'
    assert refined.get_constraint('C').path == 'Employee.age'
    assert len(query.constraints) == 2
    assert_bound_logic(refined)
    refined.verify()


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_codeless_refinement_is_unconditional_and_omitted_from_logic(model_xml, profile):
    model = Model(model_xml)
    query = Query(model, root='Department', compatibility=profile).select('name')
    employees = model.Department.employees
    subclass = employees < model.Manager
    assert isinstance(subclass, CodelessNode)
    refined = query.where(subclass | (employees.title == 'Lead'))
    assert refined.get_subclass_dict() == {'Department.employees': 'Manager'}
    assert refined.get_logic().get_codes() == ['A']
    assert refined.get_constraint('A').path == 'Department.employees.title'
    assert query.get_subclass_dict() == {}
    only_subclass = query.where(subclass)
    assert only_subclass.get_logic() == ''
    assert 'constraintLogic' not in ET.fromstring(only_subclass.to_xml()).attrib
    only_subclass.verify()
    if profile == 'legacy':
        assert str(refined.c('employees').title) == 'Department.employees.title'
        assert refined.c('employees')._subclasses == refined.get_subclass_dict()
    existing = Query(model, root='Department', compatibility=profile).where(name='staff')
    assert str(existing.where(subclass).get_logic()) == 'A'
    root_refined = Query(model, root='Employee', compatibility=profile).where(
        model.Employee < model.CEO)
    root_refined.select('salary')
    assert root_refined.views == ['Employee.salary']
    assert root_refined.to_spec().root_class == 'Employee'


@pytest.mark.parametrize('operation,op', [
    (lambda c: c.age == None, 'IS NULL'),  # noqa: E711
    (lambda c: c.age != None, 'IS NOT NULL'),  # noqa: E711
    (lambda c: c.age < 10, '<'), (lambda c: c.age <= 10, '<='),
    (lambda c: c.age > 10, '>'), (lambda c: c.age >= 10, '>='),
    (lambda c: c.name == 'Alice', '='), (lambda c: c.name != 'Alice', '!='),
    (lambda c: c.name == ['Alice'], 'ONE OF'), (lambda c: c.name != ['Alice'], 'NONE OF'),
    (lambda c: c.name.in_(['Alice']), 'ONE OF'), (lambda c: c.name ^ ['Alice'], 'NONE OF'),
    (lambda c: c.name < ['Alice'], 'ONE OF'), (lambda c: c.name <= ['Alice'], 'ONE OF'),
    (lambda c: c.department == c.departmentThatRejectedMe, 'IS'),
    (lambda c: c.department != c.departmentThatRejectedMe, 'IS NOT'),
    (lambda c: c.department % ('sales', 'Company'), 'LOOKUP'),
    (lambda c: c.department % 'sales', 'LOOKUP'),
])
def test_live_column_operators_reach_constraints_and_xml(legacy_service_factory, operation, op):
    query = legacy_service_factory().select('Employee.name')
    refined = query.where(operation(query.c('Employee')))
    con = refined.get_constraint('A')
    assert con.op == op
    xml = ET.fromstring(refined.to_xml()).find('constraint')
    assert xml.attrib['op'] == con.to_dict()['op']
    if op in ('IS', 'IS NOT'):
        assert xml.attrib['loopPath'] == 'Employee.departmentThatRejectedMe'
    if op in ('ONE OF', 'NONE OF'):
        assert [v.text for v in xml.findall('value')] == ['Alice']
    if op == 'LOOKUP':
        assert xml.attrib['value'] == 'sales'
        assert xml.attrib.get('extraValue') == con.extra_value
    assert_bound_logic(refined)
    refined.verify()


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_column_selection_normalizes_class_reference_and_attribute_paths(native_service_factory, profile):
    service = native_service_factory(compatibility=profile)
    employee = service.model.Employee
    assert employee.select('name', 'age').views == ['Employee.name', 'Employee.age']
    default = employee.select()
    assert default.views == ['Employee.age', 'Employee.end', 'Employee.fullTime', 'Employee.id', 'Employee.name']
    assert employee.name.select().views == ['Employee.name']
    assert employee.department.select('name').views == ['Employee.department.name']
    assert 'Employee.department.name' in employee.department.select().views
    assert employee.where(employee.age > 20).get_constraint('A').value == 20
    assert employee.filter(name='Alice').get_constraint('A').value == 'Alice'
    assert service.select().select(employee.name, employee.age).views == ['Employee.name', 'Employee.age']
    assert service.select().select(employee).views == default.views


def test_column_select_preserves_refinement_and_bound_query_without_mutation(legacy_service_factory):
    query = legacy_service_factory().select('Employee.name').where(age=20)
    root = query.c('Employee')
    root < query.model.CEO
    selected = root.select('salary')
    assert selected.views == ['Employee.salary']
    assert selected.get_constraint('A').value == 20
    assert selected.get_subclass_dict() == {'Employee': 'CEO'}
    assert query.get_subclass_dict() == {}
    assert query.views == ['Employee.name']
    standalone = query.model.Department.employees
    standalone < query.model.Manager
    assert standalone.select('title').views == ['Department.employees.title']


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('version', [7, 8])
def test_attribute_column_iteration_uses_actual_offline_rows(native_service_factory, profile, version):
    rows = (b'{"results":[\n[{"value":"Alice"}],\n[{"value":null}]\n],"wasSuccessful":true}\n'
            if version < 8 else b'{"results":[\n["Alice"],\n[null]\n],"wasSuccessful":true}\n')
    class CountRowsSession(FixtureSession):
        def _capture(self, method, url, data=None, headers=None, **options):
            result = super()._capture(method, url, data, headers, **options)
            params = parse_qs(data.decode() if isinstance(data, bytes) else data or '')
            return b'2' if params.get('format') == ['count'] else result

    session = CountRowsSession(FixtureSession.service(version=version, rows=rows).routes)
    service = native_service_factory(session=session, compatibility=profile)
    assert list(service.model.Employee.name) == ['Alice', None]
    assert len(service.model.Employee.name) == 2
    assert all(response.closed for response in session.responses)
    assert session.close_calls == 0


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_where_membership_and_reference_ops_keep_profile_policy(model_xml, profile):
    model = Model(model_xml)
    query = Query(model, root='Employee', compatibility=profile)
    assert query.where_in('name', ['Alice']).get_constraint('A').op == 'ONE OF'
    assert query.where(name=('Alice', 'Bob')).get_constraint('A').op == 'ONE OF'
    member = query.where('Employee', 'IN', ['staff']).get_constraint('A') if profile == 'legacy' else query.where('name', 'IN', ['Alice']).get_constraint('A')
    assert member.op == ('IN' if profile == 'legacy' else 'ONE OF')
    for op in ('LIKE', 'NOT LIKE', 'CONTAINS'):
        assert query.where(model.Employee.name, op, 'A%').get_constraint('A').op == op
    assert query.where('Employee', 'ISA', ['Manager', 'CEO']).get_constraint('A').values == ['Manager', 'CEO']
    assert query.where('Employee', 'WITHIN', ['1..5']).get_constraint('A').op == 'WITHIN'


def test_dsl_errors_leave_original_query_untouched(model_xml):
    model = Model(model_xml)
    query = Query(model, root='Employee').select('name')
    before = query.to_xml()
    for operation, error in [
        (lambda: query.where(query.c('name') == query.c('end')), ConstraintError),
        (lambda: query.where(query.c('department') == query.c('address')), ConstraintError),
        (lambda: query.where(query.c('name') % 'Alice'), ConstraintError),
        (lambda: query.where(query.c('Employee') << model.Address), ConstraintError),
        (lambda: query.where('age', 'BOGUS', 20), TypeError),
        (lambda: query.where(ConstraintNode(path='age', op='>', value=20, typo=True)), TypeError),
        (lambda: query.column('missing'), ModelError),
        (lambda: query.c('name').missing, AttributeError),
        (lambda: query.c('name').in_('staff'), TypeError),
        (lambda: query.c('name') ^ 'staff', TypeError),
    ]:
        with pytest.raises(error):
            operation()
        assert query.to_xml() == before


def test_direct_model_column_selection_needs_no_service(model_xml):
    model = Model(model_xml)
    selected = model.Employee.select('name')
    assert selected.compatibility == 'legacy' and selected.service is None
    assert selected.views == ['Employee.name']
    assert selected.root is model.get_class('Employee')
    assert selected.where(model.Employee.age > 20).get_constraint('A').value == 20


def test_named_list_protocol_column_operations_upload_once(model_xml):
    model = Model(model_xml)
    query = Query(model, root='Employee')
    calls = []

    class UploadService:
        def create_list(self, uploaded):
            calls.append(uploaded)
            return type('NamedList', (), {'name': 'selected staff'})()

    class ListQuery:
        service = UploadService()

        def to_query(self):
            return self

        def make_list_constraint(self, path, op):
            return ConstraintNode(path, op, self)

    other = ListQuery()
    for node in (model.Employee == other, model.Employee != other,
                 model.Employee.in_(other), model.Employee ^ other):
        refined = query.where(node)
        assert refined.get_constraint('A').list_name == 'selected staff'
    assert calls == [other] * 4


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('op', ['IS NULL', 'IS NOT NULL'])
def test_unary_keyword_add_constraint_is_legacy_only_and_where_stays_equality(model_xml, profile, op):
    query = Query(Model(model_xml), root='Employee', compatibility=profile)
    con = query.add_constraint(age=op)
    assert con.op == (op if profile == 'legacy' else '=')
    xml = ET.fromstring(query.to_xml()).find('constraint')
    assert ('value' in xml.attrib) == (profile == 'native')
    keyword_where = Query(query.model, root='Employee', compatibility=profile).where(age=op)
    assert keyword_where.get_constraint('A').op == '='
    assert keyword_where.get_constraint('A').value == op


def test_where_multiple_trees_and_tuples_group_independently(model_xml):
    query = Query(Model(model_xml), root='Employee')
    employee = query.c('Employee')
    refined = query.where((employee.name == 'Alice') | (employee.name == 'Bob'),
                          (employee.age < 30) | (employee.age > 60),
                          (employee.fullTime, '=', True))
    assert str(refined.get_logic()) == '(A or B) and (C or D) and E'
    assert_bound_logic(refined)
    refined.verify()


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_subclass_on_right_still_validates_left_attribute(model_xml, profile):
    model = Model(model_xml)
    query = Query(model, root='Employee', compatibility=profile)
    root = model.Employee
    subclass = root < model.CEO
    refined = query.where((root.salary > 100) | subclass)
    assert refined.get_constraint('A').path == 'Employee.salary'
    assert refined.get_subclass_dict() == {'Employee': 'CEO'}
    assert str(refined.get_logic()) == 'A'
    refined.verify()


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('form', ['shift-left', 'shift-right', 'less-equal', 'keywords'])
def test_subclass_overloads_have_no_boolean_code(model_xml, profile, form):
    model = Model(model_xml)
    query = Query(model, root='Employee', compatibility=profile)
    calls = {
        'shift-left': lambda: query.where(model.Employee << model.CEO),
        'shift-right': lambda: query.where(model.Employee >> model.CEO),
        'less-equal': lambda: query.where(model.Employee <= model.CEO),
        'keywords': lambda: query.where(CodelessNode(path=model.Employee, subclass=model.CEO)),
    }
    refined = calls[form]()
    assert refined.get_subclass_dict() == {'Employee': 'CEO'}
    assert refined.get_logic() == ''
    assert refined.constraint_factory.get_next_code() == 'A'
    refined.select('salary').verify()
    assert query.constraints == []


def test_legacy_where_two_arguments_refine_class_and_imply_root_reference_ops(model_xml):
    model = Model(model_xml)
    query = Query(model, root='Employee')
    refined = query.where('Employee', 'CEO').select('salary')
    assert refined.get_subclass_dict() == {'Employee': 'CEO'}
    assert refined.get_logic() == ''
    assert query.where('LOOKUP', 'Alice').get_constraint('A').path == 'Employee'


def test_where_infers_root_and_does_not_share_supplied_constraint(model_xml):
    model = Model(model_xml)
    query = Query(model)
    supplied = query.constraint_factory.make_constraint('Employee.age', '>', 20)
    refined = query.where(supplied)
    assert query.root is None and query.constraints == []
    assert refined.root is model.get_class('Employee')
    assert refined.get_constraint('A') is not supplied
    refined.get_constraint('A').value = 30
    assert supplied.value == 20


@pytest.mark.parametrize('fields', [('name id',), ('name,id',), (['name', 'id'],)])
def test_reference_column_selection_normalizes_each_relative_field(model_xml, fields):
    model = Model(model_xml)
    query = model.Employee.department.select(*fields)
    assert query.views == ['Employee.department.name', 'Employee.department.id']


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('mixed', [False, True])
def test_codeless_node_keyword_construction_uses_subclass_factory(model_xml, profile, mixed):
    model = Model(model_xml)
    query = Query(model, root='Employee', compatibility=profile)
    node = (CodelessNode('Employee', subclass='CEO') if mixed
            else CodelessNode(path='Employee', subclass='CEO'))
    refined = query.where(node)
    assert refined.get_subclass_dict() == {'Employee': 'CEO'}
    assert refined.get_logic() == ''
    refined.select('salary').verify()


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('fields', [
    {'path': 'alpha', 'op': 'beta'},
    {'path': 'alpha', 'subclass': 'beta'},
    {'path': 'alpha', 'op': 'beta', 'subclass': 'gamma'},
])
def test_where_keyword_field_names_never_select_constructor_calls(profile, fields):
    model = Model('''<model name="keywords" package="org.example"><class name="Record">
        <attribute name="name" type="String"/><attribute name="path" type="String"/>
        <attribute name="op" type="String"/><attribute name="subclass" type="String"/>
        </class></model>''')
    query = Query(model, root='Record', compatibility=profile).where(name='initial')
    before = query.to_xml()
    refined = query.where(**fields)
    assert len(refined.coded_constraints) == len(fields) + 1
    assert refined.uncoded_constraints == []
    assert {con.path: (con.op, con.value) for con in refined.coded_constraints} == {
        'Record.name': ('=', 'initial'),
        **{'Record.' + path: ('=', value) for path, value in fields.items()},
    }
    assert query.to_xml() == before
    assert_bound_logic(refined)
    refined.verify()


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('method', ['add_constraint', 'where'])
def test_external_constraint_node_protocol_remains_supported(profile, method):
    query = Query(root='Employee', validate=False, compatibility=profile)
    node = SimpleNamespace(vargs=('age', '>', 30), kwargs={})
    result = getattr(query, method)(node)
    con = result if method == 'add_constraint' else result.get_constraint('A')
    assert con.path == 'Employee.age' and con.op == '>' and con.value == 30
    assert node.vargs == ('age', '>', 30) and node.kwargs == {}
    if method == 'where':
        assert query.constraints == []
        assert_bound_logic(result)
    before = query.to_xml()
    malformed = SimpleNamespace(vargs=('age', '>', 30), kwargs={'typo': True})
    with pytest.raises(TypeError, match='typo'):
        getattr(query, method)(malformed)
    assert query.to_xml() == before


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('form', ['keyword', 'tuple', 'positional', 'and-tree', 'empty'])
def test_flat_where_keeps_default_logic_dynamic_after_mutating_additions(profile, form):
    query = Query(root='Employee', validate=False, compatibility=profile)
    if form == 'empty':
        query.add_constraint('age', '=', 30)
    before = query.to_xml()
    calls = {
        'keyword': lambda: query.where(age=30),
        'tuple': lambda: query.where(('age', '=', 30)),
        'positional': lambda: query.where('age', '=', 30),
        'and-tree': lambda: query.where(
            ConstraintNode('age', '=', 30) & ConstraintNode('age', '<', 90)),
        'empty': lambda: query.where(),
    }
    refined = calls[form]()
    assert query.to_xml() == before
    added = refined.add_constraint('name', '=', 'Alice')
    assert added.code in refined.get_logic().get_codes()
    xml = ET.fromstring(refined.to_xml())
    assert set(refined.constraint_dict) == set(refined.get_logic().get_codes())
    assert added.code in xml.attrib['constraintLogic'].split()
    refined.validate_logic()
    assert refined._logic is None
    next_query = refined.where(fullTime=True)
    assert next_query._logic is None
    assert set(next_query.constraint_dict) == set(next_query.get_logic().get_codes())
    next_query.validate_logic()
    assert len(next_query.constraints) == len(refined.constraints) + 1
    assert_bound_logic(next_query)
