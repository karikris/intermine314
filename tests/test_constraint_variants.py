"""Pinned 1.13 constraint calls, model semantics, and shared wire contracts."""
from types import SimpleNamespace
from urllib.error import HTTPError
from xml.etree import ElementTree as ET

import pytest

from intermine314 import constraints as facade
from intermine314.model import Model, ModelError
from intermine314.query import Query
from intermine314.query import constraints as c
from intermine314.query.builder import ConstraintError
from intermine314.query.spec import QuerySpec, query_spec_to_xml
from tests.fixtures.compatibility import fixture_bytes


@pytest.mark.parametrize('name,args,expected', [
    ('BinaryConstraint', ('Employee.name', '=', 12), 'Employee.name = 12'),
    ('MultiConstraint', ('Employee.name', 'ONE OF', ['a', 'b']), "Employee.name ONE OF ['a', 'b']"),
    ('SubClassConstraint', ('Employee', 'Manager'), 'Employee ISA Manager'),
    ('ListConstraint', ('Employee', 'IN', 'staff'), 'Employee IN staff'),
    ('LoopConstraint', ('Employee', 'IS NOT', 'Employee.department.manager'), 'Employee IS NOT Employee.department.manager'),
    ('TernaryConstraint', ('Employee', 'LOOKUP', 'Fred', 'UK'), 'Employee LOOKUP Fred IN UK'),
    ('RangeConstraint', ('Range', 'OVERLAPS', ['1..5']), "Range OVERLAPS ['1..5']"),
    ('IsaConstraint', ('Employee', 'ISA', ['Manager']), "Employee ISA ['Manager']"),
])
def test_public_variants_human_strings_and_facade_identity(name, args, expected):
    cls = getattr(c, name)
    assert getattr(facade, name) is cls
    con = cls(*args)
    assert con.to_string() == expected
    assert repr(con) == f'<{name}: {expected}>'


@pytest.mark.parametrize('args,kwargs,cls,extra', [
    (('Employee', 'IN', 'staff'), {}, 'ListConstraint', {'value': 'staff'}),
    (('Employee',), {'op': 'NOT IN', 'list_name': 'staff'}, 'ListConstraint', {'value': 'staff'}),
    ((), {'path': 'Employee', 'op': 'IS', 'loopPath': 'Employee.department.manager'}, 'LoopConstraint', {'op': '=', 'loopPath': 'Employee.department.manager'}),
    (('Employee', 'LOOKUP'), {'value': 'Fred', 'extra_value': 'UK', 'code': 'K'}, 'TernaryConstraint', {'extraValue': 'UK'}),
    (('Range', 'WITHIN'), {'values': ('1..5',)}, 'RangeConstraint', {'value': ['1..5']}),
    (('Employee',), {'op': 'ISA', 'values': ['Manager']}, 'IsaConstraint', {'value': ['Manager']}),
    (('Employee', 'IS NULL'), {}, 'UnaryConstraint', {'op': 'IS NULL'}),
    (('Employee', 'IS NOT NULL', 'K'), {}, 'UnaryConstraint', {'code': 'K'}),
    (('Employee', 'Manager'), {}, 'SubClassConstraint', {'type': 'Manager'}),
    (('Employee',), {'subclass': 'Manager'}, 'SubClassConstraint', {'type': 'Manager'}),
])
def test_legacy_factory_positional_keyword_and_mixed_dispatch(args, kwargs, cls, extra):
    con = c.ConstraintFactory(compatibility='legacy').make_constraint(*args, **kwargs)
    assert isinstance(con, getattr(c, cls))
    assert con.to_dict().items() >= extra.items()


def test_factory_profiles_aliases_contains_and_code_reservations():
    factory = c.ConstraintFactory()
    assert facade.ConstraintFactory is c.ConstraintFactory
    assert factory.make_constraint('Employee.name', 'IN', [1, 2]).values == ['1', '2']
    assert isinstance(factory.make_constraint('Employee', 'IN', 'staff'), c.ListConstraint)
    assert isinstance(factory.make_constraint('Employee', 'NOT IN', SimpleNamespace(name='staff')), c.ListConstraint)
    assert factory.make_constraint('Employee.name', [1, 2]).op == 'ONE OF'
    assert factory.make_constraint('Employee.name', 'x').value == 'x'
    assert isinstance(factory.make_constraint('Employee.name', 'CONTAINS', 'x'), c.BinaryConstraint)
    assert isinstance(factory.make_constraint('Range', 'CONTAINS', ['1..5']), c.RangeConstraint)
    factory = c.ConstraintFactory()
    assert factory.make_constraint('Employee.name', '=', 'x', code='A').code == 'A'
    assert factory.make_constraint('Employee.name', '=', 'x').code == 'B'
    with pytest.raises(TypeError, match='code'):
        factory.make_constraint('Employee.name', '=', 'x', code='A')
    for _ in range(24):
        factory.make_constraint('Employee.name', '=', 'x')
    assert factory.make_constraint('Employee.name', '=', 'x').code == 'AA'


def test_list_protocol_upload_once_and_errors_are_not_swallowed():
    calls = []
    query = SimpleNamespace(service=SimpleNamespace(create_list=lambda q: calls.append(q) or SimpleNamespace(name='uploaded')))
    source = SimpleNamespace(to_query=lambda: query)
    factory = c.ConstraintFactory(compatibility='legacy')
    with pytest.raises(TypeError):
        factory.make_constraint('Employee', 'IN', source, ignored=True)
    assert calls == []
    con = factory.make_constraint('Employee', 'IN', source)
    assert calls == [query] and con.list_name == 'uploaded' and con.code == 'A'
    assert factory.make_constraint('Employee', 'IN', SimpleNamespace(name='existing')).list_name == 'existing'
    def fail(q):
        calls.append(q)
        raise TypeError('upload failed')
    query.service.create_list = fail
    with pytest.raises(TypeError, match='upload failed'):
        factory.make_constraint('Employee', 'IN', source)
    assert calls == [query, query]
    assert factory.make_constraint('Employee', 'IN', 'ok').code == 'C'


def test_list_upload_http_error_propagates_and_invalid_op_never_uploads():
    calls = []
    failure = HTTPError('https://offline.example/list', 403, 'denied', {}, None)
    def upload(query):
        calls.append(query)
        raise failure
    query = SimpleNamespace(service=SimpleNamespace(create_list=upload))
    source = SimpleNamespace(to_query=lambda: query)
    with pytest.raises(TypeError):
        c.ListConstraint('Employee', 'UNKNOWN', source)
    assert calls == []
    with pytest.raises(HTTPError) as raised:
        c.ConstraintFactory().make_constraint('Employee', 'IN', source)
    assert raised.value is failure and calls == [query]
    failure.close()


@pytest.mark.parametrize('op', sorted(c.RangeConstraint.OPS))
def test_every_range_operator_serializes_child_values(op):
    query = Query(root='Range')
    con = query.add_constraint('Range', op, ('1..5', '10..20'))
    assert isinstance(con, c.RangeConstraint)
    node = ET.fromstring(query.to_xml()).find('constraint')
    assert node.attrib['op'] == op
    assert [v.text for v in node] == ['1..5', '10..20']


def test_lookup_omitted_extra_and_unary_empty_placeholder():
    query = Query(root='Employee')
    query.add_constraint('Employee', 'LOOKUP', 'Fred')
    query.add_constraint('name', 'IS NULL', None)
    query.add_constraint('age', 'IS NOT NULL', None, 'Q')
    nodes = ET.fromstring(query.to_xml()).findall('constraint')
    assert 'extraValue' not in nodes[0].attrib
    assert 'value' not in nodes[1].attrib
    assert query.get_constraint('Q').op == 'IS NOT NULL'


def test_external_codes_and_clone_reservations_do_not_overwrite():
    query = Query(root='Employee')
    query.add_constraint(c.BinaryConstraint('name', '=', 'external', code='A'))
    assert query.add_constraint('name', '=', 'generated').code == 'B'
    with pytest.raises(ConstraintError, match='code'):
        query.add_constraint(c.BinaryConstraint('name', '=', 'duplicate', code='A'))
    clone = query.clone()
    assert clone.add_constraint('name', '=', 'clone').code == 'C'
    assert query.add_constraint('name', '=', 'original').code == 'C'
    assert clone.get_constraint('A') is not query.get_constraint('A')


@pytest.mark.parametrize('args,kwargs', [
    (('Employee.name', '=', 'x'), {'ignored': True}),
    (('Employee.name', '=', 'x'), {'value': 'duplicate'}),
    (('Employee.name',), {'op': 'ONE OF'}),
    (('Employee.name', '=', 'x'), {'extra_value': 'wrong'}),
    (('Employee', 'IS NULL', 'A', 'unused'), {}),
])
def test_factory_rejects_invalid_calls_without_consuming_codes(args, kwargs):
    factory = c.ConstraintFactory()
    with pytest.raises(TypeError):
        factory.make_constraint(*args, **kwargs)
    assert factory.make_constraint('Employee.name', '=', 'x').code == 'A'


@pytest.mark.parametrize('value', ['IN', 'NOT IN', '=', 'CONTAINS', 'LOOKUP', 'ISA', 'ONE OF', 'IS'])
def test_native_two_argument_scalar_shorthand_keeps_operator_looking_values(value):
    factory = c.ConstraintFactory()
    con = factory.make_constraint('Employee.name', value)
    assert isinstance(con, c.BinaryConstraint)
    assert con.op == '=' and con.value == value
    query = Query(Model(fixture_bytes('model.xml')), root='Employee', compatibility='native')
    con = query.add_constraint('name', value)
    assert con.op == '=' and con.value == value and con.path == 'Employee.name'


@pytest.mark.parametrize('args,kwargs,cls', [
    (('LOOKUP', 'Susan'), {}, 'TernaryConstraint'),
    (('LOOKUP', 'Susan', 'UK'), {}, 'TernaryConstraint'),
    (('LOOKUP',), {'value': 'Susan', 'extra_value': 'UK'}, 'TernaryConstraint'),
    (('IN', 'staff'), {}, 'ListConstraint'),
    (('NOT IN',), {'list_name': 'staff'}, 'ListConstraint'),
    (('ISA', ['Manager']), {}, 'IsaConstraint'),
    (('ISA',), {'values': ['Manager']}, 'IsaConstraint'),
    (('OVERLAPS', ['1..5']), {}, 'RangeConstraint'),
    (('WITHIN',), {'values': ['1..5']}, 'RangeConstraint'),
    (('CONTAINS', ['1..5']), {}, 'RangeConstraint'),
])
@pytest.mark.parametrize('root_from_view', [False, True])
def test_legacy_reference_operators_accept_implied_query_root(args, kwargs, cls, root_from_view):
    query = Query(Model(fixture_bytes('model.xml')), root=None if root_from_view else 'Employee')
    if root_from_view:
        query.add_view('Employee.age')
    con = query.add_constraint(*args, **kwargs)
    assert isinstance(con, getattr(c, cls))
    assert con.path == 'Employee' and con.code == 'A'
    assert ET.fromstring(query.to_xml()).find('constraint').attrib['path'] == 'Employee'


def test_native_query_keeps_first_operator_looking_path_as_scalar_shorthand():
    query = Query(root='Employee')
    con = query.add_constraint('LOOKUP', 'Susan')
    assert con.path == 'Employee.LOOKUP' and con.op == '=' and con.value == 'Susan'


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_query_profiles_clone_and_convenience_collection_membership(profile):
    query = Query(Model(fixture_bytes('model.xml')), root='Employee', compatibility=profile)
    if profile == 'legacy':
        con = query.add_constraint('Employee', 'IN', 'staff')
        assert isinstance(con, c.ListConstraint)
    else:
        con = query.add_constraint('name', 'IN', ['Fred'])
        assert isinstance(con, c.MultiConstraint)
    clone = query.where_in('name', ['Alice'])
    assert clone.constraint_factory.compatibility == profile
    assert clone.get_constraint('B').op == 'ONE OF'
    assert len(query.constraints) == 1
    assert query.where(name=['Alice']).get_constraint('B').op == 'ONE OF'


def test_all_variants_xml_and_shared_executor_payload(native_service_factory):
    service = native_service_factory(compatibility='legacy')
    query = service.select('Employee.name')
    query.add_constraint('Employee', 'IN', 'staff & alumni')
    query.add_constraint('Employee', 'IS NOT', 'department.manager')
    query.add_constraint('Employee', 'LOOKUP', 'Fred <Jones>', 'UK & Ireland')
    query.add_constraint('age', 'OVERLAPS', ['1..5', '20..30'])
    query.add_constraint('Employee', 'ISA', ['Manager', 'CEO'])
    query.add_constraint('Employee', subclass='Manager')
    xml = query.to_xml()
    assert query.to_query_params()['query'] == query_spec_to_xml(query.to_spec()) == xml
    assert service.execute(query.to_spec()).to_query_params()['query'] == xml
    root = ET.fromstring(xml)
    nodes = root.findall('constraint')
    assert nodes[0].attrib == {'path': 'Employee', 'op': 'IN', 'code': 'A', 'value': 'staff & alumni'}
    assert nodes[1].attrib == {'path': 'Employee', 'op': '!=', 'code': 'B', 'loopPath': 'Employee.department.manager'}
    assert nodes[2].attrib['extraValue'] == 'UK & Ireland'
    assert nodes[2].attrib['value'] == 'Fred <Jones>'
    assert [v.text for v in nodes[3]] == ['1..5', '20..30']
    assert [v.text for v in nodes[4]] == ['Manager', 'CEO']
    assert nodes[5].attrib == {'path': 'Employee', 'type': 'Manager'}
    assert root.attrib['constraintLogic'] == 'A and B and C and D and E'
    assert ET.tostring(ET.fromstring(query.to_formatted_xml())).replace(b'\n', b'')
    assert query.clone().to_xml() == xml


@pytest.mark.parametrize('profile,expected', [('native', ''), ('legacy', 'None')])
def test_binary_none_xml_normalization_is_shared_and_profile_specific(profile, expected, native_service_factory):
    con = c.BinaryConstraint('Employee.name', '=', None)
    assert con.to_dict()['value'] == 'None'
    spec = QuerySpec(root_class='Employee', constraints=(con,), compatibility=profile)
    assert ET.fromstring(query_spec_to_xml(spec)).find('constraint').attrib['value'] == expected

    query = Query(root='Employee', compatibility=profile)
    query.add_constraint('name', '=', None)
    for xml in (query.to_xml(), query.to_formatted_xml(), query.to_query_params()['query'], query_spec_to_xml(query.to_spec())):
        assert ET.fromstring(xml).find('constraint').attrib['value'] == expected
    element = ET.Element('query')
    query._append_constraint_xml(element, query.get_constraint('A'))
    assert element.find('constraint').attrib['value'] == expected
    assert query._build_query_xml_element().find('constraint').attrib['value'] == expected
    executor = native_service_factory(compatibility=profile).execute(query.to_spec())
    assert ET.fromstring(executor.to_query_params()['query']).find('constraint').attrib['value'] == expected


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_new_lookup_none_value_retains_source_stringification(profile):
    query = Query(root='Employee', compatibility=profile)
    con = query.add_constraint('Employee', 'LOOKUP', None)
    assert con.to_dict()['value'] == 'None'
    assert ET.fromstring(query.to_xml()).find('constraint').attrib['value'] == 'None'


@pytest.mark.parametrize('args,error,match', [
    (('Employee', '=', 'x'), ConstraintError, 'attribute'),
    (('Employee', 'ONE OF', ['x']), ConstraintError, 'attribute'),
    (('name', 'LOOKUP', 'x'), ConstraintError, 'object'),
    (('name', 'IN', 'staff'), ConstraintError, 'object'),
    (('name', 'ISA', ['Manager']), ConstraintError, 'object'),
    (('Employee', 'ISA', ['Missing']), ConstraintError, 'class'),
    (('Employee', 'IS', 'department'), ConstraintError, 'incompatible'),
    (('Employee', 'IS', 'name'), ConstraintError, 'object'),
    (('name', 'IS', 'Employee'), ConstraintError, 'object'),
    (('Employee', 'IS', 'missing'), ModelError, 'missing'),
    (('name', 'Manager'), ConstraintError, 'subclass'),
    (('Employee', 'Department'), ConstraintError, 'subclass'),
    (('missing', 'OVERLAPS', ['1..5']), ModelError, 'missing'),
])
def test_model_semantic_validation(args, error, match):
    query = Query(Model(fixture_bytes('model.xml')), root='Employee')
    with pytest.raises(error, match=match):
        query.add_constraint(*args)


def test_valid_object_variants_subclass_revalidation_and_ranges():
    query = Query(Model(fixture_bytes('model.xml')), root='Employee')
    query.add_constraint('Employee', 'Manager')
    query.add_constraint('Employee', 'IS', 'department.manager')
    query.add_constraint('department', 'LOOKUP', 'Sales')
    query.add_constraint('Employee', 'ISA', ['Employee', 'Manager'])
    query.add_constraint('department', 'IN', 'departments')
    query.add_constraint('age', 'WITHIN', ['1..5'])
    query.add_constraint('department', 'OUTSIDE', ['1..5'])
    query.add_constraint('department', 'IS NULL')
    query.verify_constraint_paths()
    assert query.get_subclass_dict() == {'Employee': 'Manager'}


@pytest.mark.parametrize('model', [None, Model(fixture_bytes('model.xml'))], ids=['no-model', 'model'])
def test_syntax_fallback_and_validate_false(model):
    query = Query(model, root='Employee', compatibility='legacy', validate=False)
    query.add_constraint('missing', 'LOOKUP', 'x')
    query.add_constraint('missing', 'IS', 'alsoMissing')
    assert query.get_constraint('B').loopPath == 'Employee.alsoMissing'
    if model is None:
        query.verify_constraint_paths()
    with pytest.raises(TypeError):
        query.add_constraint('bad-path', 'IN', 'staff')
