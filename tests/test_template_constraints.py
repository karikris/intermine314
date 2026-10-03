"""Template state over the real typed constraint/model/factory implementation."""
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

from intermine314 import constraints as public
from intermine314.model import Model, ModelError
from intermine314.query import Query
from intermine314.query import constraints as c
from intermine314.query.builder import ConstraintError
from tests.fixtures.compatibility import fixture_bytes

VARIANTS = [
    ('UnaryConstraint', ('Employee.name', 'IS NULL')),
    ('BinaryConstraint', ('Employee.name', '=', None)),
    ('ListConstraint', ('Employee', 'IN', 'staff')),
    ('LoopConstraint', ('Employee', 'IS', 'Employee.department.manager')),
    ('TernaryConstraint', ('Employee', 'LOOKUP', 'Susan', 'UK')),
    ('MultiConstraint', ('Employee.name', 'ONE OF', ['a', 'b'])),
    ('RangeConstraint', ('Employee.age', 'WITHIN', ['1..5'])),
    ('IsaConstraint', ('Employee', 'ISA', ['Manager'])),
    ('SubClassConstraint', ('Employee', 'Manager')),
]


@pytest.mark.parametrize('name,args', VARIANTS)
@pytest.mark.parametrize('editable', [True, False, 'true', 'false'])
def test_typed_variants_preserve_base_behavior_and_boolean_editability(name, args, editable):
    cls = getattr(c, 'Template' + name)
    assert getattr(public, 'Template' + name) is cls
    con = cls(*args, editable=editable, optional='on')
    ordinary = getattr(c, name)(*args)
    assert isinstance(con, getattr(c, name)) and isinstance(con, c.TemplateConstraint)
    assert con.to_dict() == ordinary.to_dict()
    assert con.editable is (editable is True or editable == 'true')
    suffix = '(editable, on)' if con.editable else '(non-editable, on)'
    expected = {
        'UnaryConstraint': 'Employee.name IS NULL',
        'BinaryConstraint': 'Employee.name = None',
        'ListConstraint': 'Employee IN staff',
        'LoopConstraint': 'Employee IS Employee.department.manager',
        'TernaryConstraint': 'Employee LOOKUP Susan IN UK',
        'MultiConstraint': "Employee.name ONE OF ['a', 'b']",
        'RangeConstraint': "Employee.age WITHIN ['1..5']",
        'IsaConstraint': "Employee ISA ['Manager']",
        'SubClassConstraint': 'Employee ISA Manager',
    }
    assert con.to_string() == expected[name] + ' ' + suffix
    assert repr(con) == f'<Template{name}: {con.to_string()}>'


@pytest.mark.parametrize('optional,required,on', [('locked', True, True), ('on', False, True), ('off', False, False)])
@pytest.mark.parametrize('editable', [True, False])
def test_mixin_status_switching_and_exact_errors(optional, required, on, editable):
    con = public.TemplateConstraint(editable=editable, optional=optional)
    assert con.required is required and con.optional is (not required)
    assert con.switched_on is on and con.switched_off is (not on)
    assert con.get_switchable_status() == optional
    if editable and not required:
        con.switch_off()
        assert con.switched_off and con.get_switchable_status() == 'off'
        con.switch_on()
        assert con.switched_on and con.get_switchable_status() == 'on'
    else:
        for switch in (con.switch_on, con.switch_off):
            with pytest.raises(ValueError, match='^This constraint is not switchable$'):
                switch()
        assert con.get_switchable_status() == optional


def test_defaults_separate_arguments_and_invalid_status():
    con = public.TemplateConstraint()
    assert (con.editable, con.required, con.switched_on) == (True, True, True)
    assert con.to_string() == '(editable, locked)'
    assert (con.REQUIRED, con.OPTIONAL_ON, con.OPTIONAL_OFF) == ('locked', 'on', 'off')
    args = {'path': 'Employee.name', 'value': 'x', 'editable': True, 'optional': 'off'}
    assert con.separate_arg_sets(args) == ({'path': 'Employee.name', 'value': 'x'}, {'editable': True, 'optional': 'off'})
    assert args['editable'] is True
    for value in ('invalid', None, True):
        with pytest.raises(TypeError, match='^Bad value for optional$'):
            public.TemplateConstraint(optional=value)


@pytest.mark.parametrize('name,args', VARIANTS)
@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_factory_dispatches_real_template_classes_and_preserves_profiles(name, args, profile):
    factory = public.TemplateConstraintFactory(compatibility=profile)
    # Native two-argument shorthand is equality; refinement uses its keyword.
    if name == 'SubClassConstraint':
        con = factory.make_constraint(path='Employee', subclass='Manager', editable=True, optional='off')
    else:
        con = factory.make_constraint(*args, editable=True, optional='off')
    assert type(con) is getattr(c, 'Template' + name)
    assert con.editable and con.switched_off and factory.compatibility == profile
    assert factory.CONSTRAINT_CLASSES == frozenset(getattr(c, 'Template' + n) for n, _ in VARIANTS)


def test_factory_bind_validation_prevents_uploads_and_reserves_real_codes():
    calls = []
    query = SimpleNamespace(service=SimpleNamespace(create_list=lambda q: calls.append(q) or SimpleNamespace(name='uploaded')))
    source = SimpleNamespace(to_query=lambda: query)
    factory = public.TemplateConstraintFactory(compatibility='legacy')
    for kwargs in ({'ignored': True}, {'optional': 'bad'}, {'code': 'bad'}):
        with pytest.raises(TypeError):
            factory.make_constraint('Employee', 'IN', source, **kwargs)
    assert calls == []
    con = factory.make_constraint('Employee', 'IN', source, code='A', editable='true')
    assert type(con) is c.TemplateListConstraint and con.code == 'A' and calls == [query]
    with pytest.raises(TypeError, match='already in use'):
        factory.make_constraint('Employee.name', '=', 'x', code='A')
    for _ in range(25):
        factory.make_constraint('Employee.name', '=', 'x')
    assert factory.make_constraint('Employee.name', '=', 'x').code == 'AA'


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_template_foundation_model_typed_validation_editable_subset_and_none_xml(profile):
    from intermine314.query import Template

    model = Model(fixture_bytes('model.xml'))
    template = Template(model, root='Employee', compatibility=profile)
    assert isinstance(template, Query) and template.model is model
    assert isinstance(template.constraint_factory, public.TemplateConstraintFactory)
    assert template.constraint_factory.compatibility == profile
    subclass = template.add_constraint('Employee', subclass='Manager', editable=True)
    assert type(subclass) is c.TemplateSubClassConstraint and not hasattr(subclass, 'code')
    assert template.get_subclass_dict() == {'Employee': 'Manager'}
    binary = template.add_constraint('name', '=', None, optional='off')
    hidden = template.add_constraint('age', '>', 21, editable=False)
    loop = template.add_constraint('Employee', 'IS', 'department.manager')
    assert loop.loopPath == 'Employee.department.manager'
    assert template.editable_constraints == [binary, loop, subclass]
    assert hidden not in template.editable_constraints
    binary.switch_on()
    assert binary.required is False
    wire = ET.fromstring(template.to_xml()).findall('constraint')
    assert next(n for n in wire if n.get('code') == binary.code).get('value') == ('None' if profile == 'legacy' else '')
    with pytest.raises(ConstraintError, match='attribute'):
        template.add_constraint('Employee', '=', 'x')
    with pytest.raises(ConstraintError, match='class'):
        template.add_constraint('Employee', 'ISA', ['Missing'])
    with pytest.raises(ModelError, match='missing'):
        template.add_constraint('missing', '=', 'x')
    with pytest.raises(ConstraintError, match='subclass'):
        template.add_constraint('department', subclass='Employee')


@pytest.mark.parametrize('profile,expected', [('native', 'TemplateMultiConstraint'), ('legacy', 'TemplateListConstraint')])
def test_native_collection_aliases_and_legacy_lists_are_not_confused(profile, expected):
    factory = public.TemplateConstraintFactory(compatibility=profile)
    con = factory.make_constraint('Employee', 'IN', ['a', 'b'], editable='false')
    assert type(con).__name__ == expected and not con.editable
    con = factory.make_constraint(path='Employee', op='IS', value='Employee.department.manager', optional='on')
    assert type(con) is c.TemplateLoopConstraint and con.code == 'B'
    con = factory.make_constraint('Employee.name', 'IS NULL', None, 'K')
    assert type(con) is c.TemplateUnaryConstraint and con.code == 'K'


def test_template_public_imports_remain_lazy_and_use_owned_modules():
    import subprocess
    import sys

    code = '''
import sys
import intermine314.query as query
assert 'intermine314.query.template' not in sys.modules
from intermine314.constraints import TemplateConstraintFactory
assert 'intermine314.query.template' not in sys.modules
Template = query.Template
assert Template.__module__ == 'intermine314.query.template'
assert TemplateConstraintFactory.__module__ == 'intermine314.query.constraints'
for name in ('intermine', 'polars', 'duckdb', 'pyarrow', 'pandas'):
    assert name not in sys.modules, name
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('profile', ['native', 'legacy'])
def test_template_model_inference_and_service_binding(profile, native_service_factory):
    from intermine314.query import Template

    model = Model(fixture_bytes('model.xml'))
    inferred = Template(model, root='Employee')
    assert inferred.compatibility == 'legacy' and inferred.root is model.get_class('Employee')
    assert inferred.editable_constraints == []
    service = native_service_factory(compatibility=profile)
    template = Template(model, service=service, root='Employee', compatibility=profile)
    assert template.service is service
    assert template.prefetch_depth == service.prefetch_depth
    assert template.prefetch_id_only == service.prefetch_id_only
    if profile == 'legacy':
        assert template.root is model.get_class('Employee')
    else:
        assert template.root == 'Employee'


def test_template_subclass_human_string_and_repr_use_upstream_isa():
    con = public.TemplateSubClassConstraint('Employee', 'Manager')
    assert con.to_string() == 'Employee ISA Manager (editable, locked)'
    assert repr(con) == '<TemplateSubClassConstraint: Employee ISA Manager (editable, locked)>'
    con = public.TemplateSubClassConstraint('Employee', 'Manager', editable=False, optional='off')
    assert con.to_string() == 'Employee ISA Manager (non-editable, off)'
    assert repr(con) == '<TemplateSubClassConstraint: Employee ISA Manager (non-editable, off)>'
    assert c.SubClassConstraint('Employee', 'Manager').to_string() == 'Employee Manager'
