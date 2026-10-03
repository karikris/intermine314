"""Named Template calls through the shared managed wire and export runtime."""
from dataclasses import FrozenInstanceError
from io import BytesIO, StringIO
from urllib.parse import parse_qs
from xml.etree import ElementTree as ET

import pytest

from intermine314.model import Model
from intermine314.query import (
    ConstraintError,
    ParallelOptions,
    QueryParseError,
    Template,
)
from intermine314.results import ResultObject, ResultRow, encode_dict, encode_str
from intermine314.service import Service
from intermine314.webservice import Service as LegacyService
from tests.fixtures.compatibility import (
    FixtureConnection,
    FixtureSession,
    fixture_bytes,
)
from tests.test_query_eager_results import payload

ROOT = 'https://offline.example/service'
XML = '''<template name="Names" userName="Müller" title="Find names" dataTypes="Employee Manager" comment="saved">
<query name="Names" view="Employee.name" constraintLogic="A and B and C and D">
<constraint path="Employee.name" op="ONE OF" code="A" editable="true"><value>old</value><value>other</value></constraint>
<constraint path="Employee.name" op="!=" value="hidden" code="B" editable="false"/>
<constraint path="Employee.name" op="!=" value="off" code="C" editable="true" switchable="off"/>
<constraint path="Employee.name" op="!=" value="on" code="D" editable="true" switchable="on"/>
</query></template>'''
VALUES = ['Å & b', 'two+three']


class TemplateSession(FixtureSession):
    def __init__(self, rows=None, count=1):
        super().__init__(FixtureSession.service().routes)
        self.rows = rows if rows is not None else payload(['Ada'])
        self.count = count

    def request(self, method, url, data=None, **kwargs):
        if method == 'POST' and url.endswith('/template/results'):
            params = parse_qs(data.decode(), keep_blank_values=True)
            body = str(self.count).encode() if params['format'] == ['count'] else self.rows
            self.routes[method, '/service/template/results'] = body
        return super().request(method, url, data=data, **kwargs)


def posts(session):
    return [(r.path, parse_qs(r.data.decode() if isinstance(r.data, bytes) else r.data, keep_blank_values=True))
            for r in session.requests if r.method == 'POST']


def template(service):
    return Template.from_xml(XML, service.model, service=service, compatibility=service.compatibility)


def assert_wire(session, values=VALUES):
    for path, params in posts(session):
        assert path == '/service/template/results'
        assert params['name'] == ['Names'] and params['userName'] == ['Müller']
        assert params['value1'] == values and params['code1'] == ['A']
        assert params['constraint1'] == params['constraint2'] == ['Employee.name']
        assert params['code2'] == ['D'] and params['value2'] == ['on']
        assert 'constraint3' not in params and 'query' not in params and 'path' not in params
    assert posts(session)
    assert all(r.closed for r in session.responses)
    assert session.close_calls == 0


@pytest.mark.parametrize('source_kind', ['text', 'bytes', 'file', 'borrowed-text', 'borrowed-bytes'])
def test_xml_once_metadata_flags_clone_and_profile(source_kind, tmp_path):
    model = Model(fixture_bytes('model.xml'))
    path = tmp_path / 'template.txt'
    path.write_text(XML)
    source = {'text': XML, 'bytes': XML.encode(), 'file': path,
              'borrowed-text': StringIO(XML), 'borrowed-bytes': BytesIO(XML.encode())}[source_kind]
    t = Template.from_xml(source, model)
    assert (t.name, t.title, t.user_name, t.view_types, t.comment) == ('Names', 'Find names', 'Müller', ['Employee', 'Manager'], 'saved')
    assert t.model is model and t.root is model.get_class('Employee') and t.compatibility == 'legacy'
    assert [c.code for c in t.editable_constraints] == ['A', 'C', 'D']
    assert t.get_constraint('C').switched_off
    clone = t.clone()
    clone.view_types.append('Company')
    clone.get_constraint('A').values.append('new')
    clone.get_constraint('C').switch_on()
    clone.add_constraint('Employee.age', '>', 20)
    assert t.view_types == ['Employee', 'Manager'] and t.get_constraint('A').values == ['old', 'other']
    assert t.get_constraint('C').switched_off and 'E' not in t.constraint_dict
    assert clone.model is t.model and clone.service is t.service
    assert str(clone.get_logic()) == str(t.get_logic())
    assert clone.add_user_name('other') is None and t.user_name == 'Müller'
    assert clone.user_name == 'other' and clone.to_query_params()['userName'] == 'other'
    for xml in (t.to_xml(), t.to_formatted_xml(), t.to_Node().toxml()):
        wrapper = ET.fromstring(xml)
        assert wrapper.tag == 'template' and wrapper.get('userName') == 'Müller'
        assert wrapper.find('query/constraint[@code="B"]').get('editable') == 'false'
        assert wrapper.find('query/constraint[@code="C"]').get('switchable') == 'off'
        assert wrapper.find('query/constraint[@code="A"]').get('switchable') is None
        restored = Template.from_xml(xml, model)
        assert restored.to_query_params() == t.to_query_params()
    if hasattr(source, 'read'):
        assert not source.closed


@pytest.mark.parametrize('bad', [False, True])
def test_owned_xml_closes_once_and_borrowed_failure_stays_open(monkeypatch, bad):
    from intermine314.query import builder
    connections = []
    def opened(source):
        con = FixtureConnection(b'<template>' if bad else XML.encode())
        connections.append(con)
        return con
    monkeypatch.setattr(builder, 'openAnything', opened)
    if bad:
        with pytest.raises(QueryParseError):
            Template.from_xml('saved.txt')
    else:
        assert Template.from_xml('saved.txt').title == 'Find names'
    assert len(connections) == 1 and connections[0].close_calls == 1


def test_one_wrapper_required_and_borrowed_parse_failure():
    for xml in ('<query/>', '<root><template/><template/><query/></root>'):
        with pytest.raises(QueryParseError, match='Only one <template>'):
            Template.from_xml(xml)
    source = StringIO('<template>')
    with pytest.raises(QueryParseError):
        Template.from_xml(source)
    assert not source.closed


@pytest.mark.parametrize('client', [Service, LegacyService])
@pytest.mark.parametrize('helper', ['results', 'rows', 'get_results_list', 'get_row_list', 'all', 'first', 'one', 'count', 'size', 'iter_rows', 'iter_batches', 'run_parallel'])
def test_adjusted_helpers_actual_wire_defaults_and_immutable_caller(client, helper):
    session = TemplateSession()
    with client(ROOT, session=session) as service:
        t = template(service)
        before = t.to_xml()
        kwargs = {'A': {'values': VALUES}}
        if helper in ('results', 'get_results_list', 'all', 'first', 'one'):
            kwargs['row'] = 'dict'
        if helper == 'run_parallel':
            kwargs['parallel_options'] = ParallelOptions(page_size=1, max_workers=2)
        value = getattr(t, helper)(**kwargs)
        if helper in ('count', 'size'):
            assert value == 1
        elif helper in ('first', 'one'):
            assert value == {'Employee.name': 'Ada'}
        else:
            assert len(list(value)) == 1
        assert t.to_xml() == before
        assert_wire(session)


@pytest.mark.parametrize('client,expected', [(Service, dict), (LegacyService, ResultObject)])
def test_default_results_and_rows(client, expected):
    session = TemplateSession(payload({'name': 'Ada'}) if client is LegacyService else payload(['Ada']))
    with client(ROOT, session=session) as service:
        t = template(service)
        assert isinstance(t.get_results_list(A={'values': VALUES})[0], expected)
        session.rows = payload(['Ada'])
        assert isinstance(t.get_row_list(A={'values': VALUES})[0], ResultRow if client is LegacyService else dict)
        assert_wire(session)


@pytest.mark.parametrize('adjustment,error', [({'Z': 'x'}, ConstraintError), ({'B': 'x'}, ConstraintError),
    ({'A': {'oops': 1}}, TypeError), ({'A': {'op': '='}}, TypeError),
    ({'A': {'values': 'bad'}}, TypeError), ({'D': {'value': ['bad']}}, TypeError)])
def test_invalid_adjustments_before_http_and_caller_preserved(adjustment, error):
    session = TemplateSession()
    with Service(ROOT, session=session) as service:
        t = template(service)
        before = t.to_xml()
        with pytest.raises(error):
            t.results(**adjustment)
        assert t.to_xml() == before and posts(session) == []


def test_editable_codeless_refinement_preserved_but_invalid_wire_rejected():
    session = TemplateSession()
    with Service(ROOT, session=session) as service:
        t = template(service)
        c = t.add_constraint('Employee', subclass='Manager', editable=True)
        assert c in t.editable_constraints and not hasattr(c, 'code')
        assert ET.fromstring(t.to_xml()).find('query/constraint[@type="Manager"]') is not None
        with pytest.raises(ConstraintError, match='editable=False'):
            t.count()
        assert posts(session) == []
        c.editable = False
        assert t.count(A={'values': VALUES}) == 1
        assert_wire(session)


def test_shared_encoder_repeated_sequences_preserves_native_scalars():
    from urllib.parse import urlencode
    params = {'a': ['Å', 'a+b'], 'b': ('x', 'y'), 'empty': [], 'n': None, 'yes': True, 'int': 3}
    assert parse_qs(urlencode(encode_dict(params), True)) == {'a': ['Å', 'a+b'], 'b': ['x', 'y'], 'n': ['None'], 'yes': ['True'], 'int': ['3']}
    assert encode_str(3) == '3' and encode_str(None) == 'None'


def test_executor_metadata_snapshot_and_summary():
    session = TemplateSession(payload({'item': 'Ada', 'count': 1}))
    with Service(ROOT, session=session) as service:
        t = template(service)
        adjusted = t.get_adjusted_template({'A': {'values': VALUES}})
        spec = adjusted.to_spec()
        with pytest.raises(FrozenInstanceError):
            spec.template.user_name = 'wrong'
        assert service.execute(spec).count() == 1
        assert t.summarise('name', A={'values': VALUES}) == {'Ada': 1}
        assert posts(session)[-1][1]['summaryPath'] == ['Employee.name']
        assert_wire(session)


@pytest.mark.parametrize('args', [('Employee', 'Manager'), ('Employee', 'IS', 'Employee.department.manager'),
    ('Employee.age', 'WITHIN', ['1..5']), ('Employee', 'ISA', ['Manager'])])
def test_unsupported_active_editable_forms_fail_before_wire_but_allow_fixed_or_off(args):
    session = TemplateSession()
    with LegacyService(ROOT, session=session) as service:
        t = template(service)
        con = t.add_constraint(*args, optional='on')
        with pytest.raises(ConstraintError, match='editable=False'):
            t.results()
        assert not posts(session)
        con.switch_off()
        assert t.count(A={'values': VALUES}) == 1
        con.switch_on()
        con.editable = False
        assert t.count(A={'values': VALUES}) == 1
        assert_wire(session)


def test_empty_multivalue_rejected_but_empty_scalar_and_single_empty_value_supported():
    session = TemplateSession()
    with Service(ROOT, session=session) as service:
        t = template(service)
        with pytest.raises(ConstraintError, match='at least one'):
            t.count(A={'values': []})
        assert not posts(session)
        assert t.count(A={'values': ['']}, D='') == 1
        assert posts(session)[-1][1]['value1'] == [''] and posts(session)[-1][1]['value2'] == ['']


@pytest.mark.parametrize('client', [Service, LegacyService])
@pytest.mark.parametrize('helper', ['dataframe', 'to_parquet', 'export', 'to_duckdb'])
@pytest.mark.parametrize('empty', [False, True])
def test_adjusted_analytics_exact_decimal_typed_schema_and_wire(client, helper, empty, tmp_path):
    from decimal import Decimal

    import polars as pl

    from intermine314.export import query_parquet
    model_xml = b'<model name="export" package="test"><class name="Record"><attribute name="name" type="String"/><attribute name="amount" type="java.math.BigDecimal"/></class></model>'
    session = TemplateSession(b'{"results":[\n["Ada",1234567890.123456789]\n],"wasSuccessful":true}\n' if not empty else payload())
    session.routes['GET', '/service/model'] = model_xml
    with client(ROOT, session=session) as service:
        t = Template(service.model, service=service, root='Record', compatibility=service.compatibility)
        t.name = 'Amounts'
        t.add_view('Record.name', 'Record.amount')
        t.add_constraint('name', 'ONE OF', ['old'])
        before = t.to_xml()
        destination = tmp_path / 'result.parquet'
        kwargs = {'A': {'values': VALUES}}
        if helper == 'dataframe':
            frame = t.dataframe(**kwargs)
        elif helper == 'to_duckdb':
            with t.to_duckdb(destination, single_file=True, managed=True, **kwargs) as con:
                frame = con.execute('select * from results').pl()
        else:
            getattr(t, helper)(destination, single_file=True, **kwargs)
            frame = query_parquet(destination)
        assert frame.schema == {'Record.name': pl.String, 'Record.amount': pl.Decimal(38, 0 if empty else 9)}
        assert frame.rows() == ([] if empty else [('Ada', Decimal('1234567890.123456789'))])
        assert t.to_xml() == before
        for path, form in posts(session):
            assert path == '/service/template/results' and form['name'] == ['Amounts']
            assert form['value1'] == VALUES and form['code1'] == ['A'] and 'query' not in form
        assert posts(session) and all(r.closed for r in session.responses)


@pytest.mark.parametrize('pagination', ['auto', 'offset'])
def test_parallel_count_and_pages_share_adjustments(pagination):
    session = TemplateSession(count=3)
    with Service(ROOT, session=session) as service:
        t = template(service)
        rows = list(t.run_parallel(A={'values': VALUES}, parallel_options=ParallelOptions(
            pagination=pagination, page_size=1, max_workers=2)))
        assert len(rows) == 3
        assert sorted(int(form['start'][0]) for _, form in posts(session) if form['format'] != ['count']) == [0, 1, 2]
        assert_wire(session)
        with pytest.raises(ValueError, match='pagination'):
            t.run_parallel(parallel_options=ParallelOptions(pagination='keyset'))


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('view', [None, 'name', 'department.name'])
def test_template_list_creation_append_operations_and_in_actual_wire(monkeypatch, profile, view):
    import json
    from urllib.parse import urlsplit

    from tests.test_lists_operations import server
    service, session, inventory, members = server(monkeypatch, profile)
    original = session.request
    def request(method, url, **kwargs):
        path = urlsplit(url).path
        if method != 'POST' or path not in ('/service/template/tolist', '/service/template/append/tolist'):
            return original(method, url, **kwargs)
        form = parse_qs(kwargs['data'].decode(), keep_blank_values=True)
        assert form['name'] == ['Names'] and form['value1'] == VALUES and form['code1'] == ['A']
        assert form['code2'] == ['D'] and 'constraint3' not in form and 'query' not in form
        expected = 'Employee.department.id' if view == 'department.name' else 'Employee.id'
        assert form['path'] == [expected]
        name = form['listName'][0]
        members[name] = (members.get(name, set()) if 'append' in path else set()) | {3, 4}
        inventory[name] = dict(inventory['left'], name=name, size=len(members[name]))
        session.routes['GET', '/service/lists'] = json.dumps({'wasSuccessful': True, 'lists': list(inventory.values())}).encode()
        session.routes[method, path] = json.dumps({'wasSuccessful': True, 'listName': name}).encode()
        return FixtureSession.request(session, method, url, **kwargs)
    monkeypatch.setattr(session, 'request', request)
    t = template(service).get_adjusted_template({'A': {'values': VALUES}})
    t.clear_view()
    if view:
        t.add_view(view)
    before = t.to_xml()
    assert t.to_query() is t
    item = service.create_list(t, name='chosen')
    assert members[item.name] == {3, 4}
    left = service.get_list('left')
    assert left.append(t) is left and members['left'] == {1, 2, 3, 4}
    union = t | service.get_list('right')
    assert members[union.name] == {3, 4}
    q = service.new_query('Employee')
    con = q.add_constraint('Employee', 'IN', t)
    assert members[con.list_name] == {3, 4}
    assert t.to_xml() == before
    assert all(r.closed for r in session.responses)
    assert {path for path, _ in posts(session)} == {'/service/template/tolist', '/service/template/append/tolist'}


def test_lookup_unary_list_scalar_adjustments_and_snapshot():
    t = Template(root='Employee')
    t.name = 'Lookup'
    t.add_view('Employee.name')
    t.add_constraint('Employee', 'LOOKUP', 'old', 'UK')
    t.add_constraint('Employee.name', 'IS NULL')
    t.add_constraint('Employee', 'IN', 'old-list')
    adjusted = t.get_adjusted_template({'A': {'value': 'Ada', 'extra_value': 'US'}, 'B': {'op': 'IS NOT NULL'}, 'C': 'new-list'})
    assert adjusted.to_query_params() == {'name': 'Lookup', 'userName': '', 'constraint1': 'Employee',
        'op1': 'LOOKUP', 'code1': 'A', 'value1': 'Ada', 'extra1': 'US', 'constraint2': 'Employee.name',
        'op2': 'IS NOT NULL', 'code2': 'B', 'constraint3': 'Employee', 'op3': 'IN', 'code3': 'C', 'value3': 'new-list'}
    assert t.get_constraint('A').value == 'old' and t.get_constraint('C').list_name == 'old-list'
    spec = adjusted.to_spec()
    adjusted.get_constraint('A').value = 'mutated'
    adjusted.view_types.append('other')
    assert spec.constraints[0].value == 'Ada' and spec.template.view_types == ()


def test_eager_summary_option_is_not_interpreted_as_constraint_code():
    session = TemplateSession(payload({'item': 'Ada', 'count': 1}))
    with Service(ROOT, session=session) as service:
        assert template(service).get_results_list(summary_path='name', A={'values': VALUES}) == [{'item': 'Ada', 'count': 1}]
        assert posts(session)[-1][1]['summaryPath'] == ['Employee.name']
        assert_wire(session)


def test_set_operation_rejects_unsupported_template_before_uploading_any_operand(monkeypatch):
    from tests.test_lists_operations import server
    service, session, _, _ = server(monkeypatch)
    valid = service.select('Employee.name')
    invalid = template(service)
    invalid.add_constraint('Employee', subclass='Manager')
    with pytest.raises(ConstraintError, match='editable=False'):
        service._get_list_manager().union([valid, invalid])
    assert not posts(session)


@pytest.mark.parametrize('profile', ['native', 'legacy'])
@pytest.mark.parametrize('mode', ['dict', 'rr', 'list', 'json', 'tsv'])
def test_flat_modes_use_template_wire(profile, mode):
    session = TemplateSession(b'Ada\n' if mode == 'tsv' else payload(['Ada']))
    with Service(ROOT, session=session, compatibility=profile) as service:
        value = template(service).get_results_list(mode, A={'values': VALUES})[0]
        if mode == 'dict':
            assert value == {'Employee.name': 'Ada'}
        elif mode == 'rr':
            assert value[0] == 'Ada'
        else:
            assert value == ('Ada' if mode == 'tsv' else ['Ada'])
        assert_wire(session)


@pytest.mark.parametrize('bad', [False, True])
def test_bound_url_uses_configured_opener_once_and_closes(bad):
    session = TemplateSession()
    session.routes['GET', '/service/saved'] = b'<template>' if bad else XML.encode()
    with Service(ROOT, session=session, token='secret', verify_tls='/chosen/ca.pem') as service:
        if bad:
            with pytest.raises(QueryParseError):
                Template.from_xml(ROOT + '/saved', service=service)
        else:
            assert Template.from_xml(ROOT + '/saved', service=service).title == 'Find names'
        request, = [r for r in session.requests if r.path.endswith('/saved')]
        assert parse_qs(request.url.split('?')[1])['token'] == ['secret']
        assert request.options['verify'] == '/chosen/ca.pem'
        assert all(r.closed for r in session.responses)


@pytest.mark.parametrize('helper', ['get_results_list', 'first', 'one', 'dataframe'])
@pytest.mark.parametrize('failure', [ValueError, KeyboardInterrupt])
def test_template_owned_stream_cleanup_on_parser_failure_or_interrupt(helper, failure, monkeypatch):
    session = TemplateSession()
    original = session.request
    def request(method, url, **kwargs):
        response = original(method, url, **kwargs)
        if method == 'POST' and b'format=count' not in kwargs.get('data', b''):
            def lines(**kw):
                yield b'{"results":['
                raise failure('interrupted read')
            response.iter_lines = lines
        return response
    monkeypatch.setattr(session, 'request', request)
    with Service(ROOT, session=session) as service:
        t = template(service)
        before = t.to_xml()
        from intermine314.query.parallel_offset import ParallelExecutionError
        expected = ParallelExecutionError if helper == 'dataframe' and failure is ValueError else failure
        with pytest.raises(expected):
            getattr(t, helper)(A={'values': VALUES})
        assert t.to_xml() == before
        assert_wire(session)


def test_legacy_list_name_adjustment_and_optional_switch_adjustments_actual_wire():
    session = TemplateSession()
    with Service(ROOT, session=session) as service:
        t = Template(service.model, service=service, root='Employee')
        t.name = 'Members'
        t.add_view('Employee.name')
        t.add_constraint('Employee', 'IN', 'old-list')
        t.add_constraint('Employee.name', '=', 'old', optional='off')
        before = t.to_xml()
        assert t.count(A={'list_name': 'new-list'}, B={'switched_on': True, 'value': 'Ada'}) == 1
        form = posts(session)[-1][1]
        assert form['value1'] == ['new-list'] and form['value2'] == ['Ada']
        assert form['code1'] == ['A'] and form['code2'] == ['B']
        assert t.to_xml() == before
        for invalid in ({'A': {'value': 'one', 'list_name': 'two'}}, {'A': {'list_name': ['bad']}},
                        {'B': {'switched_on': 'true'}}):
            with pytest.raises(TypeError):
                t.count(**invalid)
        with pytest.raises(ValueError, match='not switchable'):
            t.count(A={'switched_on': False})
        assert len(posts(session)) == 1


def test_mapping_adjustments_preserve_original_mapping_and_reject_mapping_scalar():
    from collections import UserDict
    session = TemplateSession()
    with Service(ROOT, session=session) as service:
        t = template(service)
        options = UserDict({'values': VALUES.copy()})
        assert t.count(A=options) == 1
        assert options == {'values': VALUES} and t.get_constraint('A').values == ['old', 'other']
        assert_wire(session)
        with pytest.raises(TypeError, match='scalar'):
            t.count(D={'value': UserDict({'unexpected': 'mapping'})})
        assert len(posts(session)) == 1
