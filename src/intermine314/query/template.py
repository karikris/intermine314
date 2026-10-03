"""Saved named templates over the shared Query execution and resource runtime."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from inspect import signature

from intermine314.query.builder import ConstraintError, Query, QueryParseError
from intermine314.query.constraints import (
    BinaryConstraint,
    IsaConstraint,
    ListConstraint,
    LoopConstraint,
    MultiConstraint,
    RangeConstraint,
    SubClassConstraint,
    TemplateConstraintFactory,
    TernaryConstraint,
)
from intermine314.query.spec import TemplateMetadata


def template_query_params(name, user_name, constraints):
    """Encode active editable values accepted by the named-template server."""
    params = {'name': name, 'userName': user_name}
    index = 1
    for constraint in constraints:
        if not constraint.editable or constraint.switched_off:
            continue
        if isinstance(constraint, (SubClassConstraint, LoopConstraint, RangeConstraint, IsaConstraint)):
            raise ConstraintError(
                f'{type(constraint).__name__} cannot be edited through the template endpoint; '
                'set editable=False to retain its saved server value, or switch off an optional constraint'
            )
        if isinstance(constraint, MultiConstraint) and not constraint.values:
            raise ConstraintError('Editable template multi-value constraints require at least one value; '
                                  'provide values or switch off an optional constraint')
        for key, value in constraint.to_dict().items():
            key = {'path': 'constraint', 'extraValue': 'extra'}.get(key, key)
            params[key + str(index)] = deepcopy(value)
        index += 1
    return params


class Template(Query):
    """A named query whose editable values can be adjusted on independent clones.

    All inherited analytics helpers accept constraint-code keyword arguments
    alongside their ordinary Query options. The shared executor and transport
    handle both profiles, streaming, pagination and typed exports.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_factory = TemplateConstraintFactory(compatibility=self.compatibility)
        self.title = ''
        self.user_name = ''
        self.view_types = []
        self.comment = ''

    @property
    def editable_constraints(self):
        return [constraint for constraint in self.constraints if constraint.editable]

    def clone(self):
        clone = super().clone()
        for attr in ('title', 'user_name', 'view_types', 'comment'):
            setattr(clone, attr, deepcopy(getattr(self, attr)))
        return clone

    def add_user_name(self, user_name):
        self.user_name = user_name

    @classmethod
    def from_xml(cls, xml, *args, **kwargs):
        return super().from_xml(xml, *args, **kwargs)

    def _load_xml_metadata(self, doc):
        templates = doc.getElementsByTagName('template')
        if len(templates) != 1:
            raise QueryParseError(
                'wrong number of templates in xml. Only one <template> element is allowed. '
                f'Found {len(templates)}'
            )
        template = templates[0]
        self._xml_template_name = template.getAttribute('name')
        self.title = template.getAttribute('title')
        self.user_name = template.getAttribute('userName')
        self.view_types = template.getAttribute('dataTypes').split()
        self.comment = template.getAttribute('comment')

    @staticmethod
    def _constraint_xml_arguments(element):
        arguments = Query._constraint_xml_arguments(element)
        if element.hasAttribute('editable'):
            arguments['editable'] = element.getAttribute('editable')
        if element.hasAttribute('switchable'):
            arguments['optional'] = element.getAttribute('switchable')
        return arguments

    def to_spec(self):
        spec = super().to_spec()
        return replace(spec, constraints=tuple(deepcopy(spec.constraints)), template=TemplateMetadata(
            user_name=self.user_name, title=self.title, view_types=tuple(self.view_types), comment=self.comment,
        ))

    def to_query_params(self):
        return template_query_params(self.name, self.user_name, self.constraints)

    def get_results_path(self):
        return self.service.TEMPLATEQUERY_PATH

    def get_list_upload_uri(self):
        return self.service.root + '/template/tolist'

    def get_list_append_uri(self):
        return self.service.root + '/template/append/tolist'

    def _list_upload_params(self, *, append=False):
        return dict(self.to_query_params(), path=self.views[0])

    def get_adjusted_template(self, con_values):
        clone = self.clone()
        for code, options in con_values.items():
            con = clone.get_constraint(code)
            if not con.editable:
                raise ConstraintError(f"There is a constraint '{code}' on this query, but it is not editable")
            options = options if isinstance(options, Mapping) else {'value': options}
            allowed = {'op', 'switched_on'}
            if isinstance(con, MultiConstraint):
                allowed.add('values')
            elif isinstance(con, (BinaryConstraint, ListConstraint)):
                allowed.add('value')
            if isinstance(con, ListConstraint):
                allowed.add('list_name')
                if 'value' in options and 'list_name' in options:
                    raise TypeError('Specify either value or list_name, not both')
            if isinstance(con, TernaryConstraint):
                allowed.add('extra_value')
            unknown = options.keys() - allowed
            if unknown:
                raise TypeError(f'Invalid adjustment fields for constraint {code}: {sorted(unknown)}')
            if 'op' in options:
                op = str(options['op']).strip().upper()
                if op not in con.OPS:
                    raise TypeError(f'{op} not in {sorted(con.OPS)}')
                con.op = op
            for key, value in options.items():
                if key == 'op':
                    continue
                if key == 'switched_on':
                    if not isinstance(value, bool):
                        raise TypeError('switched_on must be a boolean')
                    (con.switch_on if value else con.switch_off)()
                    continue
                if key == 'values':
                    if not isinstance(value, (list, tuple, set)):
                        raise TypeError('values must be a list, tuple, or set')
                    value = [str(item) for item in value]
                elif isinstance(value, (list, tuple, set, Mapping)):
                    raise TypeError(f'{key} must be a scalar')
                if key == 'value' and isinstance(con, ListConstraint):
                    key = 'list_name'
                setattr(con, key, deepcopy(value))
        clone.verify_constraint_paths()
        return clone

    def results(self, row=None, start=0, size=None, summary_path=None, **con_values):
        clone = self.get_adjusted_template(con_values)
        return Query.results(clone, row, start, size, summary_path=summary_path)

    def get_results_list(self, row=None, start=0, size=None, summary_path=None, **con_values):
        clone = self.get_adjusted_template(con_values)
        return Query.get_results_list(clone, row, start, size, summary_path=summary_path)

    all = get_results_list

    def rows(self, start=0, size=None, row=None, **con_values):
        clone = self.get_adjusted_template(con_values)
        return Query.rows(clone, start, size, row)

    def get_row_list(self, start=0, size=None, **con_values):
        clone = self.get_adjusted_template(con_values)
        return Query.get_row_list(clone, start, size)

    def count(self, **con_values):
        return Query.count(self.get_adjusted_template(con_values))

    size = count

    def one(self, row='jsonobjects', **con_values):
        return Query.one(self.get_adjusted_template(con_values), row)

    def _adjusted_query_call(self, method, args, kwargs):
        # Query's signature is the single source of execution option names.
        # Remaining keywords are constraint codes, validated before work begins.
        controls = signature(method).parameters
        values = {key: value for key, value in kwargs.items() if key not in controls}
        options = {key: value for key, value in kwargs.items() if key in controls}
        clone = self.get_adjusted_template(values)
        return method(clone, *args, **options)

    def iter_rows(self, *args, **kwargs):
        return self._adjusted_query_call(Query.iter_rows, args, kwargs)

    def iter_batches(self, *args, **kwargs):
        return self._adjusted_query_call(Query.iter_batches, args, kwargs)

    def run_parallel(self, *args, **kwargs):
        return self._adjusted_query_call(Query.run_parallel, args, kwargs)

    def to_parquet(self, *args, **kwargs):
        return self._adjusted_query_call(Query.to_parquet, args, kwargs)

    def export(self, *args, **kwargs):
        return self._adjusted_query_call(Query.export, args, kwargs)

    def dataframe(self, *args, **kwargs):
        return self._adjusted_query_call(Query.dataframe, args, kwargs)

    def to_duckdb(self, *args, **kwargs):
        return self._adjusted_query_call(Query.to_duckdb, args, kwargs)
