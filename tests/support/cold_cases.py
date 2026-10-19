"""Independent cold-import programs; assertions execute in fresh interpreters."""

WILDCARD = r"""
import importlib.abc, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars','duckdb','pyarrow','numpy','pandas','matplotlib','intermine'}:
            raise AssertionError('Unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
for module in ORDER:
    namespace = {}
    exec('from intermine314.' + module + ' import *', namespace)
from intermine314.webservice import ServiceError, WebserviceError, idresolution, Query
from intermine314.service.errors import ServiceError as NativeServiceError
from intermine314.results import VERSION
from intermine314 import VERSION as PackageVersion
from intermine314.query import constraints
from intermine314.constraints import PATH_PATTERN
from intermine314.query.pathfeatures import PATH_PATTERN as NativePattern
assert ServiceError is NativeServiceError
assert VERSION == PackageVersion and PATH_PATTERN is NativePattern
assert constraints.BinaryConstraint is __import__('intermine314.constraints',fromlist=['BinaryConstraint']).BinaryConstraint
try:
    raise NativeServiceError('server failure')
except ServiceError:
    pass
"""

CASES = {
    'runtime_contracts.runtime_import_keeps_analytics_and_benchmark_dependencies_lazy': (r"""import json,sys; import intermine314; mods=sorted(m for m in sys.modules if m=='intermine' or m.startswith('intermine.') or any(m==p or m.startswith(p+'.') for p in ('pandas','polars','duckdb','pyarrow'))); print(json.dumps(mods))""", []),
    'lists_enrichment.enrichment_ordinary_calls_keep_analytics_imports_lazy': (r"""import sys
from tests.support.enrichment import enrichment_client
_, session, item = enrichment_client()
list(item.calculate_enrichment('widget'))
assert not any(name in sys.modules for name in ('polars', 'duckdb', 'pyarrow', 'pandas'))""", None),
    'legacy_module_facades.cold_wildcard_imports_preserve_legacy_bindings_without_analytics-1': ("ORDER = ('webservice', 'query', 'model', 'constraints', 'results', 'lists.list')\n" + WILDCARD, None),
    'legacy_module_facades.cold_wildcard_imports_preserve_legacy_bindings_without_analytics-2': ("ORDER = ('model', 'constraints', 'query', 'lists.list', 'results', 'webservice')\n" + WILDCARD, None),
    'legacy_module_facades.cold_wildcard_imports_preserve_legacy_bindings_without_analytics-3': ("ORDER = ('results', 'lists.list', 'query', 'webservice', 'constraints', 'model')\n" + WILDCARD, None),
    'legacy_module_facades.facade_imports_are_lazy_and_never_require_analytics': (r"""
import importlib
import importlib.abc
import logging
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'matplotlib', 'plotly', 'intermine'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
original_handlers = list(logging.getLogger().handlers)
modules = [importlib.import_module('intermine314.' + name) for name in
           ('webservice', 'constraints', 'pathfeatures', 'results', 'errors', 'query', 'decorators', 'util', 'model', 'lists')]
assert logging.getLogger().handlers == original_handlers
assert 'intermine314.service.service' not in sys.modules
assert 'intermine314.query.builder' not in sys.modules
assert 'intermine314.service.session' not in sys.modules
for module in modules:
    for name in module.__all__:
        getattr(module, name)
""", None),
    'query_parquet.public_helper_signature_and_lazy_imports': (r"""import json,sys; from intermine314.export import query_parquet; print(json.dumps(sorted(m for m in sys.modules if any(m==p or m.startswith(p+'.') for p in ('pandas','polars','duckdb','pyarrow')))))""", []),
    'service_metadata.metadata_facade_shares_methods_and_has_no_analytics_imports': (r"""
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'intermine'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
from intermine314.service.service import Service as Native
from intermine314.webservice import Service as Legacy
from tests.fixtures.compatibility import FixtureSession, SERVICE_ROOT
for name in ('search', 'widgets', 'release', 'resolve_service_path', 'version'):
    assert getattr(Native, name) is getattr(Legacy, name)
session = FixtureSession.service()
session.routes[('POST', '/service/search')] = b'{"error":null,"results":[],"facets":{}}'
session.routes[('GET', '/service/check/probe')] = b'/probe'
with Legacy(SERVICE_ROOT, session=session) as service:
    assert service.search('unicode 人') == ([], {})
    assert 'age_groups' in service.widgets
    assert isinstance(service.release, str)
    assert service.resolve_service_path('probe') == b'/probe'
assert 'intermine314.query.builder' not in sys.modules
assert all(response.close_calls == 1 for response in session.responses)
""", None),
    'import_csv.public_import_signature_and_imports_are_lazy': (r"""import json,sys; from intermine314.export import import_csv; print(json.dumps(sorted(m for m in sys.modules if any(m==p or m.startswith(p+'.') for p in ('pandas','polars','duckdb','pyarrow')))))""", []),
    'lists_crud.ordinary_list_facades_and_usage_do_not_import_analytics': (r"""import json, sys
from intermine314.lists import List, ListManager, ListServiceError
from intermine314.lists.listmanager import safe_key
from intermine314.service.service import Service
from tests.fixtures.compatibility import FixtureSession, assert_closed, fixture_bytes
session=FixtureSession.service()
session.routes['GET','/service/lists']=fixture_bytes('lists.json')
service=Service('https://offline.example/service', session=session)
manager=service.list_manager(); manager.get_all_lists()
print(json.dumps([m for m in sys.modules if m.split('.')[0] in ('pandas','polars','duckdb','pyarrow')]))""", []),
    'query_manager_helpers.helpers_import_and_execute_without_heavy_or_original_imports': (r"""
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'matplotlib', 'lxml', 'intermine'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
from intermine314 import query_manager as qm
assert 'intermine314.service.service' not in sys.modules
from tests.support.saved_queries import helper_session, TOKEN, XML
session = helper_session()
assert qm.save_mine_and_token('OfflineMine', TOKEN, session=session) is None
assert qm.get_all_query_names() == 'Zulu, a/+&人'
assert qm.post_query(XML, overwrite=True) == 'a/+&人 is posted'
qm._state.close()
assert session.close_calls == 0
""", None),
    'bar_chart_helpers.import_configuration_and_raw_query_keep_analytics_lazy': (r"""
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'numpy', 'matplotlib', 'lxml', 'intermine'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
from intermine314 import bar_chart as bc
assert 'intermine314.service.service' not in sys.modules
from tests.support.saved_queries import helper_session
session = helper_session()
session.routes[('GET', '/custom/service/query/results')] = b'gene\ta\t1\n'
assert bc.save_mine_and_token('OfflineMine', 'secret', session=session) is None
assert bc.get_query('<query/>') == [['gene', 'a', '1'], '']
bc._state.close()
""", None),
    'bar_chart_helpers.missing_optional_plots_dependency_is_actionable_before_network': (r"""
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
""", None),
    'template_constraints.template_public_imports_remain_lazy_and_use_owned_modules': (r"""
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
""", None),
    'template_discovery.discovery_imports_stay_lazy_and_getters_need_no_analytics': (r"""
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'intermine'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
from intermine314.service.service import Service as Native
from intermine314.webservice import Service as Legacy
from tests.fixtures.compatibility import FixtureSession, SERVICE_ROOT, fixture_bytes
for name in ('templates', 'all_templates', 'all_templates_names', 'get_template', 'get_template_by_user'):
    assert getattr(Native, name) is getattr(Legacy, name)
for cls in (Native, Legacy):
    session = FixtureSession.service()
    session.routes[('GET', '/service/templates')] = fixture_bytes('templates.xml')
    session.routes[('GET', '/service/alltemplates')] = fixture_bytes('all-templates.xml')
    with cls(SERVICE_ROOT, session=session) as service:
        assert 'employeeByName' in service.templates
        assert service.all_templates_names['bob'] == ['shared']
        if cls is Native:
            assert 'intermine314.query.builder' not in sys.modules
        assert service.get_template('employeeByName').compatibility == service.compatibility
        assert service.get_template_by_user('shared', 'bob').user_name == 'bob'
    assert session.close_calls == 0
    assert all(response.close_calls == 1 for response in session.responses)
""", None),
    'idresolution.import_and_execution_require_no_query_analytics_or_original_client': (r"""
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'matplotlib', 'plotly', 'intermine'} or fullname.startswith('intermine314.query'):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
import intermine314.webservice
assert 'intermine314.idresolution' not in sys.modules
from intermine314.idresolution import Job, get_json
from tests.fixtures.compatibility import SERVICE_ROOT
from tests.support.jobs import job_session
service = intermine314.webservice.Service(SERVICE_ROOT, session=job_session())
job = service.resolve_ids('Employee', ['00123'])
assert isinstance(job, Job)
assert job.fetch_status() == 'PENDING'
assert job.fetch_results()['unresolved'] == ['unknown']
assert job.delete() is None
service.close()
""", None),
    'registry_helpers.helpers_import_and_execution_keep_analytics_and_original_package_lazy': (r"""
import importlib.abc
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'matplotlib', 'intermine'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
from intermine314 import registry
assert 'intermine314.service.service' not in sys.modules
from tests.support.registry import helper_session
session = helper_session()
assert registry.getVersion('OfflineMine', session=session)['API Version:'] == '30'
assert registry.getInfo('OfflineMine', session=session) is None
assert registry.getMines(session=session) is None
assert registry.getData('OfflineMine', session=session) is None
assert all(response.close_calls == 1 for response in session.responses)
assert session.close_calls == 0
""", None),
}
