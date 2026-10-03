import importlib
import inspect

import pytest

import intermine314.query.pathfeatures as pathfeatures
import intermine314.service as service_package
from intermine314.query.builder import Query
from intermine314.query.constraints import ConstraintFactory
from intermine314.service.service import Registry, Service


def test_removed_service_aliases_are_not_present():
    assert not hasattr(Service, "tor")
    assert not hasattr(Service, "flush")
    assert not hasattr(Service, "release")
    assert not hasattr(Service, "resolve_service_path")
    assert not hasattr(Service, "get_anonymous_token")
    assert not hasattr(Service, "list_manager")
    assert not hasattr(Service, "create_list")
    assert not hasattr(Registry, "tor")
    assert not hasattr(service_package, "tor_proxy_url")
    assert not hasattr(service_package, "tor_session")
    assert not hasattr(service_package, "tor_service")
    assert not hasattr(service_package, "tor_registry")


def test_removed_column_filter_alias_is_not_present():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("intermine314.model.operators")


def test_runtime_registry_module_is_not_present():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("intermine314.registry")


def test_runtime_tor_convenience_module_is_not_present():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("intermine314.service.tor")


def test_runtime_service_iterators_module_is_not_present():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("intermine314.service.iterators")


def test_removed_query_convenience_helpers_are_not_present():
    assert not hasattr(Query, "duckdb_view")
    assert not hasattr(Query, "one")
    assert not hasattr(Query, "first")
    assert not hasattr(Query, "get_results_list")
    assert not hasattr(Query, "get_row_list")
    assert not hasattr(Query, "to_Node")
    assert not hasattr(Query, "add_path_description")
    assert not hasattr(Query, "verify_pd_paths")


def test_csv_query_signatures_and_dataframe_are_available():
    for method in (Query.to_parquet, Query.to_duckdb, Query.dataframe):
        parameters = inspect.signature(method).parameters
        for name in ("csv_input", "csv_options"):
            assert parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is None
    parameters = inspect.signature(Query.dataframe).parameters
    assert list(parameters) == ["self", "start", "size", "csv_input", "csv_options", "parquet_path"]
    assert parameters["start"].default == 0
    assert parameters["size"].default is None
    assert parameters["parquet_path"].kind == inspect.Parameter.KEYWORD_ONLY


def test_export_signature_requires_explicit_csv_and_defaults_to_one_file():
    parameters = inspect.signature(Query.export).parameters
    assert parameters["format"].default == "parquet"
    assert parameters["single_file"].default is True
    for name, parameter in parameters.items():
        if name not in ("self", "path"):
            assert parameter.kind == inspect.Parameter.KEYWORD_ONLY


def test_removed_path_description_feature_is_not_present():
    assert not hasattr(pathfeatures, "PathDescription")


def test_restored_constraint_families_are_exported():
    public = importlib.import_module("intermine314.constraints")
    canonical = importlib.import_module("intermine314.query.constraints")
    for name in ("ListConstraint", "LoopConstraint", "TernaryConstraint", "RangeConstraint", "IsaConstraint"):
        assert name in canonical.__all__
        assert getattr(public, name) is getattr(canonical, name)
    assert len(ConstraintFactory.CONSTRAINT_CLASSES) == 9
