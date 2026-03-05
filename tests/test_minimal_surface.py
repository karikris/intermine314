from intermine314.model.operators import Column
from intermine314.query.builder import Query
import intermine314.query.pathfeatures as pathfeatures
from intermine314.service.service import Registry, Service


def test_removed_service_aliases_are_not_present():
    assert not hasattr(Service, "new_query")
    assert not hasattr(Service, "tor")
    assert not hasattr(Service, "flush")
    assert not hasattr(Service, "release")
    assert not hasattr(Service, "resolve_service_path")
    assert not hasattr(Service, "get_anonymous_token")
    assert not hasattr(Service, "list_manager")
    assert not hasattr(Service, "create_list")
    assert not hasattr(Registry, "tor")


def test_removed_column_filter_alias_is_not_present():
    assert not hasattr(Column, "filter")


def test_removed_query_convenience_helpers_are_not_present():
    assert not hasattr(Query, "dataframe")
    assert not hasattr(Query, "duckdb_view")
    assert not hasattr(Query, "one")
    assert not hasattr(Query, "first")
    assert not hasattr(Query, "get_results_list")
    assert not hasattr(Query, "get_row_list")
    assert not hasattr(Query, "to_Node")
    assert not hasattr(Query, "add_path_description")
    assert not hasattr(Query, "verify_pd_paths")


def test_removed_path_description_feature_is_not_present():
    assert not hasattr(pathfeatures, "PathDescription")
