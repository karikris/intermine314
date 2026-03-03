from __future__ import annotations

import intermine314
import intermine314.config.storage_policy as storage_policy
import intermine314.query as query_pkg
import intermine314.service.transport as transport_policy
from intermine314.query import ParallelOptions, Query
from intermine314.service import Registry, Service


def test_top_level_contract_is_fetch_only():
    assert intermine314.__all__ == ["VERSION", "__version__", "fetch_from_mine"]
    assert callable(intermine314.fetch_from_mine)
    assert not hasattr(intermine314, "Service")


def test_query_contract_exposes_only_query_and_parallel_options():
    assert query_pkg.__all__ == ["Query", "ParallelOptions"]
    assert query_pkg.Query is Query
    assert query_pkg.ParallelOptions is ParallelOptions
    assert not hasattr(query_pkg, "Template")


def test_service_contract_removes_out_of_scope_apis():
    assert hasattr(Service, "select")
    assert hasattr(Service, "new_query")
    assert hasattr(Service, "close")
    assert hasattr(Service, "get_results")
    assert Registry is not None

    removed_names = (
        "list_manager",
        "iter_created_lists",
        "create_batched_lists",
        "search",
        "widgets",
        "resolve_ids",
        "iter_resolve_ids",
        "register",
        "deregister",
        "get_deregistration_token",
        "list_templates",
        "get_template",
        "get_template_by_user",
        "templates",
        "all_templates",
        "all_templates_names",
    )
    for name in removed_names:
        assert not hasattr(Service, name)


def test_policy_entrypoints_remain_explicit():
    assert callable(storage_policy.validate_parquet_compression)
    assert callable(storage_policy.validate_duckdb_identifier)
    assert callable(storage_policy.default_parquet_compression)
    assert callable(transport_policy.enforce_tor_dns_safe_proxy_url)
