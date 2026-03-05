import json
import os
import subprocess
import sys
import inspect
import importlib
from pathlib import Path

import intermine314.export.fetch as export_fetch
import intermine314.query.builder as query_builder
import intermine314.config.storage_policy as storage_policy
import intermine314.query.constraints as query_constraints
import intermine314.service.iterators as service_iterators
from intermine314.service.errors import TorConfigurationError
from intermine314.service.transport import enforce_tor_dns_safe_proxy_url
import pytest


def test_storage_policy_is_single_sourced_for_query_and_export():
    assert query_builder._validate_parquet_compression is storage_policy.validate_parquet_compression
    assert export_fetch.validate_parquet_compression is storage_policy.validate_parquet_compression
    assert query_builder._validate_duckdb_identifier is storage_policy.validate_duckdb_identifier
    assert export_fetch.validate_duckdb_identifier is storage_policy.validate_duckdb_identifier


def test_runtime_import_does_not_pull_benchmark_only_dependencies():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    cmd = [
        sys.executable,
        "-c",
        (
            "import json,sys; import intermine314; "
            "mods=sorted(m for m in sys.modules if m=='intermine' or m.startswith('intermine.') "
            "or m=='pandas' or m.startswith('pandas.')); "
            "print(json.dumps(mods))"
        ),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads((proc.stdout or "[]").strip() or "[]")
    assert loaded == []


def test_fetch_from_mine_removes_policy_profiles_and_keeps_parallel_contract():
    params = list(inspect.signature(export_fetch.fetch_from_mine).parameters.keys())
    assert "workflow" not in params
    assert "production_profile" not in params
    assert "resource_profile" not in params
    assert "parallel_profile" not in params
    assert "large_query_mode" not in params

    for expected in [
        "mine_url",
        "root_class",
        "views",
        "page_size",
        "max_workers",
        "ordered",
        "prefetch",
        "inflight_limit",
        "max_inflight_bytes_estimate",
        "parquet_path",
        "duckdb_table",
        "managed",
    ]:
        assert expected in params


def test_legacy_object_row_modes_are_removed():
    assert "rr" not in query_builder.VALID_ITER_ROW_MODES
    assert "rr" not in query_builder.VALID_RESULT_ROW_MODES
    assert "list" not in service_iterators.ResultIterator.ROW_FORMATS
    assert "rr" not in service_iterators.ResultIterator.ROW_FORMATS
    assert not hasattr(service_iterators, "ResultObject")
    assert "json" not in service_iterators.ResultIterator.ROW_FORMATS
    assert "jsonrows" not in service_iterators.ResultIterator.ROW_FORMATS


def test_rich_constraint_parser_surface_removed():
    assert not hasattr(query_constraints, "LogicParser")
    assert not hasattr(query_constraints, "LogicGroup")
    assert not hasattr(query_constraints, "ListConstraint")


def test_constraint_factory_keeps_minimal_equality_and_in_support():
    factory = query_constraints.ConstraintFactory()
    eq_constraint = factory.make_constraint("Gene.symbol", "=", "abc")
    in_constraint = factory.make_constraint("Gene.symbol", "IN", ["abc", "def"])
    assert isinstance(eq_constraint, query_constraints.BinaryConstraint)
    assert isinstance(in_constraint, query_constraints.MultiConstraint)
    assert in_constraint.op == "ONE OF"
    assert in_constraint.to_dict()["value"] == ["abc", "def"]


def test_query_set_logic_is_not_supported_in_minimal_surface():
    with pytest.raises(NotImplementedError):
        query_builder.Query.set_logic(object(), "A and B")


def test_query_summary_surface_removed():
    params = list(inspect.signature(query_builder.Query.results).parameters.keys())
    assert "summary_path" not in params
    assert not hasattr(query_builder.Query, "summarise")


def test_query_data_plane_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("intermine314.query.data_plane")


def test_fetch_from_mine_workflow_contract_is_locked():
    """
    Golden contract:
    - if workflow is not part of the minimal API, passing it raises TypeError.
    - if workflow is reintroduced, non-elt values must be rejected.
    """

    params = inspect.signature(export_fetch.fetch_from_mine).parameters
    base_kwargs = {
        "mine_url": "https://example.org/service",
        "root_class": "Gene",
        "views": ["Gene.symbol"],
        "parquet_path": "/tmp/intermine314-workflow-contract.parquet",
    }

    if "workflow" not in params:
        with pytest.raises(TypeError, match="workflow"):
            export_fetch.fetch_from_mine(**base_kwargs, workflow="elt")
        with pytest.raises(TypeError, match="workflow"):
            export_fetch.fetch_from_mine(**base_kwargs, workflow="legacy")
        return

    with pytest.raises((TypeError, ValueError), match="workflow|elt"):
        export_fetch.fetch_from_mine(**base_kwargs, workflow="legacy")


def test_tor_strict_mode_requires_dns_safe_proxy_scheme():
    with pytest.raises(TorConfigurationError, match="socks5h"):
        enforce_tor_dns_safe_proxy_url(
            "socks5://127.0.0.1:9050",
            tor_mode=True,
            context="golden-test",
            strict_tor_proxy_scheme=True,
            allow_insecure_tor_proxy_scheme=False,
        )

    assert (
        enforce_tor_dns_safe_proxy_url(
            "socks5h://127.0.0.1:9050",
            tor_mode=True,
            context="golden-test",
            strict_tor_proxy_scheme=True,
            allow_insecure_tor_proxy_scheme=False,
        )
        == "socks5h://127.0.0.1:9050"
    )
