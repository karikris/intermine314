import json
import os
import subprocess
import sys
import inspect
import importlib
from pathlib import Path

import intermine314.export.fetch as export_fetch
import intermine314.query.builder as query_builder
from intermine314.query.spec import QuerySpec
import intermine314.config.storage_policy as storage_policy
import intermine314.query.constraints as query_constraints
import intermine314.service.service as service_module
import intermine314.service.session as service_session_module
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
    assert "count" not in query_builder.VALID_RESULT_ROW_MODES
    assert "list" not in service_session_module.ResultIterator.ROW_FORMATS
    assert "rr" not in service_session_module.ResultIterator.ROW_FORMATS
    assert "count" not in service_session_module.ResultIterator.ROW_FORMATS
    assert not hasattr(service_session_module, "ResultObject")
    assert "json" not in service_session_module.ResultIterator.ROW_FORMATS
    assert "jsonrows" not in service_session_module.ResultIterator.ROW_FORMATS


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


def test_query_count_requests_wire_count_directly():
    class _Conn:
        def __init__(self, payload):
            self.payload = payload
            self.closed = False

        def read(self):
            return self.payload

        def close(self):
            self.closed = True

    class _Opener:
        def __init__(self):
            self.calls = []
            self.conn = _Conn(b"42\n")

        def open(self, url, data=None):
            self.calls.append((url, data))
            return self.conn

    class _Service:
        root = "https://example.org/service"
        QUERY_PATH = "/query/results"

        def __init__(self):
            self.opener = _Opener()

    class _Harness:
        def __init__(self):
            self.service = _Service()
            self.views = []
            self.root = "Gene"
            self.added = []

        def clone(self):
            return self

        def add_view(self, value):
            self.added.append(value)

        def to_query_params(self):
            return {"query": "<query/>"}

        def get_results_path(self):
            return self.service.QUERY_PATH

    harness = _Harness()
    got = query_builder.Query.count(harness)
    assert got == 42
    assert harness.added == ["Gene"]
    url, data = harness.service.opener.calls[0]
    assert url == "https://example.org/service/query/results"
    assert isinstance(data, bytes)
    decoded = data.decode("utf-8")
    assert "query=%3Cquery%2F%3E" in decoded
    assert "format=count" in decoded
    assert harness.service.opener.conn.closed is True


def test_query_minimal_where_helpers_delegate_to_constraint_encoder():
    class _WhereHarness:
        def __init__(self):
            self.calls = []

        def where(self, *cons, **kwargs):
            self.calls.append((cons, kwargs))
            return "ok"

    eq_h = _WhereHarness()
    assert query_builder.Query.where_eq(eq_h, "Gene.symbol", "zen") == "ok"
    assert eq_h.calls == [((("Gene.symbol", "=", "zen"),), {})]

    in_h = _WhereHarness()
    assert query_builder.Query.where_in(in_h, "Gene.symbol", ["a", "b"]) == "ok"
    assert in_h.calls == [((("Gene.symbol", "IN", ["a", "b"]),), {})]

    class _RawClone:
        def __init__(self):
            self.calls = []

        def add_constraint(self, *args):
            self.calls.append(args)

    class _RawHarness:
        def __init__(self):
            self.clone_obj = _RawClone()

        def clone(self):
            return self.clone_obj

    raw_h = _RawHarness()
    got = query_builder.Query.where_raw(raw_h, "=", "Gene.symbol", "abc")
    assert got is raw_h.clone_obj
    assert raw_h.clone_obj.calls == [("Gene.symbol", "=", "abc")]


def test_service_execute_exposes_count_and_results_contract():
    class _Conn:
        def __init__(self, payload):
            self.payload = payload
            self.closed = False

        def read(self):
            return self.payload

        def close(self):
            self.closed = True

    class _Opener:
        def __init__(self):
            self.calls = []
            self.conn = _Conn(b"3\n")

        def open(self, url, data=None):
            self.calls.append((url, data))
            return self.conn

    class _Service:
        QUERY_PATH = "/query/results"
        root = "https://example.org/service"

        def __init__(self):
            self.opener = _Opener()
            self.result_calls = []

        def get_results(self, path, params, rowformat, view, cld=None):
            self.result_calls.append((path, params, rowformat, view, cld))
            return iter([{"Gene.symbol": "zen"}])

    service = _Service()
    spec = QuerySpec(root_class="Gene", views=("Gene.symbol",))
    executor = service_module.Service.execute(service, spec)

    rows = list(executor.results(row="dict", start=2, size=5))
    assert rows == [{"Gene.symbol": "zen"}]
    path, params, rowformat, view, cld = service.result_calls[0]
    assert path == service.QUERY_PATH
    assert rowformat == "dict"
    assert params["start"] == 2
    assert params["size"] == 5
    assert view == ["Gene.symbol"]
    assert cld == "Gene"

    got = executor.count()
    assert got == 3
    url, data = service.opener.calls[0]
    assert url == "https://example.org/service/query/results"
    assert isinstance(data, bytes)
    decoded = data.decode("utf-8")
    assert "format=count" in decoded
    assert service.opener.conn.closed is True


def test_service_user_agent_defaults_to_static_runtime_value():
    assert "resolve_mine_user_agent" not in inspect.getsource(service_module)
    assert service_module._resolve_service_user_agent("https://example.org/service", None) == "intermine314/benchmark-runtime"


def test_fetch_from_mine_creates_duckdb_view_without_prepared_params(monkeypatch, tmp_path):
    executed = {"sql": None, "params": None}

    class _DummyConnection:
        def execute(self, sql, params=None):
            executed["sql"] = sql
            executed["params"] = params
            return self

    class _DummyDuckDB:
        def connect(self, database=":memory:"):
            _ = database
            return _DummyConnection()

    class _DummyQuery:
        def clear_view(self):
            return None

        def add_view(self, *views):
            _ = views
            return None

        def to_parquet(self, path, **kwargs):
            _ = kwargs
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"PAR1")
            return None

    class _DummyService:
        def __init__(self, mine_url):
            _ = mine_url
            self.closed = False
            self.query = _DummyQuery()

        def select(self, root_class):
            _ = root_class
            return self.query

        def close(self):
            self.closed = True

    monkeypatch.setattr(export_fetch, "Service", _DummyService)
    monkeypatch.setattr(export_fetch, "_require_duckdb", lambda _api_name: _DummyDuckDB())

    parquet_path = tmp_path / "duck'quote.parquet"
    payload = export_fetch.fetch_from_mine(
        mine_url="https://example.org/service",
        root_class="Gene",
        views=["Gene.primaryIdentifier"],
        parquet_path=parquet_path,
        size=10,
        managed=False,
    )

    assert payload["duckdb_table"] == "results"
    assert executed["params"] is None
    assert "read_parquet(" in str(executed["sql"])
    assert "duck''quote.parquet" in str(executed["sql"])


def test_service_select_resolves_and_caches_model_name_for_default_query_path():
    class _ModelConn:
        def __init__(self, payload):
            self.payload = payload
            self.closed = False

        def read(self):
            return self.payload

        def close(self):
            self.closed = True

    class _Opener:
        def __init__(self):
            self.calls = []

        def open(self, url, *args, **kwargs):
            _ = (args, kwargs)
            self.calls.append(url)
            return _ModelConn(b"<model name='genomic'></model>")

    service = object.__new__(service_module.Service)
    service.root = "https://example.org/service"
    service.MODEL_PATH = "/model"
    service.opener = _Opener()
    service._model_name = None
    service._query_model = None
    service.prefetch_depth = 1
    service.prefetch_id_only = False

    first_query = service_module.Service.select(service, "Gene.primaryIdentifier")
    second_query = service_module.Service.select(service, "Gene.primaryIdentifier")

    assert getattr(first_query.model, "name", None) == "genomic"
    assert getattr(second_query.model, "name", None) == "genomic"
    assert service.opener.calls == ["https://example.org/service/model"]
