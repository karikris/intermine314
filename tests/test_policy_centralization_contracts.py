from types import SimpleNamespace
import inspect
import importlib

import pytest

import intermine314.config.storage_policy as storage_policy
import intermine314.config.transport_policy as transport_policy
import intermine314.export.fetch as export_fetch
import intermine314.query.builder as query_builder
import intermine314.service.tor as tor_module
import intermine314.service.transport as transport_module
from intermine314.service.session import InterMineURLOpener


def _retry_snapshot(session):
    adapter = session.get_adapter("https://")
    retries = adapter.max_retries
    return {
        "total": retries.total,
        "backoff_factor": retries.backoff_factor,
        "status_forcelist": set(retries.status_forcelist or ()),
        "allowed_methods": set(retries.allowed_methods or ()),
    }


class _ProxySession:
    def __init__(self, *, proxy_url: str, trust_env: bool = False):
        self.proxies = {"http": proxy_url, "https": proxy_url}
        self.trust_env = trust_env

    def request(self, *_args, **_kwargs):
        raise AssertionError("request should not be called in policy wiring tests")


def test_storage_policy_is_single_sourced_for_query_and_export():
    assert query_builder._validate_parquet_compression is storage_policy.validate_parquet_compression
    assert export_fetch.validate_parquet_compression is storage_policy.validate_parquet_compression
    assert query_builder._validate_duckdb_identifier is storage_policy.validate_duckdb_identifier
    assert export_fetch.validate_duckdb_identifier is storage_policy.validate_duckdb_identifier


def test_storage_policy_contract_defaults_and_validation():
    default_codec = storage_policy.default_parquet_compression()
    valid_codecs = storage_policy.valid_parquet_compressions()

    assert default_codec in valid_codecs
    assert storage_policy.validate_parquet_compression(None) == default_codec
    assert storage_policy.validate_parquet_compression(" SnApPy ") == "snappy"
    with pytest.raises(ValueError, match="Unsupported Parquet compression"):
        storage_policy.validate_parquet_compression("invalid")

    assert storage_policy.validate_duckdb_identifier("results_01") == "results_01"
    with pytest.raises(ValueError, match="valid SQL identifier"):
        storage_policy.validate_duckdb_identifier("results-table")


def test_transport_retry_policy_resolves_from_runtime_defaults(monkeypatch):
    stub_runtime_defaults = SimpleNamespace(
        transport_defaults=SimpleNamespace(
            default_http_retry_total=11,
            default_http_retry_backoff_seconds=1.25,
            default_http_retry_status_codes=(429, 503),
            default_http_retry_methods=("GET", "POST"),
        )
    )
    runtime_defaults_mod = importlib.import_module("intermine314.config.runtime_defaults")
    monkeypatch.setattr(runtime_defaults_mod, "get_runtime_defaults", lambda: stub_runtime_defaults)

    policy = transport_policy.resolve_http_retry_policy()

    assert policy.total == 11
    assert policy.backoff_factor == 1.25
    assert policy.status_forcelist == (429, 503)
    assert policy.allowed_methods == ("GET", "POST")


def test_transport_retry_policy_is_single_sourced_for_transport_tor_and_session(monkeypatch):
    policy = transport_policy.HTTPRetryPolicy(
        total=7,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 503),
        allowed_methods=("GET", "POST"),
    )
    monkeypatch.setattr(transport_module, "resolve_http_retry_policy", lambda: policy)

    direct_session = transport_module.build_session(proxy_url=None)
    tor_session = tor_module.tor_session()
    opener = InterMineURLOpener(proxy_url=None)

    expected = {
        "total": 7,
        "backoff_factor": 0.75,
        "status_forcelist": {429, 500, 503},
        "allowed_methods": {"GET", "POST"},
    }
    assert _retry_snapshot(direct_session) == expected
    assert _retry_snapshot(tor_session) == expected
    assert _retry_snapshot(opener._session) == expected


def test_tor_policy_module_has_no_stale_config_constants_dependency():
    source = inspect.getsource(tor_module)
    assert "config.constants" not in source


def test_tor_proxy_url_defaults_resolve_from_runtime_defaults(monkeypatch):
    stub_service_defaults = SimpleNamespace(
        default_registry_instances_url="https://registry.example/service/instances",
        default_request_timeout_seconds=33,
        default_tor_proxy_scheme="socks5h",
        default_tor_socks_host="127.0.0.8",
        default_tor_socks_port=9150,
    )
    stub_runtime_defaults = SimpleNamespace(service_defaults=stub_service_defaults)
    monkeypatch.setattr(tor_module, "get_runtime_defaults", lambda: stub_runtime_defaults)

    assert tor_module.tor_proxy_url() == "socks5h://127.0.0.8:9150"


def test_tor_registry_defaults_are_single_sourced_from_runtime_defaults(monkeypatch):
    stub_service_defaults = SimpleNamespace(
        default_registry_instances_url="https://registry.example/service/instances",
        default_request_timeout_seconds=33,
        default_tor_proxy_scheme="socks5h",
        default_tor_socks_host="127.0.0.8",
        default_tor_socks_port=9150,
    )
    stub_runtime_defaults = SimpleNamespace(service_defaults=stub_service_defaults)
    monkeypatch.setattr(tor_module, "get_runtime_defaults", lambda: stub_runtime_defaults)

    class _FakeRegistry:
        last_kwargs = None

        def __init__(self, **kwargs):
            _FakeRegistry.last_kwargs = dict(kwargs)

    service_module = importlib.import_module("intermine314.service.service")
    monkeypatch.setattr(service_module, "Registry", _FakeRegistry)
    session = _ProxySession(proxy_url="socks5h://127.0.0.8:9150", trust_env=False)

    registry = tor_module.tor_registry(session=session, strict=True)
    assert isinstance(registry, _FakeRegistry)
    assert _FakeRegistry.last_kwargs is not None
    assert _FakeRegistry.last_kwargs["registry_url"] == "https://registry.example/service/instances"
    assert _FakeRegistry.last_kwargs["request_timeout"] == 33
    assert _FakeRegistry.last_kwargs["proxy_url"] == "socks5h://127.0.0.8:9150"
    assert _FakeRegistry.last_kwargs["strict_tor_proxy_scheme"] is True
