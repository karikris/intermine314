from __future__ import annotations

import ipaddress
import os
import socket
from pathlib import Path

import pytest

from tests.live_test_config import LIVE_TESTS_ENABLED


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _benchmark_tests_enabled() -> bool:
    return _env_flag("INTERMINE314_RUN_BENCHMARK_TESTS", False)


def _full_tests_enabled() -> bool:
    return _env_flag("INTERMINE314_RUN_FULL_TESTS", False)


_LEAN_CORE_TEST_FILES = {
    "test_runtime_defaults_model.py",
    "test_resource_profile.py",
    "test_openanything_transport.py",
    "test_tor_convenience.py",
    "test_query_parallel_offset.py",
}

_LEAN_NODEID_PREFIXES = (
    "tests/test_runtime_defaults_model.py::",
    "tests/test_resource_profile.py::",
    "tests/test_openanything_transport.py::test_openanything_streaming_response_closes_on_early_termination",
    "tests/test_tor_convenience.py::test_tor_service_defaults_to_strict_dns_safe_proxy_scheme",
    "tests/test_query_parallel_offset.py::TestQueryParallelOffset::test_ordered_mode_early_termination_closes_executor_context",
)


def _is_benchmark_test_path(path: Path) -> bool:
    name = path.name.lower()
    if name.startswith("test_benchmarking_"):
        return True
    if name.startswith("test_perf_") or name.startswith("perf_"):
        return True
    parts = {part.lower() for part in path.parts}
    return "benchmarks" in parts and "cases" in parts and name.startswith("test_")


def _is_unit_test_path(path: Path) -> bool:
    if path.suffix.lower() != ".py":
        return False
    name = path.name.lower()
    if not name.startswith("test_"):
        return False
    if "live_" in name:
        return False
    return "tests" in {part.lower() for part in path.parts}


def pytest_ignore_collect(collection_path, config):  # pragma: no cover - pytest hook
    del config
    path = Path(str(collection_path))
    benchmark_enabled = _benchmark_tests_enabled()
    full_enabled = _full_tests_enabled()
    if (not benchmark_enabled) and _is_benchmark_test_path(path):
        return True
    if (not full_enabled) and (not benchmark_enabled) and _is_unit_test_path(path):
        if path.name not in _LEAN_CORE_TEST_FILES and (not path.name.startswith("live_")):
            return True
    if LIVE_TESTS_ENABLED:
        return False
    return path.name.startswith("live_")


def pytest_collection_modifyitems(config, items):  # pragma: no cover - pytest hook
    del config
    if _full_tests_enabled() or _benchmark_tests_enabled():
        return
    selected = []
    deselected = []
    for item in items:
        nodeid = str(item.nodeid)
        if any(nodeid.startswith(prefix) for prefix in _LEAN_NODEID_PREFIXES):
            selected.append(item)
        else:
            deselected.append(item)
    if deselected:
        items[:] = selected


def _is_loopback_host(host: str) -> bool:
    value = str(host).strip().lower()
    if not value:
        return False
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _extract_host(address: object) -> str | None:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return None


@pytest.fixture(autouse=True)
def _disable_live_network_guard(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    default_enabled = not LIVE_TESTS_ENABLED
    guard_enabled = _env_flag("INTERMINE314_TEST_DISABLE_NETWORK", default_enabled)
    if not guard_enabled:
        return
    if request.node.get_closest_marker("allow_network") is not None:
        return

    original_create_connection = socket.create_connection
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_create_connection(address, *args, **kwargs):
        host = _extract_host(address)
        if host is not None and not _is_loopback_host(host):
            raise AssertionError(f"live network disabled in tests: attempted create_connection to {host!r}")
        return original_create_connection(address, *args, **kwargs)

    def guarded_connect(sock, address):
        host = _extract_host(address)
        if host is not None and not _is_loopback_host(host):
            try:
                sock.close()
            except Exception:
                pass
            raise AssertionError(f"live network disabled in tests: attempted connect to {host!r}")
        return original_connect(sock, address)

    def guarded_connect_ex(sock, address):
        host = _extract_host(address)
        if host is not None and not _is_loopback_host(host):
            try:
                sock.close()
            except Exception:
                pass
            raise AssertionError(f"live network disabled in tests: attempted connect_ex to {host!r}")
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
