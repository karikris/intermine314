from __future__ import annotations

import ipaddress
import os
import socket
from pathlib import Path

import pytest


def _live_tests_enabled() -> bool:
    value = os.getenv("INTERMINE314_RUN_LIVE_TESTS")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pytest_ignore_collect(collection_path, config):  # pragma: no cover - pytest hook
    del config
    path = Path(str(collection_path))
    if path.name.startswith("live_") and not _live_tests_enabled():
        return True
    return False


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
    if _live_tests_enabled():
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
