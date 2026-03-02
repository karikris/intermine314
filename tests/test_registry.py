from __future__ import annotations

import io
from unittest.mock import patch

from intermine314.service import Registry


class _FakeRegistryOpener:
    payload = b'{"instances":[{"name":"MineA","url":"https://mine-a.example/service"}]}'
    opened_urls: list[str] = []

    def __init__(self, **_kwargs):
        self._session = object()
        self._owns_session = False

    def open(self, url):
        self.opened_urls.append(str(url))
        return io.BytesIO(self.payload)

    def close(self):
        return None


def test_registry_request_url_construction_is_canonical():
    _FakeRegistryOpener.opened_urls = []
    cases = [
        ("https://registry.example.org", "https://registry.example.org/service/instances"),
        ("https://registry.example.org/", "https://registry.example.org/service/instances"),
        ("https://registry.example.org/registry", "https://registry.example.org/registry/mines.json"),
        (
            "https://registry.example.org/service/instances",
            "https://registry.example.org/service/instances",
        ),
    ]
    with patch("intermine314.service.service.InterMineURLOpener", _FakeRegistryOpener):
        for registry_url, expected_url in cases:
            _ = Registry(registry_url=registry_url)
            assert _FakeRegistryOpener.opened_urls[-1] == expected_url


def test_registry_json_shape_parsing_supports_instances_and_mines():
    with patch("intermine314.service.service.InterMineURLOpener", _FakeRegistryOpener):
        _FakeRegistryOpener.payload = b'{"instances":[{"name":"MineA","url":"https://mine-a.example/service"}]}'
        reg_instances = Registry(registry_url="https://registry.example.org")
        assert reg_instances.keys() == ["MineA"]

        _FakeRegistryOpener.payload = b'{"mines":[{"name":"MineB","url":"https://mine-b.example/service"}]}'
        reg_mines = Registry(registry_url="https://registry.example.org")
        assert reg_mines.keys() == ["MineB"]
