from types import MappingProxyType

import pytest

from intermine314.config.loader import (
    load_runtime_defaults,
    load_toml,
)


def test_load_runtime_defaults_from_packaged_resources(monkeypatch):
    monkeypatch.delenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", raising=False)
    loaded = load_runtime_defaults()
    assert isinstance(loaded, dict)
    assert "query_defaults" in loaded
    assert loaded["query_defaults"]["default_parallel_workers"] > 0


def test_load_runtime_defaults_honors_override_path(tmp_path, monkeypatch):
    override = tmp_path / "runtime-defaults.toml"
    override.write_text(
        "[query_defaults]\ndefault_parallel_workers = 3\ndefault_parallel_page_size = 123\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", str(override))
    loaded = load_runtime_defaults()
    assert loaded["query_defaults"]["default_parallel_workers"] == 3
    assert loaded["query_defaults"]["default_parallel_page_size"] == 123


def test_load_runtime_defaults_rejects_oversized_override_path(tmp_path, monkeypatch):
    override = tmp_path / "runtime-defaults-large.toml"
    payload = "[query_defaults]\ndefault_parallel_workers = 3\ncomment = \"" + ("x" * 1_100_000) + "\"\n"
    override.write_text(payload, encoding="utf-8")
    monkeypatch.setenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", str(override))
    loaded = load_runtime_defaults()
    assert loaded == {}


def test_load_toml_copy_on_read_and_read_only_modes(tmp_path):
    config_path = tmp_path / "runtime-defaults.toml"
    config_path.write_text(
        "[query_defaults]\ndefault_parallel_workers = 5\n[payload]\nitems = [1, 2, 3]\n",
        encoding="utf-8",
    )

    first = load_toml(config_path)
    second = load_toml(config_path)
    first["query_defaults"]["default_parallel_workers"] = 99
    first["payload"]["items"].append(4)
    assert second["query_defaults"]["default_parallel_workers"] == 5
    assert second["payload"]["items"] == [1, 2, 3]

    payload = load_toml(config_path, read_only=True)
    assert isinstance(payload, MappingProxyType)
    assert isinstance(payload["query_defaults"], MappingProxyType)
    assert payload["payload"]["items"] == (1, 2, 3)
    with pytest.raises(TypeError):
        payload["query_defaults"]["default_parallel_workers"] = 7

