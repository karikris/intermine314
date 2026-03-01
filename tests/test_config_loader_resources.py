import os
from pathlib import Path
from types import MappingProxyType

import pytest

from intermine314.config.loader import (
    load_mine_parallel_preferences,
    load_runtime_defaults,
    load_toml,
    resolve_runtime_defaults_path,
)


def test_load_runtime_defaults_from_packaged_resources(monkeypatch):
    monkeypatch.delenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", raising=False)

    loaded = load_runtime_defaults()

    assert isinstance(loaded, dict)
    assert "query_defaults" in loaded
    assert loaded["query_defaults"]["default_parallel_workers"] > 0


def test_load_mine_parallel_preferences_from_packaged_resources(monkeypatch):
    monkeypatch.delenv("INTERMINE314_MINE_PARALLEL_PREFERENCES_PATH", raising=False)

    loaded = load_mine_parallel_preferences()

    assert isinstance(loaded, dict)
    assert "mines" in loaded
    assert "maizemine" in loaded["mines"]


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
    payload = (
        "[query_defaults]\n"
        "default_parallel_workers = 3\n"
        "comment = \""
        + ("x" * 1_100_000)
        + "\"\n"
    )
    override.write_text(payload, encoding="utf-8")
    monkeypatch.setenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", str(override))

    loaded = load_runtime_defaults()

    assert loaded == {}


def test_resolve_runtime_defaults_path_returns_existing_packaged_path(monkeypatch):
    monkeypatch.delenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", raising=False)

    path = resolve_runtime_defaults_path()

    assert isinstance(path, Path)
    assert path.name == "defaults.toml"
    assert path.exists()


def test_load_toml_cache_invalidates_when_file_changes(tmp_path):
    import intermine314.config.loader as loader_mod

    config_path = tmp_path / "runtime-defaults.toml"
    config_path.write_text("[query_defaults]\ndefault_parallel_workers = 3\n", encoding="utf-8")

    loader_mod._load_toml_cached.cache_clear()
    first = load_toml(config_path)
    second = load_toml(config_path)
    assert first["query_defaults"]["default_parallel_workers"] == 3
    assert second["query_defaults"]["default_parallel_workers"] == 3
    assert loader_mod._load_toml_cached.cache_info().hits >= 1

    stat_before = config_path.stat()
    config_path.write_text("[query_defaults]\ndefault_parallel_workers = 9\n", encoding="utf-8")
    bumped_seconds = max(stat_before.st_mtime + 5.0, config_path.stat().st_mtime + 5.0)
    os.utime(config_path, (bumped_seconds, bumped_seconds))

    third = load_toml(config_path)
    assert third["query_defaults"]["default_parallel_workers"] == 9


def test_load_toml_returns_copy_on_read_payloads(tmp_path):
    config_path = tmp_path / "copy-on-read.toml"
    config_path.write_text(
        "[query_defaults]\ndefault_parallel_workers = 5\n[payload]\nitems = [1, 2, 3]\n",
        encoding="utf-8",
    )

    first = load_toml(config_path)
    second = load_toml(config_path)

    assert first["query_defaults"]["default_parallel_workers"] == 5
    first["query_defaults"]["default_parallel_workers"] = 99
    assert second["query_defaults"]["default_parallel_workers"] == 5
    first["payload"]["items"].append(4)
    assert second["payload"]["items"] == [1, 2, 3]


def test_load_toml_read_only_returns_recursive_mapping_proxy(tmp_path):
    config_path = tmp_path / "read-only.toml"
    config_path.write_text(
        "[query_defaults]\ndefault_parallel_workers = 5\n[payload]\nitems = [1, 2, 3]\n",
        encoding="utf-8",
    )

    payload = load_toml(config_path, read_only=True)

    assert isinstance(payload, MappingProxyType)
    assert isinstance(payload["query_defaults"], MappingProxyType)
    assert payload["payload"]["items"] == (1, 2, 3)
    with pytest.raises(TypeError):
        payload["query_defaults"]["default_parallel_workers"] = 7
