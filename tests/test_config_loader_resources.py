from pathlib import Path

from intermine314.config.loader import (
    load_mine_parallel_preferences,
    load_parallel_profiles,
    load_runtime_defaults,
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


def test_load_parallel_profiles_from_packaged_resources(monkeypatch):
    monkeypatch.delenv("INTERMINE314_PARALLEL_PROFILES_PATH", raising=False)

    loaded = load_parallel_profiles()

    assert isinstance(loaded, dict)
    assert "defaults" in loaded
    assert "profiles" in loaded


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
