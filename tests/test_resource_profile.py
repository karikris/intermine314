from __future__ import annotations

from pathlib import Path

import pytest

from intermine314.export import resource_profile


def test_resolve_default_resource_profile():
    profile = resource_profile.resolve_resource_profile(None)

    assert profile.name == "default"
    assert profile.max_workers is None
    assert profile.max_inflight_bytes_estimate is None


def test_resolve_named_tor_low_mem_profile():
    profile = resource_profile.resolve_resource_profile("tor_low_mem")

    assert profile.name == "tor_low_mem"
    assert profile.max_workers == 2
    assert profile.prefetch == 2
    assert profile.inflight_limit == 2
    assert profile.max_inflight_bytes_estimate == 64 * 1024 * 1024
    assert profile.ordered == "window"


def test_env_temp_dir_applies_only_when_profile_has_no_temp_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(resource_profile.TEMP_DIR_ENV_VAR, str(tmp_path))

    default_profile = resource_profile.resolve_resource_profile("default")
    tor_profile = resource_profile.resolve_resource_profile("tor_low_mem")

    assert default_profile.temp_dir == str(tmp_path)
    assert tor_profile.temp_dir == "/tmp"


def test_validate_temp_dir_constraints_rejects_insufficient_free_space(monkeypatch, tmp_path):
    class _Usage:
        total = 1_000
        free = 100

    monkeypatch.setattr(resource_profile.shutil, "disk_usage", lambda _path: _Usage())

    with pytest.raises(ValueError, match="requires at least"):
        resource_profile.validate_temp_dir_constraints(
            temp_dir=tmp_path,
            min_free_bytes=500,
            context="test",
        )


def test_resolve_temp_dir_creates_path(tmp_path):
    target = tmp_path / "nested" / "temp"

    resolved = resource_profile.resolve_temp_dir(str(target))

    assert isinstance(resolved, Path)
    assert resolved == target
    assert resolved.exists()
