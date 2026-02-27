from __future__ import annotations

import importlib
import sys


def _reload_benchmarks_module():
    sys.modules.pop("benchmarks.benchmarks", None)
    return importlib.import_module("benchmarks.benchmarks")


def test_benchmarks_parse_args_accepts_argv():
    benchmarks_module = _reload_benchmarks_module()

    parsed = benchmarks_module.parse_args(
        [
            "--mine-url",
            "https://example.org/mine",
            "--rows",
            "123",
            "--query-kinds",
            "simple",
            "--offline-replay-stage-io",
            "--parity-sample-mode",
            "stride",
            "--parity-sample-size",
            "32",
            "--strict-parity",
        ]
    )

    assert parsed.mine_url == "https://example.org/mine"
    assert parsed.rows == 123
    assert parsed.query_kinds == "simple"
    assert parsed.offline_replay_stage_io is True
    assert parsed.parity_sample_mode == "stride"
    assert parsed.parity_sample_size == 32
    assert parsed.strict_parity is True


def test_benchmarks_parse_args_defaults_to_strict_parity_enabled():
    benchmarks_module = _reload_benchmarks_module()

    parsed = benchmarks_module.parse_args([])

    assert parsed.strict_parity is True


def test_importing_benchmarks_has_no_legacy_shim_side_effects():
    had_urlparse = "urlparse" in sys.modules
    original_sys_path = list(sys.path)

    _reload_benchmarks_module()

    assert sys.path == original_sys_path
    if not had_urlparse:
        assert "urlparse" not in sys.modules


def test_legacy_shims_context_restores_modules():
    benchmarks_module = _reload_benchmarks_module()
    had_urlparse = "urlparse" in sys.modules
    original_urlparse = sys.modules.get("urlparse")

    with benchmarks_module.legacy_shims_context(True):
        assert "urlparse" in sys.modules

    if had_urlparse:
        assert sys.modules.get("urlparse") is original_urlparse
    else:
        assert "urlparse" not in sys.modules
