from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNERS_DIR = ROOT / "benchmarks" / "runners"


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def test_runner_helper_definitions_are_centralized():
    runner_files = [
        "benchmarks/runners/phase0_baselines.py",
        "benchmarks/runners/phase0_parallel_baselines.py",
        "benchmarks/runners/phase0_model_baselines.py",
    ]
    forbidden_helper_defs = (
        "def _stat_summary(",
        "def _pythonpath_env(",
        "def _ru_maxrss_bytes(",
        "def _run_import_baseline_subprocess(",
    )
    for rel_path in runner_files:
        body = _read(rel_path)
        for marker in forbidden_helper_defs:
            assert marker not in body, f"{marker} must stay centralized in benchmarks.runners.common ({rel_path})"


def test_runner_defaults_do_not_redefine_literal_shared_policy():
    phase0_body = _read("benchmarks/runners/phase0_baselines.py")
    live_body = _read("benchmarks/runners/run_live.py")
    parallel_body = _read("benchmarks/runners/phase0_parallel_baselines.py")
    model_body = _read("benchmarks/runners/phase0_model_baselines.py")

    assert "DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 8.0" not in phase0_body
    assert "DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 8.0" not in live_body
    assert "DEFAULT_IMPORT_REPETITIONS = 5" not in phase0_body
    assert "DEFAULT_IMPORT_REPETITIONS = 5" not in parallel_body
    assert "DEFAULT_IMPORT_REPETITIONS = 5" not in model_body


def test_common_and_constants_define_shared_primitives():
    common_body = _read("benchmarks/runners/common.py")
    constants_body = _read("benchmarks/bench_constants.py")

    assert "def stat_summary(" in common_body
    assert "def pythonpath_env(" in common_body
    assert "def ru_maxrss_bytes(" in common_body
    assert "def run_import_baseline_subprocess(" in common_body
    assert "def probe_direct(" in common_body
    assert "def probe_tor(" in common_body

    assert "DEFAULT_RUNNER_PREFLIGHT_TIMEOUT_SECONDS" in constants_body
    assert "DEFAULT_RUNNER_IMPORT_REPETITIONS" in constants_body
    assert "DEFAULT_RUNNER_PARALLEL_PROFILE" in constants_body
