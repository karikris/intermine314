import json
from types import SimpleNamespace

import pytest

from benchmarks.runners import phase0_parallel_baselines


def test_normalize_case_modes():
    assert phase0_parallel_baselines._normalize_case_modes("ordered") == ("ordered",)
    assert phase0_parallel_baselines._normalize_case_modes("unordered,ordered,unordered") == (
        "unordered",
        "ordered",
    )
    assert phase0_parallel_baselines._normalize_case_modes("") == phase0_parallel_baselines.VALID_CASE_MODES
    with pytest.raises(ValueError):
        phase0_parallel_baselines._normalize_case_modes("invalid")


def test_ru_maxrss_bytes_linux_conversion(monkeypatch):
    class _Usage:
        ru_maxrss = 123

    monkeypatch.setattr(phase0_parallel_baselines.sys, "platform", "linux")
    monkeypatch.setattr(phase0_parallel_baselines.resource, "getrusage", lambda _kind: _Usage())
    assert phase0_parallel_baselines._ru_maxrss_bytes() == 123 * 1024


def test_build_report_returns_success_when_any_mode_succeeds(monkeypatch):
    args = SimpleNamespace(
        modes="ordered,unordered",
        rows_target=1000,
        page_size=100,
        max_workers=4,
        prefetch=None,
        inflight_limit=None,
        ordered_window_pages=10,
        profile="large_query",
        large_query_mode=True,
        log_level="DEBUG",
        import_repetitions=3,
    )
    monkeypatch.setattr(
        phase0_parallel_baselines,
        "_run_import_baseline_subprocess",
        lambda repetitions: {"repetitions": repetitions, "seconds": {"mean": 0.01}},
    )

    def _worker(_args, mode):
        if mode == "ordered":
            return phase0_parallel_baselines.SUCCESS_EXIT_CODE, {"mode": mode, "status": "ok"}
        return phase0_parallel_baselines.FAIL_EXIT_CODE, {"mode": mode, "status": "failed"}

    monkeypatch.setattr(phase0_parallel_baselines, "_run_mode_worker", _worker)
    code, report = phase0_parallel_baselines._build_report(args)

    assert code == phase0_parallel_baselines.SUCCESS_EXIT_CODE
    assert report["summary"]["modes_succeeded"] == 1
    assert report["summary"]["modes_failed"] == 1
    assert report["parallel_baselines"]["ordered"]["status"] == "ok"
    assert report["parallel_baselines"]["unordered"]["status"] == "failed"


def test_worker_case_observability_ordered(capsys):
    code = phase0_parallel_baselines.run(
        [
            "--worker-case",
            "--mode",
            "ordered",
            "--rows-target",
            "500",
            "--page-size",
            "100",
            "--max-workers",
            "2",
            "--import-repetitions",
            "1",
            "--log-level",
            "DEBUG",
        ]
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == phase0_parallel_baselines.SUCCESS_EXIT_CODE
    assert payload["status"] == "ok"
    assert payload["rows_exported"] == 500
    assert payload["observability_probes"]["start_done_pair"] is True
    assert payload["observability_probes"]["ordered_scheduler_expectation"] is True
    assert payload["observability_probes"]["scheduler_debug_only"] is True


def test_worker_case_observability_unordered(capsys):
    code = phase0_parallel_baselines.run(
        [
            "--worker-case",
            "--mode",
            "unordered",
            "--rows-target",
            "500",
            "--page-size",
            "100",
            "--max-workers",
            "2",
            "--import-repetitions",
            "1",
            "--log-level",
            "DEBUG",
        ]
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == phase0_parallel_baselines.SUCCESS_EXIT_CODE
    assert payload["status"] == "ok"
    assert payload["rows_exported"] == 500
    assert payload["observability_probes"]["start_done_pair"] is True
    assert payload["observability_probes"]["ordered_scheduler_expectation"] is True
    assert payload["observability_probes"]["scheduler_debug_only"] is True
