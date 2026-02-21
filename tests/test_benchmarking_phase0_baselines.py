import json
from types import SimpleNamespace

import pytest

from benchmarks.runners import phase0_baselines


def test_normalize_mode_sequence():
    assert phase0_baselines._normalize_mode_sequence("direct") == ("direct",)
    assert phase0_baselines._normalize_mode_sequence("tor") == ("tor",)
    assert phase0_baselines._normalize_mode_sequence("both") == ("direct", "tor")
    with pytest.raises(ValueError):
        phase0_baselines._normalize_mode_sequence("invalid")


def test_ru_maxrss_bytes_linux_conversion(monkeypatch):
    class _Usage:
        ru_maxrss = 123

    monkeypatch.setattr(phase0_baselines.sys, "platform", "linux")
    monkeypatch.setattr(phase0_baselines.resource, "getrusage", lambda _kind: _Usage())
    assert phase0_baselines._ru_maxrss_bytes() == 123 * 1024


def test_build_report_returns_success_when_any_mode_succeeds(monkeypatch):
    args = SimpleNamespace(
        mode="both",
        workflow="elt",
        mine_url="https://example.org/service",
        benchmark_target="auto",
        rows_target=1000,
        page_size=100,
        max_workers=None,
        ordered="unordered",
        ordered_window_pages=10,
        parquet_compression="zstd",
        log_level="INFO",
        import_repetitions=3,
    )
    monkeypatch.setattr(
        phase0_baselines,
        "_run_import_baseline_subprocess",
        lambda repetitions: {"repetitions": repetitions, "seconds": {"mean": 0.01}},
    )

    def _worker(_args, mode):
        if mode == "direct":
            return phase0_baselines.SUCCESS_EXIT_CODE, {"mode": mode, "status": "ok"}
        return phase0_baselines.SKIP_EXIT_CODE, {"mode": mode, "status": "skipped"}

    monkeypatch.setattr(phase0_baselines, "_run_mode_worker", _worker)
    code, report = phase0_baselines._build_report(args)

    assert code == phase0_baselines.SUCCESS_EXIT_CODE
    assert report["summary"]["modes_succeeded"] == 1
    assert report["summary"]["modes_skipped"] == 1
    assert report["export_baselines"]["direct"]["status"] == "ok"
    assert report["export_baselines"]["tor"]["status"] == "skipped"


def test_worker_export_skip_on_failed_preflight(monkeypatch, capsys):
    monkeypatch.setattr(
        phase0_baselines,
        "_resolve_workload_settings",
        lambda _args: phase0_baselines.WorkloadSettings(
            mine_url="https://example.org/service",
            root_class="Gene",
            views=["Gene.primaryIdentifier"],
            joins=[],
        ),
    )
    monkeypatch.setattr(
        phase0_baselines,
        "_probe_direct",
        lambda _mine_url, _timeout: {
            "mode": "direct",
            "host": "example.org",
            "reason": "dns_failed",
            "err_type": "gaierror",
            "elapsed_s": 0.001,
        },
    )
    code = phase0_baselines.run(["--worker-export", "--mode", "direct"])
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == phase0_baselines.SKIP_EXIT_CODE
    assert payload["status"] == "skipped"
    assert payload["reason"] == "dns_failed"
