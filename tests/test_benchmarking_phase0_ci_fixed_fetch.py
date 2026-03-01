import json

from benchmarks.bench_fetch import ModeRun
from benchmarks.runners import phase0_ci_fixed_fetch


_UNIFORM_KEYS = {
    "elapsed_ms",
    "max_rss_bytes",
    "status",
    "error_type",
    "tor_mode",
    "proxy_url_scheme",
    "profile_name",
}


def test_run_skips_when_preflight_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        phase0_ci_fixed_fetch,
        "_probe_direct",
        lambda *_args, **_kwargs: {
            "mode": "direct",
            "host": "example.org",
            "reason": "dns_failed",
            "err_type": "OSError",
            "elapsed_s": 0.1,
        },
    )

    code = phase0_ci_fixed_fetch.run(
        [
            "--mine-url",
            "https://example.org/service",
            "--rows-target",
            "100",
            "--page-size",
            "50",
            "--workers",
            "2",
        ]
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == phase0_ci_fixed_fetch.SKIP_EXIT_CODE
    assert payload["status"] == "skipped"
    assert payload["reason"] == "dns_failed"
    assert _UNIFORM_KEYS.issubset(payload.keys())


def test_run_reports_ok_when_fetch_completes(monkeypatch, capsys):
    monkeypatch.setattr(
        phase0_ci_fixed_fetch,
        "_probe_direct",
        lambda *_args, **_kwargs: {
            "mode": "direct",
            "host": "example.org",
            "reason": "ok",
            "err_type": "none",
            "elapsed_s": 0.1,
        },
    )
    monkeypatch.setattr(
        phase0_ci_fixed_fetch,
        "_run_mode",
        lambda **_kwargs: ModeRun(
            mode="intermine314_w2",
            repetition=1,
            seconds=1.2,
            rows=100,
            rows_per_s=83.3333,
            retries=0,
            available_rows_per_pass=500,
            effective_workers=2,
            block_stats={"block_count": 1.0},
            stage_timings={"stream_seconds": 1.2},
        ),
    )

    code = phase0_ci_fixed_fetch.run(
        [
            "--mine-url",
            "https://example.org/service",
            "--rows-target",
            "100",
            "--page-size",
            "50",
            "--workers",
            "2",
        ]
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == phase0_ci_fixed_fetch.SUCCESS_EXIT_CODE
    assert payload["status"] == "ok"
    assert payload["run"]["rows"] == 100
    assert payload["mode"] == "intermine314_w2"
    assert _UNIFORM_KEYS.issubset(payload.keys())
