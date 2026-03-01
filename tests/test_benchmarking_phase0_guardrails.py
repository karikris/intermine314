import json
from types import SimpleNamespace

from benchmarks.runners import phase0_guardrails


_UNIFORM_KEYS = {
    "elapsed_ms",
    "max_rss_bytes",
    "status",
    "error_type",
    "tor_mode",
    "proxy_url_scheme",
    "profile_name",
}


def test_tor_safety_guardrail_rejects_dns_unsafe_proxy():
    payload = phase0_guardrails._tor_safety_guardrail()
    assert payload["safe_proxy_scheme"] == "socks5h"
    assert payload["tor_proxy_scheme"] == "socks5h"
    assert payload["tor_dns_safety"] == "enforced"
    assert payload["unsafe_proxy_rejected"] is True
    assert payload["unsafe_proxy_rejection_error_type"] == "TorConfigurationError"


def test_build_report_includes_import_surfaces_and_uniform_fields(monkeypatch):
    args = SimpleNamespace(import_repetitions=2)

    def _fake_baseline(*, import_snippet, repetitions):
        assert "import " in import_snippet
        assert repetitions == 2
        return {
            "repetitions": repetitions,
            "runs": [
                {
                    "seconds": 0.01,
                    "module_count": 100,
                    "imported_module_count": 8,
                    "tracemalloc_peak_bytes": 1024,
                    "max_rss_bytes": 2048,
                },
                {
                    "seconds": 0.02,
                    "module_count": 102,
                    "imported_module_count": 10,
                    "tracemalloc_peak_bytes": 1536,
                    "max_rss_bytes": 3072,
                },
            ],
            "seconds": {"mean": 0.015},
            "module_count": {"mean": 101.0},
            "tracemalloc_peak_bytes": {"mean": 1280.0},
        }

    monkeypatch.setattr(phase0_guardrails, "_run_import_baseline_subprocess", _fake_baseline)
    code, report = phase0_guardrails._build_report(args)

    assert code == phase0_guardrails.SUCCESS_EXIT_CODE
    assert sorted(report["runtime"]["surfaces"]) == sorted(phase0_guardrails.IMPORT_SURFACES.keys())
    for surface in phase0_guardrails.IMPORT_SURFACES.keys():
        baseline = report["import_guardrails"][surface]
        assert baseline["surface"] == surface
        assert baseline["imported_module_count"]["mean"] == 9.0
        assert baseline["max_rss_bytes"]["mean"] == 2560.0
    assert _UNIFORM_KEYS.issubset(report.keys())


def test_run_emits_uniform_payload(capsys):
    code = phase0_guardrails.run(["--import-repetitions", "1"])
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == phase0_guardrails.SUCCESS_EXIT_CODE
    assert _UNIFORM_KEYS.issubset(payload.keys())
