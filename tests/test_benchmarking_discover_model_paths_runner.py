import json

from benchmarks.runners import discover_model_paths


_UNIFORM_KEYS = {
    "elapsed_ms",
    "max_rss_bytes",
    "status",
    "error_type",
    "tor_mode",
    "proxy_url_scheme",
    "profile_name",
}


def test_discover_runner_emits_uniform_metrics_on_success(monkeypatch, capsys):
    monkeypatch.setattr(discover_model_paths, "main", lambda: 0)

    code = discover_model_paths.run()

    assert code == 0
    payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert _UNIFORM_KEYS.issubset(payload.keys())


def test_discover_runner_emits_uniform_metrics_on_failure(monkeypatch, capsys):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(discover_model_paths, "main", _boom)

    code = discover_model_paths.run()

    assert code == 1
    payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert _UNIFORM_KEYS.issubset(payload.keys())
