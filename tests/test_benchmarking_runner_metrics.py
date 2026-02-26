import time

from benchmarks.runners import runner_metrics


_UNIFORM_KEYS = {
    "elapsed_ms",
    "max_rss_bytes",
    "status",
    "error_type",
    "tor_mode",
    "proxy_url_scheme",
    "profile_name",
}


def test_measure_startup_returns_process_stable_timestamp():
    first = runner_metrics.measure_startup()
    time.sleep(0.001)
    second = runner_metrics.measure_startup()
    assert first is second
    assert first.started_at == second.started_at


def test_metric_fields_emit_uniform_contract():
    startup = runner_metrics.StartupMeasurement(started_at=time.perf_counter() - 0.01)
    payload = runner_metrics.metric_fields(
        startup=startup,
        status="ok",
        error_type="none",
        tor_mode="disabled",
        proxy_url_scheme="none",
        profile_name="test_profile",
    )
    assert _UNIFORM_KEYS.issubset(payload.keys())
