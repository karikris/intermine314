from intermine314.config import runtime_defaults as runtime_defaults_mod


def _clear_defaults_state():
    runtime_defaults_mod.clear_runtime_defaults_cache()
    runtime_defaults_mod.reset_runtime_defaults_load_telemetry()


def test_runtime_defaults_prefers_packaged_toml_when_valid(monkeypatch):
    _clear_defaults_state()
    monkeypatch.setattr(
        runtime_defaults_mod,
        "load_packaged_runtime_defaults_detailed",
        lambda: {
            "source": "packaged_toml",
            "payload": {
                "meta": {"schema_version": runtime_defaults_mod.RUNTIME_DEFAULTS_SCHEMA_VERSION},
                "query_defaults": {"default_parallel_workers": 9},
            },
            "error_kind": None,
        },
    )
    monkeypatch.setattr(runtime_defaults_mod, "load_runtime_defaults_override_detailed", lambda: None)

    defaults = runtime_defaults_mod.get_runtime_defaults()
    telemetry = runtime_defaults_mod.runtime_defaults_load_telemetry()
    assert defaults.query_defaults.default_parallel_workers == 9
    assert telemetry["source"] == "packaged_toml"
    assert telemetry["schema_status"] == "ok"


def test_runtime_defaults_invalid_override_falls_back_to_packaged(monkeypatch):
    _clear_defaults_state()
    monkeypatch.setattr(
        runtime_defaults_mod,
        "load_packaged_runtime_defaults_detailed",
        lambda: {
            "source": "packaged_toml",
            "payload": {
                "meta": {"schema_version": runtime_defaults_mod.RUNTIME_DEFAULTS_SCHEMA_VERSION},
                "query_defaults": {"default_parallel_workers": 8},
            },
            "error_kind": None,
        },
    )
    monkeypatch.setattr(
        runtime_defaults_mod,
        "load_runtime_defaults_override_detailed",
        lambda: {
            "source": "override_toml",
            "payload": None,
            "error_kind": "invalid_toml",
        },
    )

    defaults = runtime_defaults_mod.get_runtime_defaults()
    telemetry = runtime_defaults_mod.runtime_defaults_load_telemetry()
    assert defaults.query_defaults.default_parallel_workers == 8
    assert telemetry["source"] == "packaged_toml"
    assert telemetry["error_kind"] == "override_invalid_toml"


def test_runtime_defaults_missing_packaged_schema_uses_minimal_fallback(monkeypatch):
    _clear_defaults_state()
    monkeypatch.setattr(
        runtime_defaults_mod,
        "load_packaged_runtime_defaults_detailed",
        lambda: {
            "source": "packaged_toml",
            "payload": {"query_defaults": {"default_parallel_workers": 77}},
            "error_kind": None,
        },
    )
    monkeypatch.setattr(runtime_defaults_mod, "load_runtime_defaults_override_detailed", lambda: None)

    defaults = runtime_defaults_mod.get_runtime_defaults()
    telemetry = runtime_defaults_mod.runtime_defaults_load_telemetry()
    assert defaults.query_defaults.default_parallel_workers == 16
    assert telemetry["source"] == "minimal_fallback"
    assert telemetry["error_kind"] == "missing_packaged_schema"

