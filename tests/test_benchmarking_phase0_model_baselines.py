import json
from types import SimpleNamespace

import pytest

from benchmarks.runners import common as runner_common
from benchmarks.runners import phase0_model_baselines


_UNIFORM_KEYS = {
    "elapsed_ms",
    "max_rss_bytes",
    "status",
    "error_type",
    "tor_mode",
    "proxy_url_scheme",
    "profile_name",
}


def test_normalize_object_kinds():
    assert phase0_model_baselines._normalize_object_kinds("both") == phase0_model_baselines.VALID_OBJECT_KINDS
    assert phase0_model_baselines._normalize_object_kinds("path") == ("path",)
    assert phase0_model_baselines._normalize_object_kinds("column,path,column") == ("column", "path")
    with pytest.raises(ValueError):
        phase0_model_baselines._normalize_object_kinds("invalid")


def test_ru_maxrss_bytes_linux_conversion(monkeypatch):
    class _Usage:
        ru_maxrss = 123

    monkeypatch.setattr(runner_common.sys, "platform", "linux")
    monkeypatch.setattr(runner_common.resource, "getrusage", lambda _kind: _Usage())
    assert phase0_model_baselines._ru_maxrss_bytes() == 123 * 1024


def test_build_report_uses_import_and_object_baselines(monkeypatch):
    args = SimpleNamespace(
        kinds="path,column",
        object_count=100,
        import_repetitions=3,
        metric_goal="throughput",
        retain_objects=None,
        sample_mode="none",
        sample_size=64,
    )
    monkeypatch.setattr(
        phase0_model_baselines,
        "_run_import_baseline_subprocess",
        lambda repetitions: {"repetitions": repetitions, "seconds": {"mean": 0.01}},
    )
    monkeypatch.setattr(
        phase0_model_baselines,
        "_measure_object_creation",
        lambda kind, count, retain_objects, sample_mode, sample_size: {
            "kind": kind,
            "count": int(count),
            "elapsed_s": 0.1,
            "retain_objects": retain_objects,
            "sample_mode": sample_mode,
            "sample_size": sample_size,
        },
    )

    report = phase0_model_baselines._build_report(args)
    assert report["import_baseline"]["repetitions"] == 3
    assert report["summary"]["kinds"] == ["path", "column"]
    assert report["object_baselines"]["path"]["count"] == 100
    assert report["object_baselines"]["column"]["count"] == 100
    assert report["summary"]["metric_goal"] == "throughput"
    assert report["summary"]["retain_objects"] is False
    assert _UNIFORM_KEYS.issubset(report.keys())


def test_run_writes_json_report(tmp_path, capsys):
    out_path = tmp_path / "phase0_model_report.json"
    code = phase0_model_baselines.run(
        [
            "--kinds",
            "path",
            "--object-count",
            "20",
            "--import-repetitions",
            "1",
            "--metric-goal",
            "throughput",
            "--sample-mode",
            "head",
            "--sample-size",
            "5",
            "--json-out",
            str(out_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out.strip())
    assert code == phase0_model_baselines.SUCCESS_EXIT_CODE
    assert payload["summary"]["kinds"] == ["path"]
    assert payload["object_baselines"]["path"]["count"] == 20
    assert payload["summary"]["sample_mode"] == "head"
    assert payload["summary"]["sample_size"] == 5
    assert out_path.exists()
    assert _UNIFORM_KEYS.issubset(payload.keys())


def test_resolve_retain_objects_defaults_to_metric_goal():
    assert phase0_model_baselines._resolve_retain_objects("throughput", None) is False
    assert phase0_model_baselines._resolve_retain_objects("memory", None) is True
    assert phase0_model_baselines._resolve_retain_objects("throughput", True) is True


def test_measure_object_creation_sampling_without_full_retention(monkeypatch):
    class _Path:
        pass

    class _Model:
        @staticmethod
        def make_path(_path):
            return _Path()

        @staticmethod
        def column(_path):
            return _Path()

    monkeypatch.setattr(phase0_model_baselines, "_build_model", lambda: _Model())

    stats = phase0_model_baselines._measure_object_creation(
        "path",
        count=100,
        retain_objects=False,
        sample_mode="head",
        sample_size=7,
    )

    assert stats["count"] == 100
    assert stats["retain_objects"] is False
    assert stats["sample_mode"] == "head"
    assert stats["sample_size"] == 7
    assert stats["retained_count"] == 7
