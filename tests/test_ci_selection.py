"""CI reductions cannot narrow master, manual or reusable release obligations."""

import pytest

from scripts import ci_tests


def test_only_pull_request_optional_jobs_use_focused_scopes():
    for event in ("push", "workflow_dispatch", "workflow_call", "local"):
        for mode in ci_tests.MODES:
            assert ci_tests.scope_for(event, mode) == "full"
    for mode in ci_tests.MODES:
        assert ci_tests.scope_for("pull_request", mode) == (mode if mode in ci_tests.FOCUSED else "full")


def test_focused_selections_exist_and_preserve_boundary_scenarios():
    speed = ci_tests.select_tests("speed")
    plots = ci_tests.select_tests("plots")
    assert "tests/test_cold_imports.py" in speed and "tests/test_cold_imports.py" in plots
    assert "tests/test_json_precision.py" in speed and "tests/test_parallel_page_validation.py" in speed
    assert "tests/test_bar_chart_helpers.py" in plots and "tests/test_parquet_plots.py" in plots
    assert ci_tests.select_tests("full") == ("tests",)


def test_unknown_or_empty_selections_fail(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Unknown"):
        ci_tests.select_tests("typo")
    with pytest.raises(ValueError, match="Unknown"):
        ci_tests.scope_for("push", "typo")
    for scope in ("full", "plots", "speed"):
        with pytest.raises(RuntimeError, match="Empty or missing"):
            ci_tests.select_tests(scope, root=tmp_path)
    monkeypatch.setitem(ci_tests.FOCUSED, "speed", ())
    with pytest.raises(RuntimeError, match="Empty or missing"):
        ci_tests.select_tests("speed")
