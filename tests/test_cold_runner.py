"""Harness failure modes that otherwise turn import checks into false positives."""

import pytest

from tests.support.cold import run_cold


def test_isolated_child_ignores_pythonpath_and_preserves_package_origin(monkeypatch, tmp_path):
    fake = tmp_path / "intermine314"
    fake.mkdir()
    (fake / "__init__.py").write_text("raise RuntimeError('wrong package')")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = run_cold("import intermine314; assert Path(intermine314.__file__).resolve() == expected_package")
    assert result.returncode == 0, result.stdout + result.stderr


def test_timeout_reaps_child_and_reports_output():
    with pytest.raises(AssertionError, match="killed/reaped.*before timeout"):
        run_cold("import time; print('before timeout', flush=True); time.sleep(5)", timeout=0.2)
