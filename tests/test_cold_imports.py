"""Every scenario has its own isolated interpreter and bounded lifetime."""

import json

import pytest

from tests.support.cold import run_cold
from tests.support.cold_cases import CASES


@pytest.mark.parametrize("scenario", CASES)
def test_cold_import_contract(scenario):
    script, expected_stdout = CASES[scenario]
    result = run_cold(script)
    assert result.returncode == 0, result.stdout + result.stderr
    if expected_stdout is not None:
        assert json.loads(result.stdout) == expected_stdout
