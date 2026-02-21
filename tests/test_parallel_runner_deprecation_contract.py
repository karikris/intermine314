import importlib

import pytest


def test_runner_shim_removed_after_deprecation_window():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("intermine314.parallel.runner")
