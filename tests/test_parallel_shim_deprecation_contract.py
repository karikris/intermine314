import importlib

import pytest


def test_ordering_shim_removed_after_deprecation_window():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("intermine314.parallel.ordering")


def test_pagination_shim_removed_after_deprecation_window():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("intermine314.parallel.pagination")
