import importlib
import sys

import pytest


def _unload_modules(*names: str):
    for name in names:
        sys.modules.pop(name, None)


def test_config_facade_import_is_lazy():
    _unload_modules(
        "intermine314.config",
        "intermine314.config.loader",
        "intermine314.config.runtime_defaults",
        "intermine314.service.transport",
    )
    config_module = importlib.import_module("intermine314.config")

    assert "intermine314.config.loader" not in sys.modules
    assert "intermine314.config.runtime_defaults" not in sys.modules
    assert "intermine314.service.transport" not in sys.modules

    _ = config_module.load_runtime_defaults
    assert "intermine314.config.loader" in sys.modules
    assert "intermine314.config.runtime_defaults" not in sys.modules

    _ = config_module.get_runtime_defaults
    assert "intermine314.config.runtime_defaults" in sys.modules


def test_config_facade_dir_and_attribute_contract():
    _unload_modules("intermine314.config")
    config_module = importlib.import_module("intermine314.config")

    for symbol in config_module.__all__:
        assert symbol in dir(config_module)

    with pytest.raises(AttributeError):
        _ = config_module.not_a_config_symbol
