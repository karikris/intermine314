from __future__ import annotations

import importlib

from intermine314 import model as model_pkg


def _reload_model_pkg():
    module = importlib.reload(model_pkg)
    module._EMITTED_DEPRECATIONS.clear()
    module.__dict__.pop("_XML_TAG_MODEL", None)
    module.__dict__.pop("_source_ref", None)
    module.__dict__.pop("_copy_subclasses", None)
    return module


def test_model_package_public_exports_are_explicit_and_narrow():
    module = _reload_model_pkg()
    expected = {
        "Model",
        "ModelError",
        "ModelParseError",
        "PathParseError",
        "Field",
        "Attribute",
        "Reference",
        "Collection",
        "Class",
        "ComposedClass",
        "Path",
        "ConstraintTree",
        "ConstraintNode",
        "CodelessNode",
        "Column",
        "NUMERIC_TYPES",
    }
    assert set(module.__all__) == expected
    assert "_XML_TAG_MODEL" not in module.__all__
    assert "_source_ref" not in module.__all__
    assert "_copy_subclasses" not in module.__all__


def test_model_package_unknown_attribute_raises_attribute_error():
    module = _reload_model_pkg()

    try:
        _ = module._not_a_real_model_symbol
    except AttributeError:
        pass
    else:  # pragma: no cover - defensive: should never happen
        raise AssertionError("Expected AttributeError for unknown model package symbol")
