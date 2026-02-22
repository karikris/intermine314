from __future__ import annotations

import importlib
import warnings

from intermine314 import model as model_pkg
from intermine314.model import constants as model_constants
from intermine314.model import helpers as model_helpers
from intermine314.model import xmlparse as model_xmlparse


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


def test_model_package_deprecated_constant_alias_warns_once_and_resolves():
    module = _reload_model_pkg()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        first = module._XML_TAG_MODEL
        second = module._XML_TAG_MODEL

    assert first == model_constants._XML_TAG_MODEL
    assert second == first
    messages = [str(item.message) for item in caught]
    matching = [msg for msg in messages if "intermine314.model._XML_TAG_MODEL is deprecated" in msg]
    assert len(matching) == 1


def test_model_package_deprecated_helper_aliases_warn_once_and_delegate():
    module = _reload_model_pkg()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        source_ref = module._source_ref
        copy_subclasses = module._copy_subclasses

    assert source_ref is model_xmlparse._source_ref
    assert copy_subclasses is model_helpers._copy_subclasses
    messages = [str(item.message) for item in caught]
    source_matching = [msg for msg in messages if "intermine314.model._source_ref is deprecated" in msg]
    copy_matching = [msg for msg in messages if "intermine314.model._copy_subclasses is deprecated" in msg]
    assert len(source_matching) == 1
    assert len(copy_matching) == 1


def test_model_package_unknown_attribute_raises_attribute_error():
    module = _reload_model_pkg()

    try:
        _ = module._not_a_real_model_symbol
    except AttributeError:
        pass
    else:  # pragma: no cover - defensive: should never happen
        raise AssertionError("Expected AttributeError for unknown model package symbol")
