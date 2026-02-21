from __future__ import annotations

import warnings

import intermine314.util as util_pkg


def _reset_deprecations():
    util_pkg._EMITTED_DEPRECATIONS.clear()


def test_json_alias_warns_once_and_delegates():
    _reset_deprecations()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        assert util_pkg.json_loads('{"alpha":1}') == {"alpha": 1}
        assert util_pkg.json_loads('{"beta":2}') == {"beta": 2}

    messages = [str(item.message) for item in caught]
    matching = [msg for msg in messages if "intermine314.util.json_loads is deprecated" in msg]
    assert len(matching) == 1


def test_timed_alias_warns_once():
    _reset_deprecations()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        with util_pkg.timed() as payload:
            assert isinstance(payload, dict)
        with util_pkg.timed() as payload_again:
            assert isinstance(payload_again, dict)

    messages = [str(item.message) for item in caught]
    matching = [msg for msg in messages if "intermine314.util.timed is deprecated" in msg]
    assert len(matching) == 1


def test_configure_logging_alias_removed_from_util_root():
    assert "configure_logging" not in util_pkg.__all__
    assert not hasattr(util_pkg, "configure_logging")
