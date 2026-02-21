from __future__ import annotations

import warnings

import intermine314.util.logging as util_logging


def _reset_state():
    util_logging._CONFIGURE_LOGGING_WARNED = False


def test_configure_logging_is_noop_and_does_not_call_basicconfig(monkeypatch):
    _reset_state()
    calls = []

    def _fake_basic_config(*_args, **_kwargs):
        calls.append(True)

    monkeypatch.setattr(util_logging.logging, "basicConfig", _fake_basic_config)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", DeprecationWarning)
        util_logging.configure_logging(10)

    assert calls == []


def test_configure_logging_warns_once_per_process():
    _reset_state()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        util_logging.configure_logging(20)
        util_logging.configure_logging(30)

    messages = [str(item.message) for item in caught]
    matching = [msg for msg in messages if "configure_logging() is deprecated and is now a no-op" in msg]
    assert len(matching) == 1

