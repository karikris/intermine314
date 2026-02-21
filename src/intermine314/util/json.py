from __future__ import annotations

import json as _stdlib_json
import logging

try:
    import orjson as _orjson
except Exception:  # pragma: no cover
    _orjson = None

LOG = logging.getLogger(__name__)
orjson = _orjson

if _orjson is not None:
    _JSON_BACKEND = "orjson"
    _loads = _orjson.loads
    _dumps = _orjson.dumps
else:
    _JSON_BACKEND = "json"
    _loads = _stdlib_json.loads
    _dumps = _stdlib_json.dumps

if LOG.isEnabledFor(logging.DEBUG):
    LOG.debug("intermine314.util.json backend=%s", _JSON_BACKEND)


def json_backend() -> str:
    return _JSON_BACKEND


def json_loads(payload):
    if _JSON_BACKEND == "orjson":
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return _loads(payload)
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    return _loads(payload)


def json_dumps(payload):
    value = _dumps(payload)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value
