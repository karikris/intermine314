from __future__ import annotations

import json as _stdlib_json
import logging
import re

try:
    import orjson as _orjson
except Exception:  # pragma: no cover
    _orjson = None

LOG = logging.getLogger(__name__)
orjson = _orjson

# The cheap first pass avoids tokenizing ordinary payloads. The second pass
# consumes complete strings/numbers so digit strings and floats cannot trigger
# an integer fallback. Compare lexemes instead of converting enormous ints.
_LONG_DIGITS = re.compile(rb"[0-9]{19}")
_JSON_TOKENS = re.compile(rb'"(?:[^"\\]|\\.)*"|(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')
_SIGNED_MIN_MAGNITUDE = b"9223372036854775808"
_UNSIGNED_MAX = b"18446744073709551615"


def _requires_exact_integer_decoder(payload):
    if not _LONG_DIGITS.search(payload):
        return False
    for match in _JSON_TOKENS.finditer(payload):
        number = match.group(1)
        if number is None or b"." in number or b"e" in number or b"E" in number:
            continue
        negative = number.startswith(b"-")
        magnitude = number[1:] if negative else number
        limit = _SIGNED_MIN_MAGNITUDE if negative else _UNSIGNED_MAX
        if len(magnitude) > len(limit) or (len(magnitude) == len(limit) and magnitude > limit):
            return True
    return False

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
        if isinstance(payload, (bytes, bytearray, memoryview)) and _requires_exact_integer_decoder(payload):
            # stdlib accepts arbitrary precision integers subject to Python's
            # configured integer-string limit, instead of rounding to Float64.
            return _stdlib_json.loads(bytes(payload))
        return _loads(payload)
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    return _loads(payload)


def json_dumps(payload):
    value = _dumps(payload)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value
