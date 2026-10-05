from __future__ import annotations

import json as _stdlib_json
import logging
import re
from math import isfinite

try:
    import orjson as _orjson
except Exception:  # pragma: no cover
    _orjson = None

LOG = logging.getLogger(__name__)
orjson = _orjson

# The cheap first pass avoids tokenizing ordinary payloads. The second pass
# consumes complete strings/numbers so digit strings and floats cannot trigger
# an integer fallback. Compare lexemes instead of converting enormous ints.
_DIGIT_MASK = bytes(1 if 48 <= value <= 57 else 0 for value in range(256))
_LONG_DIGITS = b"\x01" * 19
_JSON_TOKENS = re.compile(rb'"(?:[^"\\]|\\.)*"|(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')
_SIGNED_MIN_MAGNITUDE = b"9223372036854775808"
_UNSIGNED_MAX = b"18446744073709551615"


def _requires_exact_integer_decoder(payload):
    # bytes.translate and substring search run in C; a regex search at every
    # possible digit position costs more than JSON decoding for small rows.
    payload = bytes(payload)
    if _LONG_DIGITS not in payload.translate(_DIGIT_MASK):
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
    try:
        value = _dumps(payload)
    except TypeError:
        # Only plain JSON containers qualify: do not broaden orjson's key/type
        # coercions merely because an unrelated integer exceeds its range.
        if _JSON_BACKEND != "orjson" or not _plain_json_with_large_integer(payload):
            raise
        value = _stdlib_json.dumps(payload)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _plain_json_with_large_integer(payload):
    pending, active, large = [(payload, False)], set(), False
    while pending:
        value, exiting = pending.pop()
        if exiting:
            active.remove(id(value))
            continue
        if isinstance(value, int):
            large |= value < -(2**63) or value >= 2**64
        elif isinstance(value, float):
            if not isfinite(value):
                return False
        elif value is None or isinstance(value, str):
            continue
        elif isinstance(value, (dict, list, tuple)):
            if id(value) in active:
                return False
            active.add(id(value))
            pending.append((value, True))
            if isinstance(value, dict):
                if not all(isinstance(key, str) for key in value):
                    return False
                pending.extend((item, False) for item in value.values())
            else:
                pending.extend((item, False) for item in value)
        else:
            return False
    return large
