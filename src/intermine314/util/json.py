from __future__ import annotations

import json

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None


def json_loads(payload):
    if orjson is not None:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return orjson.loads(payload)
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def json_dumps(payload):
    if orjson is not None:
        return orjson.dumps(payload).decode("utf-8")
    return json.dumps(payload)
