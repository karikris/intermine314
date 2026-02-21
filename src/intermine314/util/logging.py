from __future__ import annotations

import json
import logging
import warnings
from uuid import uuid4

try:
    import orjson as _orjson
except Exception:  # pragma: no cover - optional acceleration
    _orjson = None

_SENSITIVE_FIELD_TOKENS = (
    "authorization",
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
)
_TRUNCATED_SUFFIX = "...<truncated>"
_MAX_LOG_STRING_CHARS = 2048
JOB_ID_LEN = 12
_CONFIGURE_LOGGING_WARNED = False


def _is_sensitive_key(key: str) -> bool:
    lowered = str(key).strip().lower()
    return any(token in lowered for token in _SENSITIVE_FIELD_TOKENS)


def _sanitize_log_value(key: str, value):
    if _is_sensitive_key(key):
        return "<redacted>"
    if isinstance(value, str) and len(value) > _MAX_LOG_STRING_CHARS:
        return value[:_MAX_LOG_STRING_CHARS] + _TRUNCATED_SUFFIX
    return value


def _serialize_structured_payload(payload: dict[str, object]) -> str:
    if _orjson is not None:
        try:
            return _orjson.dumps(payload, default=str).decode("utf-8")
        except Exception:
            # Keep logging best-effort even for unusual payload objects.
            pass
    return json.dumps(payload, default=str, ensure_ascii=False, separators=(",", ":"))


def new_job_id(prefix: str | None = None) -> str:
    token = uuid4().hex[:JOB_ID_LEN]
    cleaned = str(prefix or "").strip()
    if cleaned:
        return f"{cleaned}_{token}"
    return token


def log_structured_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields,
) -> dict[str, object]:
    payload: dict[str, object] = {"event": str(event)}
    payload.update({k: _sanitize_log_value(k, v) for k, v in fields.items() if v is not None})
    logger.log(level, _serialize_structured_payload(payload))
    return payload


def configure_logging(level: int = logging.INFO) -> None:
    global _CONFIGURE_LOGGING_WARNED
    _ = level
    if not _CONFIGURE_LOGGING_WARNED:
        _CONFIGURE_LOGGING_WARNED = True
        warnings.warn(
            "intermine314.util.logging.configure_logging() is deprecated and is now a no-op. "
            "Configure logging in application/CLI entrypoints instead.",
            DeprecationWarning,
            stacklevel=2,
        )
