from __future__ import annotations

import json
import logging
from uuid import uuid4


def new_job_id(prefix: str | None = None) -> str:
    token = uuid4().hex[:12]
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
    payload.update({k: v for k, v in fields.items() if v is not None})
    logger.log(level, json.dumps(payload, sort_keys=True, default=str))
    return payload


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level)
