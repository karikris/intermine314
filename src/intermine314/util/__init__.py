from __future__ import annotations

from intermine314.util.json import json_dumps, json_loads
from intermine314.util.logging import log_structured_event, new_job_id
from intermine314.util.timing import timed


class ReadableException(Exception):
    def __init__(self, message, cause=None):
        self.message = message
        self.cause = cause

    def __str__(self):
        if self.cause is None:
            return repr(self.message)
        return repr(self.message) + repr(self.cause)


__all__ = [
    "ReadableException",
    "new_job_id",
    "log_structured_event",
    "timed",
    "json_loads",
    "json_dumps",
]
