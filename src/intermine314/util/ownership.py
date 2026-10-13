"""Process-local coordination for mutable clients and publication targets."""

from __future__ import annotations

import threading
import weakref
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

_GUARD = threading.Lock()
_TARGETS = weakref.WeakValueDictionary()


class OwnerState:
    def __init__(self):
        self.lock = threading.RLock()
        self.managers = weakref.WeakSet()


def owner_state(owner):
    # Attribute storage also works through the legacy weak service proxies.
    with _GUARD:
        state = getattr(owner, '_mutation_state', None)
        if state is None:
            state = OwnerState()
            owner._mutation_state = state
        return state


def serialized_owner(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with owner_state(self).lock:
            return function(self, *args, **kwargs)
    return wrapped


@contextmanager
def publication_owner(target):
    """Hold one target through preservation, replacement and rollback."""
    key = str(Path(target).resolve())
    with _GUARD:
        state = _TARGETS.get(key)
        if state is None:
            state = _TARGETS[key] = OwnerState()
    with state.lock:
        yield


def serialized_publication(function):
    @wraps(function)
    def wrapped(staged, target, backup):
        with publication_owner(target):
            return function(staged, target, backup)
    return wrapped
