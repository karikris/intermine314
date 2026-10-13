"""Per-client, per-key metadata initialization and lifecycle barriers."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from threading import Condition, RLock, get_ident

# Allocation can run Service finalizers on this same thread during GC.
_GUARD = RLock()


class MetadataState:
    def __init__(self):
        self.condition = Condition(RLock())
        self.keys = {}
        self.active = {}
        self.writer = None
        self.waiting_writers = 0

    @contextmanager
    def read(self, key):
        thread = get_ident()
        with self.condition:
            while self.writer not in (None, thread) or (self.waiting_writers and thread not in self.active and self.writer != thread):
                self.condition.wait()
            self.active[thread] = self.active.get(thread, 0) + 1
            lock = self.keys.setdefault(key, RLock())
        try:
            with lock:
                yield
        finally:
            with self.condition:
                self.active[thread] -= 1
                if not self.active[thread]:
                    del self.active[thread]
                self.condition.notify_all()

    @contextmanager
    def write(self):
        thread = get_ident()
        with self.condition:
            if thread in self.active:
                raise RuntimeError('Cannot invalidate or close metadata during its initialization')
            nested = self.writer == thread
            if not nested:
                self.waiting_writers += 1
                try:
                    while self.active or self.writer is not None:
                        self.condition.wait()
                    self.writer = thread
                finally:
                    self.waiting_writers -= 1
                    self.condition.notify_all()
        try:
            yield
        finally:
            if not nested:
                with self.condition:
                    self.writer = None
                    self.condition.notify_all()


def metadata_state(owner):
    with _GUARD:
        namespace = vars(owner)
        if '_metadata_state' not in namespace:
            namespace['_metadata_state'] = MetadataState()
        return namespace['_metadata_state']


def metadata_key(key):
    """Single initializer for a cache group; failures remain retryable."""
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            with metadata_state(self).read(key):
                return function(self, *args, **kwargs)
        return wrapped
    return decorate


def metadata_write(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with metadata_state(self).write():
            return function(self, *args, **kwargs)
    return wrapped
