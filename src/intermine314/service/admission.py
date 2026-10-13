"""Shared HTTP admission, held through response consumption and close."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit


def origin(url):
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in {'http', 'https'} or not parsed.hostname:
        raise ValueError('HTTP admission requires an HTTP(S) URL')
    return scheme, parsed.hostname.lower(), parsed.port or (443 if scheme == 'https' else 80)


def retry_after_seconds(value):
    if not value:
        return 0.0
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else 0.0
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            return max(0.0, (date - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0.0


class HostAdmission:
    def __init__(self):
        self.condition = threading.Condition()
        self.hosts = OrderedDict()
        self.active = {}

    def acquire(self, url, *, host_limit, global_limit=16, timeout=None, cancelled=None):
        key = origin(url)
        ticket = object()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            state = self.hosts.setdefault(key, dict(active={}, waiting=0, adaptive=None, cooldown=0.0, successes=0))
            self.hosts.move_to_end(key)
            state['waiting'] += 1
            try:
                while True:
                    now = time.monotonic()
                    if cancelled is not None and cancelled.is_set():
                        raise RuntimeError('ResourceRuntime is closing')
                    if deadline is not None and now >= deadline:
                        raise TimeoutError('Timed out waiting for HTTP admission')
                    local = min([host_limit, *state['active'].values()])
                    if state['adaptive'] is not None:
                        local = min(local, state['adaptive'])
                    total_limit = min([global_limit, *self.active.values()])
                    if now >= state['cooldown'] and len(state['active']) < local and len(self.active) < total_limit:
                        state['active'][ticket] = host_limit
                        self.active[ticket] = global_limit
                        break
                    self.condition.wait(timeout=min(0.1, max(0, deadline-now)) if deadline is not None else 0.1)
            finally:
                state['waiting'] -= 1
            # Bound idle-host metadata without evicting active or waiting state.
            for stale in list(self.hosts):
                if len(self.hosts) <= 512:
                    break
                item = self.hosts[stale]
                if not item['active'] and not item['waiting'] and item['cooldown'] <= time.monotonic():
                    del self.hosts[stale]

        def release():
            with self.condition:
                self.active.pop(ticket, None)
                state['active'].pop(ticket, None)
                self.condition.notify_all()

        def observe(response):
            headers = getattr(response, 'headers', {})
            history = getattr(getattr(getattr(response, 'raw', None), 'retries', None), 'history', ())
            throttled = response.status_code in {429, 503} or any(getattr(item, 'status', None) in {429, 503} for item in history)
            with self.condition:
                if throttled:
                    state['adaptive'] = max(1, min(host_limit, state['adaptive'] or host_limit) // 2)
                    state['successes'] = 0
                    state['cooldown'] = max(state['cooldown'], time.monotonic() + retry_after_seconds(headers.get('Retry-After')))
                elif response.status_code < 400 and state['adaptive'] is not None:
                    state['successes'] += 1
                    if state['successes'] >= 8:
                        state['adaptive'] = min(host_limit, state['adaptive'] + 1)
                        state['successes'] = 0
                self.condition.notify_all()
        return release, observe


HTTP_ADMISSION = HostAdmission()
