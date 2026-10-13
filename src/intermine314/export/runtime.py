"""Cooperative, process-wide native admission with explicit resource ownership.

Reservations coordinate package work. Existing external pools and unrelated
application work remain outside this advisory budget.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from importlib import import_module

from intermine314.export.resource_profile import resolve_resource_profile
from intermine314.parallel.policy import require_positive_int

_LOG = logging.getLogger(__name__)
_CURRENT = ContextVar('intermine314_resource_runtime', default=None)
_RESERVATION = ContextVar('intermine314_native_reservation', default=None)
_ENGINE_LOCK = threading.RLock()
_WARNED = set()


class _Admission:
    def __init__(self):
        self.condition = threading.Condition()
        self.waiters = deque()
        self.active = {}
        self.peak = 0

    @contextmanager
    def reserve(self, requested, budget, cancelled):
        ticket = object()
        charge = min(requested, budget)
        admitted = False
        with self.condition:
            self.waiters.append(ticket)
            try:
                while True:
                    if cancelled.is_set():
                        raise RuntimeError('ResourceRuntime is closing')
                    limit = min([budget, *[limit for _, limit in self.active.values()]])
                    used = sum(cost for cost, _ in self.active.values())
                    if self.waiters[0] is ticket and used + charge <= limit:
                        self.waiters.popleft()
                        self.active[ticket] = (charge, budget)
                        self.peak = max(self.peak, used + charge)
                        admitted = True
                        self.condition.notify_all()
                        break
                    self.condition.wait(timeout=0.1)
            finally:
                if not admitted:
                    self.waiters.remove(ticket)
                    self.condition.notify_all()
        try:
            yield charge
        finally:
            with self.condition:
                del self.active[ticket]
                self.condition.notify_all()

    def diagnostics(self):
        with self.condition:
            return dict(active_native_reservations=sum(cost for cost, _ in self.active.values()),
                        waiting_native_jobs=len(self.waiters), peak_native_reservations=self.peak)


_ADMISSION = _Admission()
_DEFAULTS_LOCK = threading.Lock()
_DEFAULTS = {}


class ResourceRuntime:
    """Own package resources; share native admission across all runtimes.

    Pass this runtime to analytical APIs and close it after consumers finish.
    Closing waits for active calls. Externally initialized engines are never
    silently resized; requested/effective pool sizes are reported instead.
    """

    def __init__(self, resource_profile=None):
        self.profile = resolve_resource_profile(resource_profile)
        self.cpu_budget = self.profile.effective_cpu_budget
        self.memory_budget_bytes = self.profile.memory_envelope
        self._condition = threading.Condition()
        self._calls = 0
        self._closed = False
        self._closing = threading.Event()
        self._engines = {}
        self._owned = []

    @contextmanager
    def activate(self):
        with self._condition:
            if self._closing.is_set():
                raise RuntimeError('ResourceRuntime is closed')
            self._calls += 1
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)
            with self._condition:
                self._calls -= 1
                self._condition.notify_all()

    def _report_pool(self, engine, requested, effective):
        self._engines[engine] = dict(requested=requested, effective=effective)
        key = (engine, requested, effective, self.cpu_budget)
        if effective > requested or effective > self.cpu_budget:
            if key not in _WARNED:
                _WARNED.add(key)
                _LOG.warning('%s pool has %s threads; requested=%s, advisory CPU budget=%s. '
                             'Configure the engine before import for matching limits.',
                             engine, effective, requested, self.cpu_budget)

    def engine(self, name):
        """Initialize only on demand, and never resize an existing pool."""
        if name not in {'polars', 'pyarrow', 'duckdb'}:
            raise ValueError(f'Unknown analytics engine: {name}')
        with _ENGINE_LOCK:
            if self._closing.is_set():
                raise RuntimeError('ResourceRuntime is closed')
            existing = name in sys.modules
            if name == 'polars' and not existing:
                os.environ.setdefault('POLARS_MAX_THREADS', str(self.profile.polars_threads or self.profile.workers_for('polars')))
            module = import_module(name)
            if name == 'polars':
                requested = self.profile.polars_threads or self.profile.workers_for('polars')
                self._report_pool(name, requested, module.thread_pool_size())
            elif name == 'pyarrow':
                cpu = self.profile.arrow_cpu_threads or self.profile.workers_for('arrow_cpu')
                io = self.profile.arrow_io_threads or self.profile.workers_for('arrow_io')
                if not existing:
                    module.set_cpu_count(cpu)
                    module.set_io_thread_count(io)
                self._report_pool(name, cpu, module.cpu_count())
                self._engines['arrow_io'] = dict(requested=io, effective=module.io_thread_count())
            return module

    @contextmanager
    def native(self, stage, *, threads=None):
        """Reserve around one native operation, never around an entire pipeline."""
        if _RESERVATION.get() is not None:
            raise RuntimeError('Native reservations cannot nest; release the prior stage first')
        with self.activate():
            if stage == 'polars':
                count = self.engine('polars').thread_pool_size()
            elif stage == 'arrow_cpu':
                count = self.engine('pyarrow').cpu_count()
            else:
                count = threads if threads is not None else self.profile.workers_for(stage)
            count = require_positive_int('native threads', count)
            with _ADMISSION.reserve(count, self.cpu_budget, self._closing) as charge:
                token = _RESERVATION.set((stage, charge))
                try:
                    yield
                finally:
                    _RESERVATION.reset(token)

    def duckdb_config(self, *, stage='duckdb_sql', temp_dir=None, ordered=True):
        config = {
            'threads': self.profile.duckdb_threads or self.profile.workers_for(stage),
            'memory_limit': f'{max(1, self.memory_budget_bytes * 3 // 5)}B',
            'preserve_insertion_order': bool(ordered), 'TimeZone': 'UTC',
        }
        if temp_dir is not None or self.profile.temp_dir is not None:
            config['temp_directory'] = str(temp_dir if temp_dir is not None else self.profile.temp_dir)
        return config

    def diagnostics(self):
        with _ENGINE_LOCK:
            engines = {name: dict(values) for name, values in self._engines.items()}
        return dict(profile=asdict(self.profile), effective_cpu_budget=self.cpu_budget,
                    memory_budget_bytes=self.memory_budget_bytes, engines=engines,
                    advisory=True, **_ADMISSION.diagnostics())

    def own(self, resource):
        """Register a closeable package-owned resource for deterministic teardown."""
        with self._condition:
            if self._closing.is_set():
                raise RuntimeError('ResourceRuntime is closed')
            self._owned.append(resource)
        return resource

    def close(self):
        if _CURRENT.get() is self:
            raise RuntimeError('Cannot close a runtime from one of its active operations')
        with self._condition:
            if self._closed:
                return
            self._closing.set()
            while self._calls:
                self._condition.wait()
            resources, self._owned = self._owned, []
            self._closed = True
        error = None
        for resource in reversed(resources):
            try:
                resource.close()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def __enter__(self):
        if self._closing.is_set():
            raise RuntimeError('ResourceRuntime is closed')
        return self

    def __exit__(self, *exc):
        self.close()


def get_runtime(resource_profile=None, runtime=None):
    if runtime is not None:
        if not isinstance(runtime, ResourceRuntime):
            raise TypeError('runtime must be a ResourceRuntime')
        if resource_profile is not None and resolve_resource_profile(resource_profile) != runtime.profile:
            raise ValueError('resource_profile conflicts with runtime.profile')
        return runtime
    current = _CURRENT.get()
    if resource_profile is None and current is not None:
        return current
    profile = resolve_resource_profile(resource_profile)
    with _DEFAULTS_LOCK:
        runtime = _DEFAULTS.get(profile)
        if runtime is None or runtime._closing.is_set():
            runtime = _DEFAULTS[profile] = ResourceRuntime(profile)
        return runtime


def shutdown_resources():
    """Close implicit package runtimes; explicit runtimes retain caller ownership."""
    with _DEFAULTS_LOCK:
        runtimes = list(_DEFAULTS.values())
        _DEFAULTS.clear()
    for runtime in runtimes:
        runtime.close()
