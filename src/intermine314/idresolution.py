"""Identifier-resolution jobs over the service's configured managed opener."""

import heapq
import json
import math
import time
import weakref
from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import ExitStack

from intermine314.export.runtime import get_runtime, resource_operation
from intermine314.parallel.policy import require_positive_int
from intermine314.service.transport import open_readonly
from intermine314.util.ownership import owner_state, serialized_owner

__all__ = ['Job', 'get_json', 'poll_jobs']

ONE_MINUTE = 60
COMPLETED = {'SUCCESS', 'ERROR'}


def get_json(service, path, key):
    """Read a job response, preserving server and missing-key exceptions."""
    with open_readonly(service.opener, service.root + path) as response:
        data = json.loads(response.read())
    if data['error'] is not None:
        raise Exception(data['error'])
    if key not in data:
        raise Exception(key + ' not returned from ' + path)
    return data[key]


class Job:
    """A resolution job whose service lifetime remains caller-controlled.

    Polling caps the delay between checks, without imposing an attempt limit.
    SUCCESS and ERROR are terminal; repeated terminal polls have no side effects.
    """

    INITIAL_DECAY = 1.25
    INITIAL_BACKOFF = .05
    MAX_BACKOFF = ONE_MINUTE

    def __init__(self, service, uid):
        self.service = weakref.proxy(service)
        self.uid = uid
        self.status = None
        self.backoff = Job.INITIAL_BACKOFF
        self.decay = Job.INITIAL_DECAY
        self.max_backoff = Job.MAX_BACKOFF
        if self.uid is None:
            raise Exception('No uid found')

    def _next_delay(self):
        delay = self.backoff
        self.backoff = min(self.max_backoff, delay * self.decay)
        return delay

    @serialized_owner
    def poll(self):
        """Wait, fetch status, and return whether the job is complete."""
        if self.status not in COMPLETED:
            time.sleep(self._next_delay())
            self.status = self.fetch_status()
        return self.status in COMPLETED

    def fetch_status(self):
        """Return current server status without changing cached poll status."""
        return get_json(self.service, f'/ids/{self.uid}/status', 'status')

    def fetch_results(self):
        """Return the server's resolution results."""
        return get_json(self.service, f'/ids/{self.uid}/result', 'results')

    def delete(self):
        """Delete the remote job, returning None after server error validation."""
        path = '/ids/' + self.uid
        data = json.loads(self.service.opener.delete(self.service.root + path))
        if data['error'] is not None:
            raise Exception(data['error'])


@resource_operation
def poll_jobs(jobs, *, max_workers=None, timeout=None, cancel_event=None, resource_profile=None, runtime=None):
    """Wait for jobs with one monotonic due queue and bounded HTTP workers.

    Returns the original jobs in input order, including duplicates. SUCCESS and
    ERROR are terminal. Workers only fetch due statuses; the calling coordinator
    owns delays, backoff and status updates. Timeout/cancellation raises without
    deleting remote jobs. Already running requests finish under their transport
    timeouts before local cleanup completes. Keep Services alive throughout.
    """
    active = get_runtime()
    workers = active._workers_for('metadata') if max_workers is None else require_positive_int('max_workers', max_workers)
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('timeout must be a positive finite number')
    if cancel_event is not None and not all(callable(getattr(cancel_event, name, None)) for name in ('is_set', 'wait')):
        raise TypeError('cancel_event must provide is_set() and wait()')
    items = list(jobs)
    if any(not isinstance(job, Job) for job in items):
        raise TypeError('poll_jobs requires Job inputs')
    unique = {id(job): job for job in items}
    deadline = None if timeout is None else time.monotonic() + timeout

    def check():
        if active._closing.is_set() or (cancel_event is not None and cancel_event.is_set()):
            from concurrent.futures import CancelledError

            raise CancelledError('Job polling cancelled')
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Job polling timed out')

    with ExitStack() as ownership:
        # Stable ordering prevents competing coordinators from deadlocking.
        for identity in sorted(unique):
            lock = owner_state(unique[identity]).lock
            check()
            while not lock.acquire(timeout=.1):
                check()
            ownership.callback(lock.release)
        due = []
        sequence = 0
        for job in unique.values():
            if job.status not in COMPLETED:
                heapq.heappush(due, (time.monotonic() + job._next_delay(), sequence, job))
                sequence += 1
        with active.executor(workers, 'intermine314-job-status') as pool:
            pending = {}
            while due or pending:
                check()
                now = time.monotonic()
                while due and due[0][0] <= now and len(pending) < workers:
                    _, _, job = heapq.heappop(due)
                    pending[pool.submit(job.fetch_status)] = job
                ready = {future for future in pending if future.done()}
                if ready:
                    for future in ready:
                        job = pending.pop(future)
                        job.status = future.result()
                        if job.status not in COMPLETED:
                            heapq.heappush(due, (time.monotonic() + job._next_delay(), sequence, job))
                            sequence += 1
                    continue
                delay = .1
                if due and len(pending) < workers:
                    delay = min(delay, max(0, due[0][0] - time.monotonic()))
                if deadline is not None:
                    delay = min(delay, max(0, deadline - time.monotonic()))
                if pending:
                    wait(pending, timeout=delay, return_when=FIRST_COMPLETED)
                elif cancel_event is not None:
                    cancel_event.wait(delay)
                else:
                    time.sleep(delay)
    return items
