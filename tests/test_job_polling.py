import threading
from concurrent.futures import CancelledError
from types import SimpleNamespace

import pytest

from intermine314.idresolution import Job, poll_jobs
from tests.test_idresolution import job_session


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.sleeps.append((threading.get_ident(), seconds))
        self.now += seconds


def setup(monkeypatch):
    import intermine314.idresolution as module
    from intermine314.service import Service
    from tests.fixtures.compatibility import SERVICE_ROOT

    clock = Clock()
    monkeypatch.setattr(module, 'time', clock)
    service = Service(SERVICE_ROOT, session=job_session())
    return clock, service


def test_coordinator_sleeps_once_for_many_jobs_and_workers_only_fetch(monkeypatch):
    clock, service = setup(monkeypatch)
    jobs = [Job(service, str(i)) for i in range(8)]
    counts, times, workers = {}, [], set()
    lock = threading.Lock()
    def fetch(job):
        with lock:
            workers.add(threading.get_ident())
            times.append(clock.now)
            counts[job.uid] = counts.get(job.uid, 0) + 1
            return 'SUCCESS' if counts[job.uid] == 2 else 'PENDING'
    for job in jobs:
        monkeypatch.setattr(job, 'fetch_status', lambda job=job: fetch(job))
    result = poll_jobs(jobs, max_workers=2)
    assert all(before is after for before, after in zip(jobs, result))
    assert all(count == 2 for count in counts.values())
    assert all(job.status == 'SUCCESS' and job.backoff == .078125 for job in jobs)
    assert min(times) >= .05
    assert all(owner == threading.get_ident() for owner, _ in clock.sleeps)
    assert 1 <= len(workers) <= 2 and threading.get_ident() not in workers
    assert sum(delay for _, delay in clock.sleeps) == pytest.approx(.1125)
    service.close()


def test_terminal_jobs_and_duplicate_inputs_are_not_repolled(monkeypatch):
    _, service = setup(monkeypatch)
    success, error = Job(service, 'a'), Job(service, 'b')
    success.status, error.status = 'SUCCESS', 'ERROR'
    def forbidden():
        pytest.fail('Terminal jobs must not fetch')
    monkeypatch.setattr(success, 'fetch_status', forbidden)
    monkeypatch.setattr(error, 'fetch_status', forbidden)
    assert poll_jobs([success, error, success]) == [success, error, success]
    assert success.backoff == error.backoff == .05
    service.close()


def test_timeout_uses_monotonic_deadline_and_keeps_remote_job(monkeypatch):
    clock, service = setup(monkeypatch)
    job = Job(service, 'pending')
    calls = []
    monkeypatch.setattr(job, 'fetch_status', lambda: calls.append(clock.now) or 'PENDING')
    with pytest.raises(TimeoutError):
        poll_jobs([job], timeout=.1)
    assert clock.now == pytest.approx(.1)
    assert calls == [.05] and job.status == 'PENDING'
    assert not any(request.method == 'DELETE' for request in service.opener._session.requests)
    service.close()


def test_cancellation_wakes_coordinator_before_first_due_poll(monkeypatch):
    clock, service = setup(monkeypatch)
    job = Job(service, 'cancelled')
    cancelled = False
    def wait(delay):
        nonlocal cancelled
        clock.sleep(delay / 2)
        cancelled = True
    event = SimpleNamespace(is_set=lambda: cancelled, wait=wait)
    def forbidden():
        pytest.fail('Cancellation must precede the first due fetch')
    monkeypatch.setattr(job, 'fetch_status', forbidden)
    with pytest.raises(CancelledError):
        poll_jobs([job], cancel_event=event)
    assert job.status is None
    service.close()


def test_fetch_failure_releases_job_owner_and_can_retry(monkeypatch):
    _, service = setup(monkeypatch)
    job = Job(service, 'retry')
    def fail():
        raise RuntimeError('status failed')
    monkeypatch.setattr(job, 'fetch_status', fail)
    with pytest.raises(RuntimeError, match='status failed'):
        poll_jobs([job])
    monkeypatch.setattr(job, 'fetch_status', lambda: 'SUCCESS')
    assert poll_jobs([job]) == [job] and job.status == 'SUCCESS'
    service.close()


@pytest.mark.parametrize('kwargs', [{'timeout': 0}, {'timeout': True}, {'timeout': float('nan')},
                                    {'max_workers': 0}, {'cancel_event': object()}])
def test_invalid_coordinator_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        poll_jobs([], **kwargs)
