"""Identifier-resolution jobs over the service's configured managed opener."""

import json
import time
import weakref

from intermine314.service.transport import open_readonly

__all__ = ['Job', 'get_json']

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

    def poll(self):
        """Wait, fetch status, and return whether the job is complete."""
        if self.status not in COMPLETED:
            backoff = self.backoff
            self.backoff = min(self.max_backoff, backoff * self.decay)
            time.sleep(backoff)
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
