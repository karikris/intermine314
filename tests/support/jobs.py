"""Fresh identifier-job route builders."""
from tests.fixtures.compatibility import FixtureSession, fixture_bytes


def job_session(version=10):
    session = FixtureSession.service(version=version)
    session.routes.update({
        ('POST', '/service/ids'): fixture_bytes('job-created.json'),
        ('GET', '/service/ids/offline-job-1/status'): fixture_bytes('job-pending.json'),
        ('GET', '/service/ids/offline-job-1/result'): fixture_bytes('job-results.json'),
        ('DELETE', '/service/ids/offline-job-1'): b'{"error":null}',
    })
    return session

