"""Registry discovery routes independent of test modules."""
from tests.fixtures.compatibility import FixtureSession, fixture_bytes


def helper_session(detail=None, rows=None):
    session = FixtureSession({
        ('GET', '/service/instances'): fixture_bytes('registry-helpers.json'),
        ('GET', '/service/instances/OfflineMine'): fixture_bytes('registry-helpers.json') if detail is None else detail,
        ('GET', '/custom/service/version/ws'): b'30',
        ('GET', '/custom/service/model'): fixture_bytes('datasets-model.xml'),
        ('POST', '/custom/service/query/results'): fixture_bytes('datasets-rows.json') if rows is None else rows,
    })
    return session

