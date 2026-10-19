"""Saved-query route builders and immutable fixture constants."""
from urllib.parse import quote

from tests.fixtures.compatibility import FixtureSession, fixture_bytes

TOKEN = 'secret+&人'
NAME = 'a/+&人'
XML = '<query name="a/+&amp;人" model="genomic" view="Gene.symbol"><constraint value="A+B&amp;人"/></query>'
NOTE = 'Note: name should contain no special symbol            and should be defined first\n'
OPTIONS = dict(request_timeout=13, proxy_url='socks5h://127.0.0.1:9050', tor=True,
               verify_tls='/custom/ca.pem', user_agent='saved-query-client')

def helper_session(queries=None, version=27):
    return FixtureSession({
        ('GET', '/service/instances'): fixture_bytes('registry-helpers.json'),
        ('GET', '/service/instances/OfflineMine'): fixture_bytes('registry-helpers.json'),
        ('GET', '/custom/service/user/queries'): fixture_bytes('saved-queries.json') if queries is None else queries,
        ('GET', '/custom/service/version'): str(version).encode(),
        ('GET', '/custom/service/version/ws'): b'30',
        ('GET', '/service/version/ws'): b'30',
        ('PUT', '/custom/service/user/queries'): b'',
        ('DELETE', '/custom/service/user/queries/' + quote(NAME, safe='')): b'',
    })

