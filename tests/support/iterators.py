"""Fresh result iterator builders without eager analytical imports."""

from types import SimpleNamespace

from tests.fixtures.compatibility import FixtureSession, fixture_bytes

ROOT = "https://offline.example/service"
VIEWS = ["Employee.name", "Employee.age", "Employee.fullTime"]


def make_iterator(mode="dict", *, version=8, payload=None, decimal_paths=(), compatibility="native"):
    from intermine314.service import session as runtime
    if payload is None:
        payload = fixture_bytes("rows-modern.json" if version >= 8 else "rows-legacy.json")
    session = FixtureSession({("POST", "/service/query/results"): payload})
    opener = runtime.InterMineURLOpener(session=session, timeout=(2, 7), verify_tls="ca.pem", user_agent="row-test")
    service = SimpleNamespace(root=ROOT, version=version, opener=opener, compatibility=compatibility)
    iterator = runtime.ResultIterator(service, "/query/results", {"query": "xml"}, mode, list(VIEWS), decimal_paths=decimal_paths)
    return iterator, session, opener

