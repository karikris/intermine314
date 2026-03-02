from __future__ import annotations

import os

import pytest

from intermine314.service import Service
from tests.live_test_config import LIVE_TEST_ROOT, require_live_tests


def _require_tor_live_tests() -> None:
    if os.getenv("INTERMINE314_RUN_TOR_LIVE_TESTS", "").strip() != "1":
        pytest.skip("Tor live tests are disabled. Set INTERMINE314_RUN_TOR_LIVE_TESTS=1 to enable.")


@pytest.mark.allow_network
def test_live_tor_smoke_version_endpoint():
    require_live_tests()
    _require_tor_live_tests()
    with Service(LIVE_TEST_ROOT, tor=True) as service:
        version = service.version
    assert isinstance(version, int)
    assert version > 0
