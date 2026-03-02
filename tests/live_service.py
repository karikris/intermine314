from __future__ import annotations

import pytest

from intermine314.service import Service
from tests.live_test_config import LIVE_TEST_ROOT, require_live_tests


@pytest.mark.allow_network
def test_live_service_smoke_version_endpoint():
    require_live_tests()
    with Service(LIVE_TEST_ROOT) as service:
        version = service.version
    assert isinstance(version, int)
    assert version > 0
