from __future__ import annotations

import pytest

from intermine314.service import Registry
from tests.live_test_config import require_live_tests


@pytest.mark.allow_network
def test_live_registry_smoke():
    require_live_tests()
    with Registry() as registry:
        mine_names = registry.keys()
        assert isinstance(mine_names, list)
        assert mine_names, "registry returned no mines"
        info = registry.info(str(mine_names[0]))
    assert isinstance(info, dict)
    assert str(info.get("name", "")).strip()
    assert str(info.get("url", "")).strip()
