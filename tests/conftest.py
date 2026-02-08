from __future__ import annotations

from pathlib import Path

from tests.live_test_config import LIVE_TESTS_ENABLED


def pytest_ignore_collect(collection_path, config):  # pragma: no cover - pytest hook
    del config
    if LIVE_TESTS_ENABLED:
        return False
    path = Path(str(collection_path))
    return path.name.startswith("live_")
