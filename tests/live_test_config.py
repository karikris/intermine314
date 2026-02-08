from __future__ import annotations

import os
import unittest


LIVE_TEST_ENV_VAR = "INTERMINE314_RUN_LIVE_TESTS"
TESTMODEL_URL_ENV_VAR = "TESTMODEL_URL"
DEFAULT_TESTMODEL_URL = "http://localhost:8080/intermine-demo/service"

LIVE_TESTS_ENABLED = os.getenv(LIVE_TEST_ENV_VAR, "").strip() == "1"
LIVE_TEST_ROOT = os.getenv(TESTMODEL_URL_ENV_VAR, DEFAULT_TESTMODEL_URL)
LIVE_TEST_SKIP_MESSAGE = f"Live tests are disabled. Set {LIVE_TEST_ENV_VAR}=1 to enable."


def require_live_tests():
    if not LIVE_TESTS_ENABLED:
        raise unittest.SkipTest(LIVE_TEST_SKIP_MESSAGE)
