import json
import logging
import unittest

from intermine314.util.logging import log_structured_event, new_job_id


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class TestStructuredLogging(unittest.TestCase):
    def test_new_job_id_prefix(self):
        job_id = new_job_id("bench")
        self.assertTrue(job_id.startswith("bench_"))
        self.assertGreater(len(job_id), len("bench_"))

    def test_log_structured_event_emits_json_message(self):
        logger = logging.getLogger("intermine314.tests.structured_logging")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = _CaptureHandler()
        logger.handlers = [handler]

        payload = log_structured_event(
            logger,
            logging.INFO,
            "export_start",
            job_id="job_123",
            ordered_mode="ordered",
            in_flight=4,
        )
        self.assertEqual(payload["event"], "export_start")
        self.assertEqual(payload["job_id"], "job_123")
        self.assertEqual(len(handler.messages), 1)

        decoded = json.loads(handler.messages[0])
        self.assertEqual(decoded["event"], "export_start")
        self.assertEqual(decoded["job_id"], "job_123")
        self.assertEqual(decoded["ordered_mode"], "ordered")
        self.assertEqual(decoded["in_flight"], 4)

    def test_log_structured_event_redacts_sensitive_fields(self):
        logger = logging.getLogger("intermine314.tests.structured_logging.redaction")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = _CaptureHandler()
        logger.handlers = [handler]

        payload = log_structured_event(
            logger,
            logging.INFO,
            "auth_event",
            access_token="abc123",
            authorization="Bearer secret",
            retries=2,
        )
        self.assertEqual(payload["access_token"], "<redacted>")
        self.assertEqual(payload["authorization"], "<redacted>")
        self.assertEqual(payload["retries"], 2)

        decoded = json.loads(handler.messages[0])
        self.assertEqual(decoded["access_token"], "<redacted>")
        self.assertEqual(decoded["authorization"], "<redacted>")

    def test_log_structured_event_truncates_large_strings(self):
        logger = logging.getLogger("intermine314.tests.structured_logging.truncation")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = _CaptureHandler()
        logger.handlers = [handler]

        huge = "x" * 3000
        payload = log_structured_event(logger, logging.INFO, "big_payload", preview=huge)

        self.assertLess(len(payload["preview"]), len(huge))
        self.assertTrue(payload["preview"].endswith("...<truncated>"))
        decoded = json.loads(handler.messages[0])
        self.assertEqual(decoded["preview"], payload["preview"])


if __name__ == "__main__":
    unittest.main()
