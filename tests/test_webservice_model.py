import unittest
from unittest.mock import patch

from intermine314.webservice import Service


class _FakeOpener:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def read(self, url, data=None):
        self.calls.append((url, data))
        return self.payload


class _FakeListManager:
    def delete_temporary_lists(self):
        return None


class TestServiceModelLoading(unittest.TestCase):
    @staticmethod
    def _make_service(opener):
        service = Service.__new__(Service)
        service.root = "https://example.org/service"
        service._model = None
        service.opener = opener
        service._list_manager = _FakeListManager()
        return service

    def test_model_reads_xml_via_opener_and_caches(self):
        xml = '<model name="mock" package="org.mock"></model>'
        opener = _FakeOpener(xml)
        created = []

        class _FakeModel:
            def __init__(self, source, service):
                created.append((source, service))
                self.source = source
                self.service = service

        service = self._make_service(opener)
        with patch("intermine314.service.service.Model", _FakeModel):
            first = service.model
            second = service.model

        self.assertIs(first, second)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0][0], xml)
        self.assertIs(created[0][1], service)
        self.assertEqual(opener.calls, [("https://example.org/service/model", None)])


if __name__ == "__main__":
    unittest.main()
