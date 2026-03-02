from unittest.mock import patch

from intermine314.service import Service


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


class TestServiceModelLoading:
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

        assert first is second
        assert len(created) == 1
        assert created[0][0] == xml
        assert created[0][1] is service
        assert opener.calls == [("https://example.org/service/model", None)]

