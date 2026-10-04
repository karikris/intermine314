"""Compressed XML discovery through the actual configured HTTP adapter."""

import pytest
import requests

from intermine314.service import Service
from intermine314.webservice import Service as LegacyService
from tests.fixtures.compatibility import fixture_bytes
from tests.fixtures.compatibility.http import ENCODERS, loopback_service


class RecordingSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.trust_env = False
        self.calls = []
        self.responses = []
        self.close_calls = 0

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs.copy()))
        response = super().request(method, url, **kwargs)
        close = response.close
        response.close_calls = 0

        def close_response():
            response.close_calls += 1
            close()

        response.close = close_response
        self.responses.append(response)
        return response

    def close(self):
        self.close_calls += 1
        super().close()


@pytest.mark.parametrize("service_class", [Service, LegacyService])
@pytest.mark.parametrize("encoding", ENCODERS)
@pytest.mark.parametrize("via_proxy", [False, True])
def test_compressed_template_discovery_retains_transport_configuration_and_cache(
    service_class, encoding, via_proxy,
):
    templates = fixture_bytes("templates.xml").replace(
        b"View all the employees with certain name", "Müller 🧬".encode(),
    )
    routes = {
        ("GET", "/service/version/ws"): b"30",
        ("GET", "/service/model"): (fixture_bytes("model.xml"), encoding),
        ("GET", "/service/templates"): (templates, encoding),
        ("GET", "/service/metadata"): ('<metadata name="Müller 🧬"/>'.encode(), encoding),
    }
    with loopback_service(routes) as (base_url, received), RecordingSession() as session:
        root = "http://mine.example/service" if via_proxy else base_url + "/service"
        proxy = base_url if via_proxy else None
        if via_proxy:
            session.proxies = {"http": proxy, "https": proxy}
        with service_class(
            root, session=session, token="fixture-token", request_timeout=3,
            verify_tls=False, proxy_url=proxy, user_agent="compression-test",
        ) as service:
            snapshot = service.templates
            assert "Müller 🧬" in snapshot["employeeByName"]
            assert service.templates is snapshot
            template = service.get_template("employeeByName")
            assert template.service is service and template.compatibility == service.compatibility
            assert service.get_template("employeeByName") is template
            assert service._get_xml("/metadata").documentElement.getAttribute("name") == "Müller 🧬"
            assert service.opener._session is session and service.opener.proxy_url == proxy
            auth = service.opener.auth_header
        assert session.close_calls == 0
        assert [call.path for call in received] == [
            "/service/version/ws", "/service/templates", "/service/model", "/service/metadata",
        ]
        assert all(response.close_calls == 1 and response.raw.closed for response in session.responses)
        for _, _, options in session.calls:
            assert options["stream"] is True
            assert options["timeout"] == (3.0, 3.0) and options["verify"] is False
        for call in received:
            assert call.headers["Authorization"] == auth
            assert call.headers["User-Agent"] == "compression-test"
        assert received[1].headers["Accept"] == received[3].headers["Accept"] == "application/xml"
        with session.get(root + "/version/ws", timeout=3) as response:
            assert response.content == b"30"
