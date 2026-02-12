import intermine314.service.tor as tor
from intermine314.service import Registry, Service


class _FakeService:
    def __init__(self, root, **kwargs):
        self.root = root
        self.kwargs = kwargs


class _FakeRegistry:
    def __init__(self, registry_url, **kwargs):
        self.registry_url = registry_url
        self.kwargs = kwargs


def test_tor_service_helper_wires_proxy_and_session(monkeypatch):
    fake_session = object()

    def fake_tor_session(**_kwargs):
        return fake_session

    monkeypatch.setattr(tor, "tor_session", fake_tor_session)
    monkeypatch.setattr("intermine314.service.service.Service", _FakeService)

    service = tor.tor_service("https://example.org/service", token="abc", request_timeout=12)

    assert isinstance(service, _FakeService)
    assert service.root == "https://example.org/service"
    assert service.kwargs["token"] == "abc"
    assert service.kwargs["request_timeout"] == 12
    assert service.kwargs["proxy_url"] == "socks5h://127.0.0.1:9050"
    assert service.kwargs["session"] is fake_session
    assert service.kwargs["tor"] is True
    assert service.kwargs["allow_http_over_tor"] is False


def test_tor_registry_helper_wires_proxy_and_session(monkeypatch):
    fake_session = object()

    def fake_tor_session(**_kwargs):
        return fake_session

    monkeypatch.setattr(tor, "tor_session", fake_tor_session)
    monkeypatch.setattr("intermine314.service.service.Registry", _FakeRegistry)

    registry = tor.tor_registry(request_timeout=8, verify_tls=False)

    assert isinstance(registry, _FakeRegistry)
    assert registry.registry_url == "https://registry.intermine.org/service/instances"
    assert registry.kwargs["request_timeout"] == 8
    assert registry.kwargs["verify_tls"] is False
    assert registry.kwargs["proxy_url"] == "socks5h://127.0.0.1:9050"
    assert registry.kwargs["session"] is fake_session
    assert registry.kwargs["tor"] is True
    assert registry.kwargs["allow_http_over_tor"] is False


def test_service_tor_classmethod(monkeypatch):
    calls = []

    def fake_tor_service(root, **kwargs):
        calls.append((root, kwargs))
        return "service-sentinel"

    monkeypatch.setattr(tor, "tor_service", fake_tor_service)

    result = Service.tor("https://example.org/service", token="abc", host="127.0.0.2", port=9051)

    assert result == "service-sentinel"
    assert calls == [
        (
            "https://example.org/service",
            {
                "host": "127.0.0.2",
                "port": 9051,
                "scheme": "socks5h",
                "session": None,
                "allow_http_over_tor": False,
                "token": "abc",
            },
        )
    ]


def test_registry_tor_classmethod(monkeypatch):
    calls = []

    def fake_tor_registry(**kwargs):
        calls.append(kwargs)
        return "registry-sentinel"

    monkeypatch.setattr(tor, "tor_registry", fake_tor_registry)

    result = Registry.tor(request_timeout=6, verify_tls=False)

    assert result == "registry-sentinel"
    assert calls == [
        {
            "registry_url": "https://registry.intermine.org/service/instances",
            "host": "127.0.0.1",
            "port": 9050,
            "scheme": "socks5h",
            "request_timeout": 6,
            "verify_tls": False,
            "session": None,
            "allow_http_over_tor": False,
        }
    ]
