"""Retry safety is an operation property, rather than an HTTP verb property."""

import pytest

from intermine314.service.transport import build_session


def test_managed_requests_choose_fixed_operation_policies(monkeypatch):
    from requests import Response

    with build_session(proxy_url=None) as session:
        selected = []
        write = session.get_adapter("http://example.test")
        read = session._read_adapter

        def respond(adapter):
            def send(request, **kwargs):
                selected.append(adapter)
                response = Response()
                response.status_code = 200
                response._content = b"{}"
                response.request = request
                return response
            return send

        monkeypatch.setattr(read, "send", respond(read))
        monkeypatch.setattr(write, "send", respond(write))
        session.request("GET", "http://example.test/mutation")
        session.request("POST", "http://example.test/query", retry_safe=True)
        session.request("GET", "http://example.test/unknown")
        assert selected == [write, read, write]
        assert write.max_retries.total == 0
        assert read.max_retries.total == 5
        assert read.max_retries.allowed_methods == ("GET", "HEAD", "POST")


def test_retry_context_resets_when_request_fails(monkeypatch):
    with build_session(proxy_url=None) as session:
        write = session.get_adapter("https://example.test")

        def fail(*args, **kwargs):
            raise RuntimeError("transport failure")

        monkeypatch.setattr(session._read_adapter, "send", fail)
        with pytest.raises(RuntimeError, match="transport failure"):
            session.request("GET", "https://example.test", retry_safe=True)
        assert session.get_adapter("https://example.test") is write
