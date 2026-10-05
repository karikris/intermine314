"""Retry safety is an operation property, rather than an HTTP verb property."""

import socket
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from intermine314.service.errors import WebserviceError
from intermine314.service.service import Service
from intermine314.service.session import InterMineURLOpener
from intermine314.service.transport import build_session
from tests.fixtures.compatibility import fixture_bytes


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


@contextmanager
def failing_server(*, disconnect=False):
    """Commit each first operation before returning an ambiguous failure."""
    counts = Counter()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            path = urlsplit(self.path).path
            with lock:
                counts[path] += 1
                first = counts[path] == 1
            failures = {"/service/ids", "/service/session", "/service/lists/rename/json",
                        "/service/query/results", "/read", "/write"}
            if path in failures and first:
                if disconnect:
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                status, body = 503, b"committed, then unavailable"
            else:
                status = 200
                body = {
                    "/service/version/ws": b"35",
                    "/service/model": fixture_bytes("model.xml"),
                    "/service/query/results": b'{"results":[\n["Alice",30],\n["Bob",40]\n],"wasSuccessful":true,"error":null}',
                    "/service/ids": b'{"error":null,"uid":"duplicate-job"}',
                }.get(path, b"{}")
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = do_PUT = do_DELETE = respond

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", counts
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("disconnect", [False, True])
@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_resolution_job_creation_is_never_retried(disconnect, profile):
    with failing_server(disconnect=disconnect) as (url, counts):
        with Service(url + "/service", token="offline-token", compatibility=profile) as service:
            with pytest.raises(WebserviceError):
                service.resolve_ids("Employee", ["identifier"])
        assert counts["/service/ids"] == 1


@pytest.mark.parametrize("path", ["/service/session", "/service/lists/rename/json"])
def test_mutating_gets_are_not_retried(path):
    with failing_server() as (url, counts), InterMineURLOpener() as opener:
        with pytest.raises(WebserviceError):
            opener.open(url + path)
        assert counts[path] == 1


def test_safe_query_post_retries_through_the_public_query_api():
    with failing_server() as (url, counts), Service(url + "/service") as service:
        query = service.new_query("Employee").select("name", "age")
        assert len([row for row in query.results("dict")]) == 2
        assert counts["/service/query/results"] == 2


def test_concurrent_read_and_mutation_keep_independent_retry_policies():
    with failing_server() as (url, counts), InterMineURLOpener() as opener:
        barrier = threading.Barrier(2)

        def request(path, safe):
            barrier.wait(timeout=5)
            with opener.open(url + path, b"payload", retry_safe=safe) as response:
                return response.read()

        with ThreadPoolExecutor(max_workers=2) as pool:
            read = pool.submit(request, "/read", True)
            write = pool.submit(request, "/write", False)
            assert read.result(timeout=5) == b"{}"
            with pytest.raises(WebserviceError):
                write.result(timeout=5)
        assert counts["/read"] == 2 and counts["/write"] == 1
        assert opener._session.get_adapter(url).max_retries.total == 0
