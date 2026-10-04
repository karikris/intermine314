"""Loopback HTTP transport for real Requests/urllib3 integration tests."""

import gzip
import threading
import zlib
from compression import zstd
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.fixtures.compatibility import CapturedRequest

ENCODERS = {None: lambda body: body, "gzip": gzip.compress, "deflate": zlib.compress, "zstd": zstd.compress}


@contextmanager
def loopback_service(routes):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def respond(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request = CapturedRequest(self.command, self.path, body, dict(self.headers))
            requests.append(request)
            route = routes.get((self.command, request.path))
            if route is None:
                self.send_error(404)
                return
            if callable(route):
                route = route(request)
            payload, encoding = route if isinstance(route, tuple) else (route, None)
            payload = ENCODERS[encoding](payload)
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            if encoding:
                self.send_header("Content-Encoding", encoding)
            self.end_headers()
            self.wfile.write(payload)

        do_GET = respond
        do_POST = respond

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()
