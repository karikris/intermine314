"""Strict in-memory transports: unregistered requests never reach the network."""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

FIXTURE_ROOT = Path(__file__).resolve().parent
SERVICE_ROOT = "https://offline.example/service"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURE_ROOT / name).read_bytes()


@dataclass
class CapturedRequest:
    method: str
    url: str
    data: bytes | str | None = None
    headers: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)

    @property
    def path(self) -> str:
        return urlsplit(self.url).path


class FixtureConnection(BytesIO):
    """urllib-style independent stream with observable, idempotent close."""

    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.close_calls = 0

    def close(self):
        if not self.closed:
            self.close_calls += 1
        super().close()


class FixtureResponse:
    """requests-style response; raw reads and line iteration consume one stream."""

    def __init__(self, payload: bytes, status_code=200):
        self.content = payload
        self.status_code = status_code
        self.reason = "OK" if status_code < 400 else "Offline HTTP error"
        self.headers = {"Content-Type": "application/json; charset=utf-8"}
        self.raw = FixtureConnection(payload)
        self.close_calls = 0
        self.closed = False

    def iter_lines(self, decode_unicode=False):
        for line in self.raw:
            value = line.rstrip(b"\r\n")
            yield value.decode("utf-8") if decode_unicode else value

    def close(self):
        if not self.closed:
            self.close_calls += 1
            self.closed = True
            self.raw.close()


class _Routes:
    def __init__(self, routes):
        self.routes = dict(routes)
        self.requests = []
        self.close_calls = 0

    def _capture(self, method, url, data=None, headers=None, **options):
        request = CapturedRequest(method.upper(), url, data, dict(headers or {}), options)
        self.requests.append(request)
        key = (request.method, request.path)
        if key not in self.routes:
            raise AssertionError(f"Unregistered offline request: {request.method} {url}")
        route = self.routes[key]
        if isinstance(route, Exception):
            raise route
        return route

    def close(self):
        self.close_calls += 1


class FixtureOpener(_Routes):
    def __init__(self, routes):
        super().__init__(routes)
        self.connections = []

    def open(self, url, data=None, headers=None, method=None, **options):
        payload = self._capture(method or ("POST" if data is not None else "GET"), url, data, headers, **options)
        connection = FixtureConnection(payload)
        self.connections.append(connection)
        return connection

    def read(self, url, data=None):
        with self.open(url, data) as connection:
            return connection.read().decode("utf-8")


class FixtureSession(_Routes):
    def __init__(self, routes):
        super().__init__(routes)
        self.responses = []

    def request(self, method, url, data=None, headers=None, **options):
        route = self._capture(method, url, data, headers, **options)
        status, payload = route if isinstance(route, tuple) else (200, route)
        response = FixtureResponse(payload, status)
        self.responses.append(response)
        return response

    @classmethod
    def service(cls, *, version=8, rows=None):
        default_rows = "rows-modern.json" if version >= 8 else "rows-legacy.json"
        return cls({
            ("GET", "/service/version/ws"): str(version).encode("ascii"),
            ("GET", "/service/version/release"): fixture_bytes("version-release.txt"),
            ("GET", "/service/model"): fixture_bytes("model.xml"),
            ("POST", "/service/query/results"): fixture_bytes(default_rows) if rows is None else rows,
            ("GET", "/service/templates/xml"): fixture_bytes("templates.xml"),
            ("GET", "/service/lists/json"): fixture_bytes("lists.json"),
            ("GET", "/service/widgets"): fixture_bytes("widgets.json"),
        })
