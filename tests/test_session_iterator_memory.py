from __future__ import annotations

from intermine314.service import session as session_module


def _json_stream_lines(rows):
    lines = [b'{"results":[']
    for idx, row in enumerate(rows):
        suffix = b"," if idx < len(rows) - 1 else b""
        lines.append(row + suffix)
    lines.append(b'],"wasSuccessful":true,"statusCode":200,"error":null}')
    return lines


class _LineConnection:
    def __init__(self, lines):
        self._iter = iter(lines)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def close(self):
        self.closed = True


class _Service:
    def __init__(self, opener, *, version=8):
        self.version = int(version)
        self.root = "https://example.org/service"
        self.opener = opener


class _TrackingOpener:
    def __init__(self, lines):
        self._lines = list(lines)
        self.connections = []

    def open(self, url, data=None, *_args, method=None, **_kwargs):
        _ = (url, data, method)
        con = _LineConnection(self._lines)
        self.connections.append(con)
        return con


def test_result_iterator_closes_connection_when_consumer_breaks_early():
    opener = _TrackingOpener(_json_stream_lines([b'["geneA"]', b'["geneB"]']))
    service = _Service(opener, version=8)
    iterator = session_module.ResultIterator(
        service,
        "/query/results",
        {"query": "xml"},
        "list",
        ["Gene.symbol"],
    )
    for row in iterator:
        assert row == ["geneA"]
        break
    assert len(opener.connections) == 1
    assert opener.connections[0].closed is True
