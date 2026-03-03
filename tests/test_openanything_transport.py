from __future__ import annotations

import intermine314.util.core as core


class _StreamingRaw:
    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)
        self._index = 0
        self.closed = False
        self.decode_content = False

    def readline(self, size=-1):
        _ = size
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line

    def close(self):
        self.closed = True


class _StreamingResponse:
    def __init__(self, lines: list[bytes]):
        self.raw = _StreamingRaw(lines)
        self.closed = False
        self.close_calls = 0

    def raise_for_status(self):
        return None

    def close(self):
        self.closed = True
        self.close_calls += 1


class _StreamingSession:
    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)
        self.calls = []
        self.responses = []

    def get(self, url, timeout=None, verify=True, stream=False):
        self.calls.append((url, timeout, verify, stream))
        response = _StreamingResponse(self._lines)
        self.responses.append(response)
        return response


def test_openanything_streaming_response_closes_on_early_termination():
    session = _StreamingSession([b"<model>\n", b"<class/>\n", b"</model>\n"])

    with core.openAnything("https://example.org/service/model", session=session) as stream:
        for line in stream:
            assert line == b"<model>\n"
            break

    assert session.calls == [("https://example.org/service/model", None, True, True)]
    assert len(session.responses) == 1
    response = session.responses[0]
    assert response.close_calls == 1
    assert response.closed is True
    assert response.raw.closed is True
