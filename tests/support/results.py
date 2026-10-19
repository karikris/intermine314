"""Result-wire builders; transports remain fresh per call."""
import json
from urllib.parse import parse_qs

from tests.fixtures.compatibility import FixtureSession


def payload(*rows):
    return ('{"results":[\n' + ',\n'.join(json.dumps(row) for row in rows) + '\n],"wasSuccessful":true}\n').encode()


class ResultSession(FixtureSession):
    """Separate count/results routes and observe bounded stream consumption."""

    def __init__(self, rows, count=1, version=8, interrupt=False):
        super().__init__(FixtureSession.service(version=version).routes)
        self.rows = rows
        self.count = count
        self.interrupt = interrupt
        self.lines = []

    def request(self, method, url, data=None, **kwargs):
        if method == "POST" and url.endswith("/query/results"):
            params = parse_qs(data.decode())
            self.routes[(method, "/service/query/results")] = str(self.count).encode() if params["format"] == ["count"] else self.rows
        response = super().request(method, url, data=data, **kwargs)
        if method == "POST" and params["format"] != ["count"]:
            original = response.iter_lines

            def lines(decode_unicode=False):
                for index, line in enumerate(original(decode_unicode)):
                    self.lines.append(line)
                    if self.interrupt and index == 1:
                        raise KeyboardInterrupt
                    yield line

            response.iter_lines = lines
        return response


def requests(session):
    return [parse_qs(request.data.decode()) for request in session.requests if request.method == "POST"]

