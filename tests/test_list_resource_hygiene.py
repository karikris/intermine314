from __future__ import annotations

from intermine314.lists.list import List


class _IterableResponse:
    def __init__(self, lines):
        self._iter = iter(lines)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def close(self):
        self.closed = True


class _TrackingOpener:
    def __init__(self, lines):
        self._lines = list(lines)
        self.responses = []

    def open(self, _url, _payload=None):
        response = _IterableResponse(self._lines)
        self.responses.append(response)
        return response


class _Service:
    version = 11
    root = "https://example.org/service"
    LIST_ENRICHMENT_PATH = "/lists/enrichment"

    def __init__(self, opener):
        self.opener = opener


class _Manager:
    pass


def _enrichment_lines():
    return (
        b'{"results":[',
        b'{"identifier":"term-1","p-value":0.01},',
        b'],"wasSuccessful":true,"statusCode":200,"error":null}',
    )


def test_calculate_enrichment_closes_response_when_consumer_breaks_early():
    opener = _TrackingOpener(_enrichment_lines())
    list_obj = List(
        service=_Service(opener),
        manager=_Manager(),
        name="genes",
        title="genes",
        type="Gene",
        size=1,
    )

    for row in list_obj.calculate_enrichment("pathway_enrichment"):
        assert row["identifier"] == "term-1"
        break

    assert opener.responses
    assert opener.responses[0].closed is True

