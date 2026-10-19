"""Enrichment payloads and fresh client builders."""
import json

from tests.support.lists import client

FIELDS = ["identifier", "description", "p-value", "matches", "populationAnnotationCount"]
ROWS = [
    dict(zip(FIELDS, ["0007", 'α, "label"\nnext', 0.002, 3, 71])),
    dict(zip(FIELDS, ["GO:2", "second", 0.0001, 1, 8])),
    dict(zip(FIELDS, ["GO:3", "third", 0.04, 9, 105])),
]

def payload(rows=ROWS, footer=b'],\n"wasSuccessful":true,\n"statusCode":200,\n"error":null\n}'):
    return b'{\n"results":[\n' + b",\n".join(json.dumps(row, ensure_ascii=False).encode() for row in rows) + (b"\n" if rows else b"") + footer


def enrichment_client(profile="native", version=11, body=None):
    service, session = client(profile, name="list α & β")
    service._version = version
    session.routes[("POST", "/service/list/enrichment")] = payload() if body is None else body
    item = service.get_list("list α & β")
    return service, session, item

