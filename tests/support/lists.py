"""Fresh list clients and strict list-operation routes."""

import importlib
import json
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree as ET

from tests.fixtures.compatibility import FixtureSession, fixture_bytes


def list_module():
    return importlib.import_module("intermine314.lists.list")


def manager_module():
    return importlib.import_module("intermine314.lists.listmanager")


def client(profile="native", *, size=2, name="identifiers", rows=None):
    from intermine314.service.service import Service
    from intermine314.webservice import Service as LegacyService
    session = FixtureSession.service(rows=rows)
    info = {"name": name, "title": "Fixed title", "type": "Employee", "size": str(size),
            "description": "Unicode α", "tags": ["tag α"], "status": "CURRENT"}
    session.routes[("GET", "/service/lists")] = json.dumps({
        "wasSuccessful": True, "lists": [info],
    }).encode()
    session.routes[("POST", "/service/lists")] = fixture_bytes("list-created.json")
    session.routes[("DELETE", "/service/lists")] = b'{"wasSuccessful":true}'
    service = (Service if profile == "native" else LegacyService)(
        "https://offline.example/service", session=session, token="secret",
        request_timeout=37, user_agent="list-client", verify_tls="/custom/ca.pem",
    )
    return service, session


def params(request):
    return parse_qs(urlsplit(request.url).query, keep_blank_values=True)




def server(monkeypatch, profile="native"):
    service, session = client(profile)
    base = json.loads(session.routes[("GET", "/service/lists")])["lists"][0]
    inventory = {"left": dict(base, name="left", size=3), "right": dict(base, name="right", size=2)}
    members = {"left": {1, 2, 3}, "right": {3, 4}}
    original = session.request

    def publish():
        session.routes[("GET", "/service/lists")] = json.dumps({"wasSuccessful": True, "lists": list(inventory.values())}).encode()

    def request(method, url, **kwargs):
        path = urlsplit(url).path
        form = parse_qs(urlsplit(url).query, keep_blank_values=True)
        if method == "POST" and path.startswith("/service/query/"):
            form = parse_qs(kwargs["data"].decode() if isinstance(kwargs["data"], bytes) else kwargs["data"], keep_blank_values=True)
            xml = ET.fromstring(form["query"][0])
            views = xml.attrib["view"].split()
            assert len(views) == 1 and views[0].endswith(".id"), "Server accepts exactly one ID view"
            constraint = xml.find("constraint[@op='IN']")
            selected = members[constraint.attrib["value"]] if constraint is not None else {2, 3}
            name = form["listName"][0]
            values = members.get(name, set()) | selected if "append" in path else set(selected)
        elif path in ["/service/lists/" + p + "/json" for p in ("union", "intersect", "diff", "subtract")]:
            name = form["name"][0]
            if "subtract" in path:
                values = set.union(*(members[n] for n in form["references"][0].split(";"))) - set.union(*(members[n] for n in form["subtract"][0].split(";")))
            else:
                sets = [members[n] for n in form["lists"][0].split(";")]
                op = set.intersection if "intersect" in path else set.symmetric_difference if "diff" in path else set.union
                values = sets[0].copy()
                for other in sets[1:]:
                    values = op(values, other)
        elif path == "/service/lists/rename":
            old, name = form["oldname"][0], form["newname"][0]
            values = members.pop(old)
            inventory[name] = dict(inventory.pop(old), name=name)
        elif method == "DELETE":
            name = form["name"][0]
            inventory.pop(name)
            members.pop(name)
            publish()
            return original(method, url, **kwargs)
        else:
            return original(method, url, **kwargs)
        members[name] = set(values)
        inventory[name] = dict(inventory.get(name, base), name=name, size=len(values),
                               description=form.get("description", [inventory.get(name, base)["description"]])[0],
                               tags=([tag for tag in form["tags"][0].split(";") if tag] if "tags" in form else inventory.get(name, base)["tags"]))
        session.routes[(method, path)] = json.dumps({"wasSuccessful": True, "listName": name, "unmatchedIdentifiers": ["unmatched"]}).encode()
        publish()
        return original(method, url, **kwargs)

    publish()
    monkeypatch.setattr(session, "request", request)
    return service, session, inventory, members

