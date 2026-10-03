"""Server wire tests for pinned list/query conversions and set operations."""
from __future__ import annotations

import json
import operator
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree as ET

import pytest

from intermine314.service.errors import WebserviceError
from tests.test_lists_crud import assert_closed, client, params


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


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("view", [None, "name", "department.name"])
def test_upload_projects_clone_preserves_original_query(monkeypatch, profile, view):
    service, session, _, _ = server(monkeypatch, profile)
    query = service.new_query("Employee")
    query.views.clear()
    if view:
        query.add_view(view)
    query.add_constraint("age", ">", 20)
    query.add_constraint("name", "!=", "x")
    query.set_logic("A or B")
    query.add_join("department", "OUTER")
    before = query.to_xml()
    assert query.to_query() is query
    manager = service.list_manager()
    item = manager.create_list(query, name="uploaded α", tags=["x", "y"], description="chosen")
    assert item.name == "uploaded α" and item.unmatched_identifiers == {"unmatched"}
    upload = next(r for r in session.requests if r.method == "POST")
    form = parse_qs(upload.data.decode(), keep_blank_values=True)
    assert upload.path == "/service/query/tolist"
    assert form["listName"] == ["uploaded α"] and form["tags"] == ["x;y"] and form["description"] == ["chosen"]
    xml = ET.fromstring(form["query"][0])
    expected = "Employee.department.id" if view == "department.name" else "Employee.id"
    assert xml.attrib["view"] == expected
    assert xml.attrib["constraintLogic"] == "A or B"
    assert len(xml.findall("constraint")) == 2 and xml.find("join").attrib["style"] == "OUTER"
    assert query.to_xml() == before and query.compatibility == profile
    assert upload.options["verify"] == "/custom/ca.pem" and upload.options["timeout"][1] == 37
    assert params(upload)["token"] == ["secret"]
    assert_closed(session)


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_list_conversion_and_constraints_and_append_query(monkeypatch, profile):
    service, session, _, members = server(monkeypatch, profile)
    manager = service._get_list_manager()
    left, right = manager.get_list("left"), manager.get_list("right")
    query = right.to_query()
    assert query.views == service.new_query("Employee").views
    constraint = ET.fromstring(query.to_xml()).find("constraint")
    assert constraint.attrib["op"] == "IN" and constraint.attrib["value"] == "right"
    node = right.make_list_constraint("Employee", "NOT IN")
    assert node.vargs == ("Employee", "NOT IN", "right")
    assert left.append(query) is left
    assert members["left"] == {1, 2, 3, 4} and left.size == 4 and left.unmatched_identifiers == {"unmatched"}
    upload = next(r for r in session.requests if r.method == "POST")
    form = parse_qs(upload.data.decode())
    assert upload.path == "/service/query/append/tolist" and form["path"] == ["None"] and form["listName"] == ["left"]
    assert ET.fromstring(form["query"][0]).attrib["view"] == "Employee.id"
    assert query.views == service.new_query("Employee").views
    node = service.new_query("Employee").make_list_constraint("Employee", "IN")
    assert node.vargs[:2] == ("Employee", "IN") and node.vargs[2] in manager._temp_lists
    assert_closed(session)


@pytest.mark.parametrize("method,endpoint,expected", [("union", "union", {1, 2, 3, 4}), ("intersect", "intersect", {3}), ("xor", "diff", {1, 2, 4}), ("subtract", "subtract", {1, 2})])
def test_manager_server_operations(monkeypatch, method, endpoint, expected):
    service, session, _, members = server(monkeypatch)
    manager = service.list_manager()
    args = ([manager.get_list("left")], ["right"]) if method == "subtract" else ([manager.get_list("left"), "right"],)
    result = getattr(manager, method)(*args, name="result", tags=["a", "b"])
    assert members[result.name] == expected and result.size == len(expected)
    request = next(r for r in session.requests if r.path.endswith("/json"))
    assert request.method == "GET" and request.path == f"/service/lists/{endpoint}/json"
    form = params(request)
    assert form["name"] == ["result"] and form["tags"] == ["a;b"]
    if method == "subtract":
        assert form["references"] == ["left"] and form["subtract"] == ["right"] and form["description"] == ["Subtraction of right from left"]
    else:
        assert form["lists"] == ["left;right"] and form["description"] == [{"union": "Union", "intersect": "Intersection", "xor": "Difference"}[method] + " of left and right"]
    assert_closed(session)


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("op,expected", [(operator.or_, {1, 2, 3, 4}), (operator.add, {1, 2, 3, 4}), (operator.and_, {3}), (operator.xor, {1, 2, 4}), (operator.sub, {1, 2})])
@pytest.mark.parametrize("query_left", [False, True])
def test_query_and_list_operators_server_results(monkeypatch, profile, op, expected, query_left):
    service, session, _, members = server(monkeypatch, profile)
    manager = service._get_list_manager()
    left, right = manager.get_list("left"), manager.get_list("right")
    result = op(left.to_query() if query_left else left, right)
    assert members[result.name] == expected
    assert result.name in manager._temp_lists
    assert_closed(session)


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("op,expected", [(operator.iand, {3}), (operator.ixor, {1, 2, 4}), (operator.isub, {1, 2})])
def test_inplace_retains_metadata_and_context_ownership(monkeypatch, named, op, expected):
    service, session, inventory, members = server(monkeypatch)
    manager = service.list_manager()
    with manager:
        original = manager.create_list(manager.get_list("left"), name="named" if named else None)
        old_name, description, tags = original.name, original.description, original.tags
        result = op(original, manager.get_list("right"))
        assert result.name == old_name and result.description == description and result.tags == tags
        assert members[old_name] == expected and result is not original
        assert (old_name in manager._temp_lists) is not named
    assert (old_name in inventory) is named
    assert manager._temp_lists == set()
    assert_closed(session)


@pytest.mark.parametrize("kind", ["list", "query", "collection", "generator"])
def test_append_queryable_classification_and_iadd(monkeypatch, kind):
    service, session, _, members = server(monkeypatch)
    manager = service.list_manager()
    left, right = manager.get_list("left"), manager.get_list("right")
    query = service.new_query("Employee")
    source = {"list": right, "query": query, "collection": [right, query], "generator": iter([right, query])}[kind]
    assert operator.iadd(left, source) is left
    assert members["left"] == ({1, 2, 3} if kind == "query" else {1, 2, 3, 4})
    assert not any(r.path == "/service/query/results" for r in session.requests)
    assert sum(r.path == "/service/lists/union/json" for r in session.requests) == (1 if kind in ("collection", "generator") else 0)
    assert_closed(session)


def test_seven_service_delegates_use_lazy_cached_manager(monkeypatch):
    service, session, inventory, _ = server(monkeypatch)
    assert service._list_manager is None
    assert service.get_list_count() == 2
    manager = service._list_manager
    assert set(service.get_all_list_names()) == {"left", "right"}
    assert {item.name for item in service.get_all_lists()} == {"left", "right"}
    assert service.get_list("left") is service.l("left") is manager.get_list("left")
    service.create_list(service.get_list("left"), name="copy")
    service.delete_lists(["copy"])
    assert "copy" not in inventory and service._list_manager is manager
    for method in ("union", "intersect", "xor", "subtract", "make_list_names"):
        assert not hasattr(service, method)
    assert_closed(session)


@pytest.mark.parametrize("kind", ["create", "append", "collection"])
def test_query_upload_http_failure_never_falls_back(monkeypatch, kind):
    service, session = client()
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    query = service.new_query("Employee")
    path = "/service/query/append/tolist" if kind == "append" else "/service/query/tolist"
    session.routes[("POST", path)] = (500, b"failed")
    with pytest.raises(WebserviceError):
        manager.create_list(query, name="copy") if kind == "create" else item.append(query if kind == "append" else [query])
    assert sum(r.method == "POST" for r in session.requests) == 1
    assert_closed(session)


@pytest.mark.parametrize("foreign", [True, False])
def test_upload_rejects_foreign_or_unbound_query_before_requests(monkeypatch, foreign):
    service, session, _, _ = server(monkeypatch)
    query = service.new_query("Employee")
    other, _ = client()
    other.root = "https://foreign.example/service"
    query.service = other if foreign else None
    before = len(session.requests)
    with pytest.raises(ValueError, match="service|Service"):
        service.list_manager().create_list(query, name="copy")
    assert len(session.requests) == before


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("organism", ["human", ["human", "mouse"]])
def test_organism_uses_existing_configured_service(monkeypatch, profile, organism):
    service, session, _, _ = server(monkeypatch, profile)
    session.routes[("GET", "/service/model")] = b'''<model name="testmodel" package="test"><class name="Gene" is-interface="true"><attribute name="symbol" type="java.lang.String"/><reference name="organism" referenced-type="Organism"/></class><class name="Organism" is-interface="true"><attribute name="name" type="java.lang.String"/></class></model>'''
    item = service.list_manager().create_list(["A", "B"], "Gene", organism=organism, name="filtered")
    assert item.name == "filtered"
    upload = next(r for r in session.requests if r.method == "POST")
    xml = ET.fromstring(parse_qs(upload.data.decode())["query"][0])
    constraints = {node.attrib["path"]: node for node in xml.findall("constraint")}
    assert constraints["Gene.symbol"].attrib["op"] == "ONE OF"
    path = "Gene.organism.name" if isinstance(organism, list) else "Gene.organism"
    assert constraints[path].attrib["op"] == ("ONE OF" if isinstance(organism, list) else "LOOKUP")
    assert upload.options["verify"] == "/custom/ca.pem" and params(upload)["token"] == ["secret"]
    assert_closed(session)


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_service_direct_delegate_hook_and_unknown_attribute(monkeypatch, profile):
    service, session, _, _ = server(monkeypatch, profile)
    before = len(session.requests)
    with pytest.raises(AttributeError, match="Could not find nonexistent"):
        service.__getattr__("nonexistent")
    assert len(session.requests) == before and service._list_manager is None
    assert service.LIST_MANAGER_METHODS == frozenset(["create_list", "delete_lists", "get_list", "l", "get_all_lists", "get_all_list_names", "get_list_count"])
    assert service.__getattr__("get_list_count")() == 2
    assert_closed(session)


@pytest.mark.parametrize("method", ["union", "intersect", "xor", "subtract"])
def test_set_operation_rejects_semicolon_in_individual_names(monkeypatch, method):
    service, session, _, _ = server(monkeypatch)
    manager = service.list_manager()
    before = len(session.requests)
    args = (["left"], ["right;unexpected"]) if method == "subtract" else (["left", "right;unexpected"],)
    with pytest.raises(ValueError, match="semicolon"):
        getattr(manager, method)(*args, name="result")
    assert len(session.requests) == before


def test_same_root_query_uses_target_transport_and_preserves_source(monkeypatch):
    target, session, _, _ = server(monkeypatch)
    source, source_session = client("legacy")
    query = source.new_query("Employee")
    before = len(source_session.requests)
    assert query.get_list_upload_uri() == source.root + "/query/tolist"
    assert query.get_list_append_uri() == source.root + "/query/append/tolist"
    target.list_manager().create_list(query, name="copy")
    assert len(source_session.requests) == before and query.service is source
    assert sum(r.path == "/service/query/tolist" for r in session.requests) == 1
    assert_closed(session)
    assert_closed(source_session)


@pytest.mark.parametrize("operation", ["create", "append", "union"])
@pytest.mark.parametrize("failure", ["parser", "unsuccessful", "interrupt"])
def test_query_and_set_failures_close_response_without_retry(monkeypatch, operation, failure):
    from intermine314.service.session import _ResponseStreamAdapter

    service, session = client()
    manager = service.list_manager()
    item = manager.get_list("identifiers")
    query = service.new_query("Employee")
    path = {"create": "/service/query/tolist", "append": "/service/query/append/tolist", "union": "/service/lists/union/json"}[operation]
    verb = "GET" if operation == "union" else "POST"
    session.routes[(verb, path)] = b'invalid' if failure == "parser" else b'{"wasSuccessful":false,"error":"denied"}'
    if failure == "interrupt":
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt("interrupted")
        monkeypatch.setattr(_ResponseStreamAdapter, "read", interrupted)
    before = len(session.requests)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else WebserviceError):
        if operation == "create":
            manager.create_list(query, name="copy")
        elif operation == "append":
            item.append(query)
        else:
            manager.union([item], name="copy")
    assert len(session.requests) == before + 1 and item.size == 2 and not item.unmatched_identifiers
    assert_closed(session)


def test_mixed_append_rejects_before_upload(monkeypatch):
    service, session, _, _ = server(monkeypatch)
    manager = service.list_manager()
    item = manager.get_list("left")
    before = len(session.requests)
    with pytest.raises(TypeError, match="mix queryables"):
        item.append([item, "identifier"])
    assert len(session.requests) == before


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("operation", ["create", "append"])
@pytest.mark.parametrize("shape", ["ambiguous", "unrooted"])
def test_invalid_projection_rejects_before_name_allocation_or_upload(monkeypatch, profile, operation, shape):
    from intermine314.query.builder import Query

    service, session, _, _ = server(monkeypatch, profile)
    manager = service.list_manager()
    item = manager.get_list("left")
    query = service.new_query("Employee.name", "Employee.department.name") if shape == "ambiguous" else Query(model=service._resolve_query_model(), service=service, compatibility=profile)
    before = len(session.requests)
    original_views = list(query.views)
    with pytest.raises(ValueError, match="select one entity path"):
        manager.create_list(query) if operation == "create" else item.append(query)
    assert len(session.requests) == before and not manager._temp_lists
    assert query.views == original_views


_OPERAND_OPERATIONS = [
    "manager_union", "manager_intersect", "manager_xor", "manager_subtract",
    "list_or", "list_add", "list_and", "list_xor", "list_sub",
    "query_or", "query_add", "query_and", "query_xor", "query_sub",
    "list_iand", "list_ixor", "list_isub", "append_collection", "iadd_collection",
]


def apply_operand_operation(operation, manager, left, right):
    if operation.startswith("manager_"):
        method = operation.removeprefix("manager_")
        args = ([left], [right]) if method == "subtract" else ([left, right],)
        return getattr(manager, method)(*args)
    if operation == "append_collection":
        return left.append([right])
    if operation == "iadd_collection":
        return operator.iadd(left, [right])
    kind, name = operation.split("_")
    op = {"or": operator.or_, "and": operator.and_, "add": operator.add,
          "xor": operator.xor, "sub": operator.sub, "iand": operator.iand,
          "ixor": operator.ixor, "isub": operator.isub}[name]
    return op(left.to_query() if kind == "query" else left, right)


@pytest.mark.parametrize("operation", _OPERAND_OPERATIONS)
@pytest.mark.parametrize("foreign", [True, False])
def test_list_operands_reject_foreign_roots_but_accept_same_root_clients(monkeypatch, operation, foreign):
    service, session, inventory, members = server(monkeypatch)
    other, other_session, _, other_members = server(monkeypatch)
    manager, other_manager = service._get_list_manager(), other.list_manager()
    left, right = manager.get_list("left"), other_manager.get_list("right")
    # Force lazy model loading before measuring operation requests.
    left.to_query()
    if foreign:
        other.root = "https://foreign.example/service"
        other_members["right"] = {99}
    before, other_before = len(session.requests), len(other_session.requests)
    if foreign:
        with pytest.raises(ValueError, match="same root"):
            apply_operand_operation(operation, manager, left, right)
        assert len(session.requests) == before and not manager._temp_lists
        assert members == {"left": {1, 2, 3}, "right": {3, 4}}
        assert set(inventory) == {"left", "right"}
    else:
        result = apply_operand_operation(operation, manager, left, right)
        expected = ({3} if operation.endswith(("intersect", "and", "iand")) else
                    {1, 2, 4} if operation.endswith(("xor", "ixor")) else
                    {1, 2} if operation.endswith(("subtract", "sub", "isub")) else
                    {1, 2, 3, 4})
        assert members[result.name] == expected
        assert len(session.requests) > before
    assert len(other_session.requests) == other_before
    assert_closed(session)
    assert_closed(other_session)


@pytest.mark.parametrize("method", ["union", "intersect", "xor", "subtract", "make_list_names", "append"])
@pytest.mark.parametrize("invalid_kind", ["foreign_list", "semicolon_name"])
def test_all_operands_prevalidated_before_uploading_queries(monkeypatch, method, invalid_kind):
    service, session, inventory, members = server(monkeypatch)
    manager = service._get_list_manager()
    left = manager.get_list("left")
    query = left.to_query()
    other, other_session, _, _ = server(monkeypatch)
    other_manager = other.list_manager()
    invalid = other_manager.get_list("right")
    other.root = "https://foreign.example/service"
    if invalid_kind == "semicolon_name":
        if method == "append":
            # Preserve the queryable collection shape while testing the name.
            invalid = manager.get_list("right")
            invalid._name = "right;unexpected"
        else:
            invalid = "right;unexpected"
    before, other_before = len(session.requests), len(other_session.requests)
    with pytest.raises(ValueError, match="same root|semicolon"):
        if method == "subtract":
            manager.subtract([query], [invalid])
        elif method == "append":
            left.append(iter([query, invalid]))
        else:
            getattr(manager, method)(iter([query, invalid]))
    assert len(session.requests) == before and len(other_session.requests) == other_before
    assert not manager._temp_lists and set(inventory) == {"left", "right"}
    assert members["left"] == {1, 2, 3}
    assert_closed(session)
    assert_closed(other_session)
