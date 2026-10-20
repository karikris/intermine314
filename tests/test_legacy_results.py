"""Legacy presentation over the shared, bounded managed result transport."""
import copy
import inspect
from urllib.parse import parse_qs

import pytest

from intermine314.results import ResultIterator, ResultRow
from intermine314.service.errors import WebserviceError
from tests.support.iterators import make_iterator


@pytest.mark.parametrize("empty", [False, True])
def test_legacy_sizing_truthiness_and_list_use_independent_validated_streams(empty):
    payload = b'{"results":[\n],"wasSuccessful":true}\n' if empty else None
    iterator, session, opener = make_iterator(compatibility="legacy", payload=payload)
    expected = 0 if empty else 3
    assert isinstance(iterator, ResultIterator)
    assert len(iterator) == expected
    assert bool(iterator) is (not empty)
    assert len(session.requests) == 2
    assert len(list(iterator)) == expected
    assert len(session.requests) == 4
    assert all(response.close_calls == 1 for response in session.responses)
    assert all(parse_qs(call.data.decode())["format"] == ["json"] for call in session.requests)
    opener.close()


def test_legacy_sizing_preserves_active_cursor_constructor_and_copy():
    iterator, session, opener = make_iterator("list", compatibility="legacy")
    first = next(iterator)
    assert first == ["foo", "bar", "baz"]
    assert len(iterator) == 3
    assert next(iterator) == [123, 1.23, -1.23]
    assert len(session.requests) == 2
    restored = copy.copy(iterator)
    assert type(restored) is type(iterator)
    assert str(inspect.signature(ResultIterator)) == "(service, path, params, rowformat, view, cld=None, *, decimal_paths=())"
    iterator.close()
    opener.close()
    assert all(response.closed for response in session.responses)


@pytest.mark.parametrize("failure", ["footer", "parser", "interrupt"])
def test_legacy_sizing_closes_on_footer_parser_and_interrupt_failures(failure):
    payload = b'{"results":[\n[1]\n],"wasSuccessful":false,"error":"failed"}\n' if failure == "footer" else None
    iterator, session, opener = make_iterator("rr", compatibility="legacy", payload=payload)

    class BrokenRow(ResultRow):
        def __init__(self, *args):
            raise KeyboardInterrupt("stop") if failure == "interrupt" else ValueError("broken row")

    if failure != "footer":
        iterator.row = BrokenRow
    expected = {"footer": WebserviceError, "parser": ValueError, "interrupt": KeyboardInterrupt}[failure]
    with pytest.raises(expected):
        len(iterator)
    assert len(session.requests) == 1
    assert all(response.close_calls == 1 for response in session.responses)
    opener.close()


@pytest.mark.parametrize("version", [7, 8])
def test_legacy_rows_share_mutable_views_and_keep_independent_indexes(version):
    iterator, _, opener = make_iterator("rr", compatibility="legacy", version=version)
    first, second, _ = [row for row in iterator]
    assert first.views is second.views is iterator.view
    assert isinstance(first.views, list)
    assert first["name"] == "foo" and second["name"] == 123
    assert first.index_map is not second.index_map
    first.index_map["alias"] = 0
    assert first["alias"] == "foo" and "alias" not in second.index_map
    first.views.append("Employee.extra")
    assert second.views[-1] == "Employee.extra"
    first.views = ["replacement"]
    assert second.views is iterator.view
    opener.close()


def test_legacy_objects_have_mutable_independent_projection_metadata():
    from intermine314.webservice import Service
    from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession
    from tests.support.results import payload

    data = {"class": "Department", "name": "A", "company": {"name": "B"}}
    session = FixtureSession.service(rows=payload(data, data))
    with Service(SERVICE_ROOT, session=session) as service:
        query = service.select("Department.name", "Department.company.name")
        first, second = [row for row in query.results()]
        assert first.selected_attributes == second.selected_attributes == ["name"]
        assert first.reference_paths == second.reference_paths == {"company.": ["company.name"]}
        first.selected_attributes.append("extra")
        first.reference_paths["company."].append("company.vatNumber")
        assert second.selected_attributes == ["name"]
        assert second.reference_paths["company."] == ["company.name"]
        assert first.company.selected_attributes == ["name", "vatNumber"]
        assert second.company.selected_attributes == ["name"]
    assert all(response.close_calls == 1 for response in session.responses)
