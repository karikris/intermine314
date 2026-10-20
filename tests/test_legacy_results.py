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
