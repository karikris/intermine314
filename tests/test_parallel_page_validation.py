"""Parallel pages must finish their protocol before any row is accepted."""

import json

import pytest

from intermine314.query.parallel_offset import (
    ParallelExecutionError,
    run_parallel_offset,
)
from intermine314.service import Service
from intermine314.service.errors import WebserviceError
from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession


def parallel(query, **options):
    return run_parallel_offset(
        query, size=2, page_size=2, max_workers=1, inflight_limit=1,
        thread_name_prefix="page-validation", **options,
    )


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_full_page_validates_failure_footer_before_yielding(profile):
    wire = (
        b'{"results":[\n["Alice",42],\n["Bob",21]\n],'
        b'"wasSuccessful":false,"error":"failure after rows"}\n'
    )
    session = FixtureSession.service(version=30, rows=wire)
    with Service(SERVICE_ROOT, session=session, compatibility=profile) as service:
        query = service.select("Employee.name", "Employee.age")
        with pytest.raises(ParallelExecutionError) as caught:
            next(parallel(query))
    assert isinstance(caught.value.__cause__, WebserviceError)
    assert caught.value.page_index == 0 and caught.value.offset == 0
    assert len([call for call in session.requests if call.path.endswith("query/results")]) == 1
    assert all(response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 0


def test_overproducing_page_reads_only_one_extra_item_and_closes():
    consumed = []
    closed = []

    class Query:
        def results(self, **kwargs):
            try:
                for value in range(100):
                    consumed.append(value)
                    yield value
            finally:
                closed.append(True)

    with pytest.raises(ParallelExecutionError) as caught:
        next(parallel(Query()))
    assert isinstance(caught.value.__cause__, WebserviceError)
    assert "more rows" in str(caught.value.__cause__)
    assert consumed == [0, 1, 2]
    assert closed == [True]


JSON_ROWS = ["dict", "list", "rr", "json", "jsonrows", "jsonobjects"]


def wire_rows(row, count):
    if row == "jsonobjects":
        values = [{"id": index, "class": "Employee", "name": f"row-{index}", "age": index} for index in range(count)]
    else:
        values = [[f"row-{index}", index] for index in range(count)]
    return b'{"results":[\n' + b',\n'.join(json.dumps(value).encode() for value in values) + b'\n'


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("row", JSON_ROWS)
@pytest.mark.parametrize("count", [0, 1, 2])
def test_successful_empty_short_and_exact_pages_are_closed_once(profile, row, count):
    wire = wire_rows(row, count) + b'],"wasSuccessful":true}\n'
    session = FixtureSession.service(version=30, rows=wire)
    with Service(SERVICE_ROOT, session=session, compatibility=profile) as service:
        query = service.select("Employee.name", "Employee.age")
        rows = list(parallel(query, row=row))
    assert len(rows) == count
    if count and row == "dict":
        assert rows[0] == {"Employee.name": "row-0", "Employee.age": 0}
    if count and row == "jsonobjects":
        assert rows[0].name == "row-0"
    assert len([call for call in session.requests if call.path.endswith("query/results")]) == 1
    assert all(response.close_calls == 1 for response in session.responses)


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("row", JSON_ROWS)
@pytest.mark.parametrize("footer", [
    b'],"wasSuccessful":false,"error":"failed"}\n',
    b'],"error":"missing status"}\n',
    b'],"wasSuccessful":true,broken}\n',
    b'',
])
def test_invalid_terminal_status_never_releases_full_page(profile, row, footer):
    session = FixtureSession.service(version=30, rows=wire_rows(row, 2) + footer)
    with Service(SERVICE_ROOT, session=session, compatibility=profile) as service:
        query = service.select("Employee.name", "Employee.age")
        with pytest.raises(ParallelExecutionError) as caught:
            next(parallel(query, row=row))
    assert isinstance(caught.value.__cause__, WebserviceError)
    assert all(response.close_calls == 1 for response in session.responses)


@pytest.mark.parametrize("row", ["json", "jsonrows"])
@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_json_null_is_a_row_and_cannot_be_mistaken_for_terminal_sentinel(profile, row):
    session = FixtureSession.service(version=30, rows=b'{"results":[\nnull,\nnull\n],"wasSuccessful":true}\n')
    with Service(SERVICE_ROOT, session=session, compatibility=profile) as service:
        query = service.select("Employee.name", "Employee.age")
        assert list(parallel(query, row=row)) == [None, None]
    assert all(response.close_calls == 1 for response in session.responses)
    session = FixtureSession.service(version=30, rows=b'{"results":[\nnull,\nnull,\nnull\n],"wasSuccessful":true}\n')
    with Service(SERVICE_ROOT, session=session, compatibility=profile) as service:
        with pytest.raises(ParallelExecutionError):
            next(parallel(service.select("Employee.name", "Employee.age"), row=row))
    assert all(response.close_calls == 1 for response in session.responses)
