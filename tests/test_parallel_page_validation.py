"""Parallel pages must finish their protocol before any row is accepted."""

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
