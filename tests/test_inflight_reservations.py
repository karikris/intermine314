"""A page's reservation survives submission, buffering, and partial emission."""

import pytest

from intermine314.query import inflight


@pytest.fixture
def queue(monkeypatch):
    monkeypatch.setattr(inflight, "estimate_rows_payload_bytes", lambda rows: rows[0]["bytes"])
    return inflight.BoundedInflightQueue(inflight_limit=4, max_inflight_bytes_estimate=100)


def complete(queue, index, estimate):
    queue.observe_completed_page(page_index=index, rows=[{"bytes": estimate}])


def test_warmup_and_emitting_pages_remain_charged_until_release(queue):
    assert queue.can_submit()
    queue.reserve_page(0)
    assert not queue.can_submit()
    complete(queue, 0, 30)
    queue.start_emitting(0)
    assert queue.stats_fields()["outstanding_pages"] == 1
    assert queue.stats_fields()["emitting_pages"] == 1
    assert queue.stats_fields()["estimated_inflight_bytes"] == 30
    for index in (1, 2):
        assert queue.can_submit()
        queue.reserve_page(index)
    assert not queue.can_submit()  # 90 + the next 30-byte reservation exceeds 100.
    assert queue.stats_fields()["bytes_cap_hits"] == 2
    queue.release_page(0)
    assert queue.can_submit()
    assert queue.stats_fields()["estimated_inflight_bytes"] == 60


def test_completed_estimate_replaces_reservation_and_stays_in_buffer(queue):
    queue.reserve_page(0)
    complete(queue, 0, 40)
    queue.release_page(0)
    queue.reserve_page(1)
    queue.reserve_page(2)
    complete(queue, 2, 80)
    stats = queue.stats_fields()
    assert stats["estimated_inflight_bytes"] == 120
    assert stats["estimated_buffered_bytes"] == 80
    assert stats["max_estimated_inflight_bytes"] == 120
    assert stats["peak_buffered_pages"] == 1
    assert not queue.can_submit()
    queue.start_emitting(2)
    assert queue.stats_fields()["estimated_inflight_bytes"] == 120
    queue.release_page(2)
    assert queue.stats_fields()["estimated_inflight_bytes"] == 40
    assert queue.can_submit()


def test_oversized_page_is_allowed_only_in_empty_window(queue):
    queue.reserve_page(0)
    complete(queue, 0, 1000)
    assert not queue.can_submit()
    queue.release_page(0)
    assert queue.can_submit()  # Progress is possible even when the estimate exceeds the budget.
    queue.reserve_page(1)
    assert queue.stats_fields()["estimated_inflight_bytes"] == 1000
    assert not queue.can_submit()


def test_estimator_failure_drains_existing_work_then_admits_one_page(queue):
    queue.reserve_page(0)
    complete(queue, 0, 20)
    queue.release_page(0)
    for index in (1, 2, 3):
        queue.reserve_page(index)
    complete(queue, 2, None)
    assert not queue.stats_fields()["bytes_cap_active"]
    assert queue.stats_fields()["bytes_estimator_failures"] == 1
    for index in (1, 2, 3):
        assert not queue.can_submit()
        queue.release_page(index)
    assert queue.can_submit()
    queue.reserve_page(4)
    assert not queue.can_submit()


def test_page_cap_applies_without_byte_estimation(monkeypatch):
    monkeypatch.setattr(inflight, "estimate_rows_payload_bytes", lambda rows: pytest.fail("unexpected estimation"))
    queue = inflight.BoundedInflightQueue(inflight_limit=2, max_inflight_bytes_estimate=None)
    for index in (0, 1):
        assert queue.can_submit()
        queue.reserve_page(index)
        queue.observe_completed_page(page_index=index, rows=[object()])
    assert not queue.can_submit()
    queue.start_emitting(0)
    assert not queue.can_submit()
    queue.release_page(0)
    assert queue.can_submit()
    assert queue.stats_fields()["peak_outstanding_pages"] == 2
