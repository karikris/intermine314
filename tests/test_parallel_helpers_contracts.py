from itertools import islice

import pytest

from intermine314.parallel import helpers


def test_ordering_normalize_mode_contracts():
    assert helpers.normalize_mode(True) == "ordered"
    assert helpers.normalize_mode(False) == "unordered"
    assert helpers.normalize_mode("window") == "window"
    assert helpers.normalize_mode("mostly_ordered") == "mostly_ordered"
    with pytest.raises(ValueError):
        helpers.normalize_mode("invalid")


def test_estimate_page_count_contracts():
    assert helpers.estimate_page_count(total_rows=0, page_size=100) == 0
    assert helpers.estimate_page_count(total_rows=100, page_size=100) == 1
    assert helpers.estimate_page_count(total_rows=101, page_size=100) == 2
    assert helpers.estimate_page_count(total_rows=-50, page_size=100) == 0
    with pytest.raises(ValueError):
        helpers.estimate_page_count(total_rows=100, page_size=0)
    with pytest.raises(TypeError):
        helpers.estimate_page_count(total_rows=0, page_size=100, minimum_one=True)


def test_iter_offset_pages_with_size():
    pages = list(helpers.iter_offset_pages(start=10, size=230, page_size=100))
    assert pages == [(10, 100), (110, 100), (210, 30)]


def test_iter_offset_pages_without_size():
    pages = list(islice(helpers.iter_offset_pages(start=5, size=None, page_size=20), 4))
    assert pages == [(5, 20), (25, 20), (45, 20), (65, 20)]


def test_iter_offset_pages_rejects_invalid_page_size():
    with pytest.raises(ValueError):
        list(helpers.iter_offset_pages(start=0, size=10, page_size=0))
