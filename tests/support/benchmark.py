"""Owned benchmark storage builders; no engine imported here."""
from decimal import Decimal

from benchmarks import bench_storage_compare as storage

ROWS = [
    {"Gene.id": 2**53 + 1, "Gene.value": Decimal("1.234567890123456789"), "Gene.code": "0007"},
    {"Gene.id": 2**53 + 2, "Gene.value": None, "Gene.code": "0008"},
    {"Gene.id": 2**53 + 3, "Gene.value": Decimal("9.2"), "Gene.code": "0009"},
]

class Page:
    def __init__(self, rows, fail=False):
        self.rows = iter(rows)
        self.fail = fail
        self.closed = False
        self.read = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.fail and self.read == 1:
            raise OSError("connection interrupted after a row")
        self.read += 1
        return next(self.rows)

    def close(self):
        self.closed = True


class Query:
    views = list(ROWS[0])

    def __init__(self, fail_calls=()):
        self.pages = []
        self.calls = []
        self.fail_calls = fail_calls

    def results(self, *, row, start, size):
        assert row == "dict"
        self.calls.append((start, size))
        page = Page(ROWS[start:start + size], len(self.calls) in self.fail_calls)
        self.pages.append(page)
        # The returned iterable and its actual iterator need not be identical.
        return IterablePage(page)


class IterablePage:
    def __init__(self, page):
        self.page = page

    def __iter__(self):
        return self.page


def export(monkeypatch, tmp_path, query, *, retries=2):
    monkeypatch.setattr(storage, "get_legacy_service_class", lambda: object)
    monkeypatch.setattr(storage, "_configure_legacy_intermine_transport", lambda **kw: None)
    monkeypatch.setattr(storage, "make_query", lambda *a, **kw: query)
    monkeypatch.setattr(storage, "count_with_retry", lambda *a, **kw: (3, 0, 0, 0))
    monkeypatch.setattr(storage, "_retry_wait_seconds", lambda attempt: 0)
    monkeypatch.setattr(storage, "_retriable_exceptions_for_mode", lambda mode: (OSError,))
    return storage._legacy_export_parquet(
        mine_url="https://example.test/service", rows_target=3, page_size=2,
        parquet_path=tmp_path / "reference.parquet", query_root_class="Gene",
        query_views=list(ROWS[0]), query_joins=[], transport_mode="direct",
        tor_proxy_url_value=None, timeout_seconds=1, max_retries=retries,
    )

