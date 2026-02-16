import unittest

from intermine314.query import builder as query_builder


class _KeysetIdProbeQuery:
    def __init__(self, dataset, tracker):
        self._dataset = list(dataset)
        self._tracker = tracker
        self.joins = []
        self.uncoded_constraints = []
        self.do_verification = False
        self._sort_order_list = query_builder.SortOrderList()

    def clone(self):
        self._tracker["clone_calls"] += 1
        clone = _KeysetIdProbeQuery(self._dataset, self._tracker)
        self._tracker["prepared_query"] = clone
        return clone

    def clear_view(self):
        return None

    def add_view(self, *_views):
        return None

    def add_sort_order(self, *_args):
        return None

    def prefix_path(self, path):
        return str(path)

    def verify_constraint_paths(self, _cons):  # pragma: no cover - defensive
        return None

    def results(self, row="list", start=0, size=None):
        _ = (row, start)
        size_limit = int(size) if size is not None else None
        cursor = None
        for con in self.uncoded_constraints:
            if isinstance(con, query_builder.constraints.BinaryConstraint):
                cursor = str(con.value)
                break

        emitted = 0
        for value in self._dataset:
            token = str(value)
            if cursor is not None and token <= cursor:
                continue
            yield [token]
            emitted += 1
            if size_limit is not None and emitted >= size_limit:
                return


class _PreparedChunkQuery:
    def __init__(self, tracker):
        self._tracker = tracker
        self.uncoded_constraints = []
        self.do_verification = False
        self._sort_order_list = query_builder.SortOrderList()

    def prefix_path(self, path):
        return str(path)

    def verify_constraint_paths(self, _cons):  # pragma: no cover - defensive
        return None

    def add_sort_order(self, *_args):
        return None

    def results(self, row="dict", start=0, size=None):
        _ = (start, size)
        values = []
        for con in self.uncoded_constraints:
            if isinstance(con, query_builder.constraints.MultiConstraint):
                values = list(con.values)
                break
        self._tracker["seen_batches"].append(list(values))
        for token in values:
            if row == "list":
                yield [token]
            else:
                yield {"id": token}


class _KeysetRunHarness:
    def __init__(self, batches):
        self._batches = [list(batch) for batch in batches]
        self.clone_calls = 0
        self.tracker = {"seen_batches": []}
        self.prepared_query = None

    def _resolve_keyset_path(self, keyset_path):
        return str(keyset_path or "Gene.id")

    def _iter_keyset_ids(self, keyset_path, keyset_batch_size):
        _ = (keyset_path, keyset_batch_size)
        for batch in self._batches:
            yield list(batch)

    def clone(self):
        self.clone_calls += 1
        self.prepared_query = _PreparedChunkQuery(self.tracker)
        return self.prepared_query


class TestQueryParallelKeyset(unittest.TestCase):
    def test_iter_keyset_ids_reuses_prepared_query(self):
        tracker = {"clone_calls": 0, "prepared_query": None}
        harness = _KeysetIdProbeQuery(
            dataset=["001", "002", "003", "004", "005"],
            tracker=tracker,
        )

        batches = list(query_builder.Query._iter_keyset_ids(harness, "Gene.id", 2))

        self.assertEqual(batches, [["001", "002"], ["003", "004"], ["005"]])
        self.assertEqual(tracker["clone_calls"], 1)
        prepared_query = tracker["prepared_query"]
        self.assertIsNotNone(prepared_query)
        binary_constraints = [
            con
            for con in prepared_query.uncoded_constraints
            if isinstance(con, query_builder.constraints.BinaryConstraint)
        ]
        self.assertLessEqual(len(binary_constraints), 1)

    def test_run_parallel_keyset_reuses_chunk_query_and_cursor_constraint(self):
        harness = _KeysetRunHarness([["1", "2"], ["3", "4"], ["5"]])

        rows = list(
            query_builder.Query._run_parallel_keyset(
                harness,
                row="dict",
                size=None,
                page_size=100,
                max_workers=2,
                ordered=True,
                prefetch=2,
                keyset_path="Gene.id",
                keyset_batch_size=2,
            )
        )

        self.assertEqual([row["id"] for row in rows], ["1", "2", "3", "4", "5"])
        self.assertEqual(harness.clone_calls, 1)
        self.assertEqual(harness.tracker["seen_batches"], [["1", "2"], ["3", "4"], ["5"]])
        self.assertIsNotNone(harness.prepared_query)
        multi_constraints = [
            con
            for con in harness.prepared_query.uncoded_constraints
            if isinstance(con, query_builder.constraints.MultiConstraint)
        ]
        self.assertEqual(len(multi_constraints), 1)


if __name__ == "__main__":
    unittest.main()
