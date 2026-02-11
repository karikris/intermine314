import unittest

from benchmarks.bench_constants import BATCH_SIZE_TEST_CHUNK_ROWS
from benchmarks.benchmarks import assign_workers_for_chunk_size, resolve_batch_size_chunk_rows


class TestBatchSizeBenchmarkHelpers(unittest.TestCase):
    def test_resolve_batch_size_chunk_rows_constant(self):
        resolved = resolve_batch_size_chunk_rows("BATCH_SIZE_TEST_CHUNK_ROWS")
        self.assertEqual(resolved, sorted(BATCH_SIZE_TEST_CHUNK_ROWS))

    def test_assign_workers_prefers_higher_for_small_chunks(self):
        workers = [4, 6, 8, 12]
        high_parallel = assign_workers_for_chunk_size(50, 10_000, workers)
        low_parallel = assign_workers_for_chunk_size(10_000, 10_000, workers)
        self.assertEqual(high_parallel, 12)
        self.assertEqual(low_parallel, 4)
        self.assertGreaterEqual(high_parallel, low_parallel)

    def test_assign_workers_handles_empty_profile(self):
        assigned = assign_workers_for_chunk_size(1_000, 10_000, [])
        self.assertGreater(assigned, 0)


if __name__ == "__main__":
    unittest.main()
