import unittest

from benchmarking import bench_constants as bc


class TestBenchmarkConstants(unittest.TestCase):
    def test_resolve_matrix_rows_constants(self):
        self.assertEqual(
            bc.resolve_matrix_rows_constant("SMALL_MATRIX_ROWS"),
            bc.rows_to_csv(bc.SMALL_MATRIX_ROWS),
        )
        self.assertEqual(
            bc.resolve_matrix_rows_constant("large_matrix_rows"),
            bc.rows_to_csv(bc.LARGE_MATRIX_ROWS),
        )

    def test_resolve_matrix_rows_passthrough(self):
        self.assertEqual(bc.resolve_matrix_rows_constant("1,2,3"), "1,2,3")

    def test_shared_constants_exist(self):
        self.assertGreater(bc.DEFAULT_MATRIX_GROUP_SIZE, 0)
        self.assertTrue("auto" in bc.AUTO_WORKER_TOKENS)
        self.assertTrue(bc.DEFAULT_PARQUET_COMPRESSION)
        self.assertTrue(bc.DEFAULT_MATRIX_STORAGE_DIR)


if __name__ == "__main__":
    unittest.main()
