import tempfile
import unittest
from pathlib import Path

from benchmarking.bench_io import analytics_columns, bench_parquet_join_engines, csv_parquet_size_stats


class TestBenchmarkIO(unittest.TestCase):
    def test_csv_parquet_size_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "a.csv"
            parquet_path = Path(tmp) / "a.parquet"
            csv_path.write_text("abcdef", encoding="utf-8")
            parquet_path.write_bytes(b"123")
            stats = csv_parquet_size_stats(csv_path, parquet_path)
            self.assertEqual(stats["csv_bytes"], 6)
            self.assertEqual(stats["parquet_bytes"], 3)
            self.assertEqual(stats["saved_bytes"], 3)
            self.assertGreater(stats["reduction_pct"], 0)

    def test_analytics_columns_deduplicates(self):
        cols = analytics_columns("Gene.CDS", "Gene.length", "Gene.CDS")
        self.assertEqual(cols, ["Gene.CDS", "Gene.length"])

    def test_bench_parquet_join_engines_smoke(self):
        try:
            import polars as pl
            import duckdb  # noqa: F401
        except Exception:
            raise unittest.SkipTest("polars/duckdb not installed")

        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "input.parquet"
            out_dir = Path(tmp) / "joins"
            pl.DataFrame(
                {
                    "id": ["g1", "g2", "g2", "g3"],
                    "a": [1, 2, 3, 4],
                    "b": ["x", "y", "y", "z"],
                    "c": [10.0, 20.0, 30.0, 40.0],
                }
            ).write_parquet(parquet_path)

            result = bench_parquet_join_engines(
                parquet_path=parquet_path,
                repetitions=1,
                output_dir=out_dir,
                join_key="id",
            )

            self.assertEqual(result["join_shape"], "two_full_outer_joins_three_tables")
            self.assertEqual(len(result["duckdb"]["row_counts"]), 1)
            self.assertEqual(len(result["polars"]["row_counts"]), 1)
            self.assertEqual(result["duckdb"]["row_counts"][0], result["polars"]["row_counts"][0])
            self.assertTrue(Path(result["duckdb"]["artifacts"][0]).exists())
            self.assertTrue(Path(result["polars"]["artifacts"][0]).exists())


if __name__ == "__main__":
    unittest.main()
