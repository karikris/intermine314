import tempfile
import unittest
from pathlib import Path

from benchmarks.bench_io import (
    analytics_columns,
    bench_parquet_join_engines,
    compare_csv_parquet_parity,
    compare_parquet_parity,
    csv_parquet_size_stats,
    normalize_sampling_mode,
)


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
            self.assertGreaterEqual(result["stage_timings_seconds"]["canonical_input_prepare_seconds"], 0.0)
            self.assertEqual(
                result["semantics"]["engine_only_delta"]["excludes"],
                ["canonical_input_prepare_seconds"],
            )
            self.assertEqual(len(result["duckdb"]["row_counts"]), 1)
            self.assertEqual(len(result["polars"]["row_counts"]), 1)
            self.assertEqual(result["duckdb"]["row_counts"][0], result["polars"]["row_counts"][0])
            self.assertTrue(Path(result["duckdb"]["artifacts"][0]).exists())
            self.assertTrue(Path(result["polars"]["artifacts"][0]).exists())
            self.assertTrue(Path(result["semantics"]["canonical_join_inputs"]["base_path"]).exists())

    def test_normalize_sampling_mode_validation(self):
        self.assertEqual(normalize_sampling_mode("head"), "head")
        self.assertEqual(normalize_sampling_mode("stride"), "stride")
        with self.assertRaises(ValueError):
            normalize_sampling_mode("invalid")

    def test_compare_csv_parquet_parity_equivalent(self):
        try:
            import polars as pl
        except Exception:
            raise unittest.SkipTest("polars not installed")

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "rows.csv"
            parquet_path = Path(tmp) / "rows.parquet"

            csv_path.write_text(
                "id,val\n"
                "g1,10\n"
                "g2,20\n"
                "g3,30\n",
                encoding="utf-8",
            )
            pl.read_csv(csv_path).write_parquet(parquet_path)

            parity = compare_csv_parquet_parity(
                csv_path=csv_path,
                parquet_path=parquet_path,
                sample_mode="head",
                sample_size=3,
                sort_by="id",
            )

            self.assertTrue(parity["rows"]["match"])
            self.assertTrue(parity["sample"]["hash_match"])
            self.assertTrue(parity["equivalent"])

    def test_compare_parquet_parity_detects_different_rows(self):
        try:
            import polars as pl
        except Exception:
            raise unittest.SkipTest("polars not installed")

        with tempfile.TemporaryDirectory() as tmp:
            left_path = Path(tmp) / "left.parquet"
            right_path = Path(tmp) / "right.parquet"
            pl.DataFrame({"id": ["g1", "g2"], "value": [1, 2]}).write_parquet(left_path)
            pl.DataFrame({"id": ["g1", "g2"], "value": [1, 99]}).write_parquet(right_path)

            parity = compare_parquet_parity(
                left_parquet_path=left_path,
                right_parquet_path=right_path,
                sample_mode="head",
                sample_size=2,
                sort_by="id",
            )

            self.assertTrue(parity["rows"]["match"])
            self.assertFalse(parity["sample"]["hash_match"])
            self.assertFalse(parity["equivalent"])


if __name__ == "__main__":
    unittest.main()
