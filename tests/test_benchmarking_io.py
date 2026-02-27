import tempfile
import unittest
from pathlib import Path

from benchmarks.bench_io import (
    analytics_columns,
    bench_engine_pipelines_from_parquet,
    bench_parquet_join_engines,
    build_stage_model,
    compare_csv_parquet_parity,
    compare_parquet_parity,
    export_matrix_storage_compare,
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

    def test_build_stage_model_schema(self):
        stage_model = build_stage_model(
            scenario_name="smoke",
            stages={"fetch": {"elapsed_seconds": 1.0, "rows_fetched": 10}},
        )
        self.assertEqual(stage_model["schema_version"], "benchmark_stage_model_v1")
        self.assertEqual(stage_model["scenario_name"], "smoke")
        self.assertTrue(stage_model["stages"]["fetch"]["enabled"])
        self.assertEqual(stage_model["stages"]["fetch"]["rows_fetched"], 10)
        self.assertFalse(stage_model["stages"]["analytics"]["enabled"])

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

    def test_bench_engine_pipelines_from_parquet_smoke(self):
        try:
            import polars as pl
            import duckdb  # noqa: F401
        except Exception:
            raise unittest.SkipTest("polars/duckdb not installed")

        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "input.parquet"
            out_dir = Path(tmp) / "engine"
            pl.DataFrame(
                {
                    "id": ["g1", "g2", "g2", "g3"],
                    "a": [1, 2, 3, 4],
                    "b": [10, 20, 30, 40],
                    "c": [100, 200, 300, 400],
                }
            ).write_parquet(parquet_path)

            result = bench_engine_pipelines_from_parquet(
                parquet_path=parquet_path,
                repetitions=1,
                output_dir=out_dir,
                join_key="id",
            )

            self.assertEqual(result["scenarios"]["elt_duckdb_pipeline"]["engine_step_input"], str(parquet_path))
            self.assertEqual(result["scenarios"]["etl_polars_pipeline"]["engine_step_input"], str(parquet_path))
            self.assertEqual(len(result["elt_duckdb_pipeline"]["runs"]), 1)
            self.assertEqual(len(result["etl_polars_pipeline"]["runs"]), 1)

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

            self.assertTrue(parity["schema"]["hash_match"])
            self.assertTrue(parity["rows"]["match"])
            self.assertTrue(parity["sample"]["hash_match"])
            self.assertTrue(parity["aggregate_invariants"]["hash_match"])
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
            self.assertTrue(parity["aggregate_invariants"]["hash_match"])
            self.assertFalse(parity["sample"]["hash_match"])
            self.assertFalse(parity["equivalent"])

    def test_export_matrix_storage_compare_offline_replay(self):
        try:
            import polars as pl
            import pandas  # noqa: F401
        except Exception:
            raise unittest.SkipTest("polars/pandas not installed")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "matrix"
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / "scenario.csv"
            parquet_path = output_dir / "scenario.parquet"
            csv_path.write_text("id,value\nk1,1\nk2,2\n", encoding="utf-8")
            pl.read_csv(csv_path).write_parquet(parquet_path)

            result = export_matrix_storage_compare(
                mine_url="https://example.org/mine",
                scenario_name="scenario",
                rows_target=2,
                page_size=2,
                workers=[2],
                include_legacy_baseline=False,
                mode_runtime_kwargs={},
                output_dir=output_dir,
                load_repetitions=1,
                query_root_class="Gene",
                query_views=["id", "value"],
                query_joins=[],
                parity_sample_mode="head",
                parity_sample_size=2,
                offline_replay=True,
            )

            self.assertTrue(result["offline_replay"])
            self.assertTrue(result["parity"]["equivalent"])
            self.assertIsNone(result["csv_export"]["seconds"])
            self.assertIsNone(result["parquet_export"]["conversion_seconds_from_csv"])
            self.assertEqual(result["stage_model"]["schema_version"], "benchmark_stage_model_v1")
            self.assertIn("fetch", result["stage_model"]["stages"])
            self.assertIn("duckdb_scan", result["stage_model"]["stages"])
            self.assertIn("polars_scan", result["stage_model"]["stages"])


if __name__ == "__main__":
    unittest.main()
