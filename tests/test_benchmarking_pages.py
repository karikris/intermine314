import json
import tempfile
import unittest
from pathlib import Path

from benchmarking.bench_pages import append_benchmark_run_pages


class TestBenchmarkPages(unittest.TestCase):
    def test_append_benchmark_run_pages_creates_results_site(self):
        report = {
            "environment": {
                "timestamp_utc": "2026-02-10T12:00:00+00:00",
                "runtime_config": {
                    "benchmark_target": "maizemine",
                    "mine_url": "https://maizemine.rnet.missouri.edu/maizemine/service",
                    "repetitions": 3,
                },
            },
            "query_benchmarks": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp) / "results"
            written = append_benchmark_run_pages(site_dir, report, json_report_path="/tmp/report.json")
            for key in ("index_html", "run_html", "run_json", "index_json"):
                self.assertTrue(Path(written[key]).exists(), key)
            self.assertTrue((site_dir / ".nojekyll").exists())
            entries = json.loads(Path(written["index_json"]).read_text(encoding="utf-8"))
            self.assertTrue(entries)
            self.assertEqual(entries[0]["target"], "maizemine")


if __name__ == "__main__":
    unittest.main()
