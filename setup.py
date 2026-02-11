from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from setuptools import Command, setup

ANALYTICS_REQUIRED_METHODS = ("dataframe", "to_parquet", "to_duckdb")

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class AnalyticsCheckCommand(Command):
    description = "Run Polars/Parquet/DuckDB compatibility smoke check"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            import duckdb
            import polars as pl
        except Exception as exc:
            raise SystemExit(
                "analyticscheck requires core workflow dependencies (polars, duckdb). "
                "Install package dependencies first (for example: pip install -e .)"
            ) from exc

        from intermine314.query import Query

        missing = [name for name in ANALYTICS_REQUIRED_METHODS if not hasattr(Query, name)]
        if missing:
            raise SystemExit("Missing Query analytics API methods: " + ", ".join(missing))

        with TemporaryDirectory() as tmp:
            parquet_file = f"{tmp}/smoke.parquet"
            pl.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]}).write_parquet(parquet_file)
            con = duckdb.connect(database=":memory:")
            try:
                got = con.execute("select sum(x) from read_parquet(?)", [parquet_file]).fetchone()[0]
            finally:
                con.close()

        if got != 6:
            raise SystemExit(f"Unexpected DuckDB/Parquet result: {got}")

        print("analyticscheck: ok")


setup(cmdclass={"analyticscheck": AnalyticsCheckCommand})
