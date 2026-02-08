"""Minimal end-to-end analytics workflow sample.

Demonstrates:
- Query execution
- Polars DataFrame materialization
- Parquet export
- DuckDB SQL querying

Targets Python 3.14+.
"""

from __future__ import annotations

from samples.common import (
    DEFAULT_RESULT_SIZE,
    DEFAULT_SERVICE_ROOT,
    export_parquet_and_open_duckdb,
    parquet_head,
    sample_output_dir,
)
from intermine314.webservice import Service

OUTPUT_SUBDIR = "polars_parquet_duckdb"
RESULT_SIZE = DEFAULT_RESULT_SIZE


def main() -> None:
    service = Service(DEFAULT_SERVICE_ROOT)
    output_dir = sample_output_dir(OUTPUT_SUBDIR)

    query = service.new_query("Gene")
    query.add_view("Gene.symbol", "Gene.length", "Gene.organism.shortName")

    parquet_path, con = export_parquet_and_open_duckdb(
        query,
        output_dir=output_dir,
        parquet_name="results_parquet",
        table_name="results",
        result_size=RESULT_SIZE,
    )

    print("Rows:", con.execute("select count(*) from results").fetchone()[0])
    print("Polars head(5):")
    print(parquet_head(parquet_path, limit=5))
    print("Parquet path:", parquet_path)
    print(
        "By organism:",
        con.execute(
            'select "Gene.organism.shortName", count(*) from results group by 1 order by 2 desc limit 10'
        ).fetchall(),
    )


if __name__ == "__main__":
    main()
