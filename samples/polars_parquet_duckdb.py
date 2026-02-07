"""Minimal end-to-end analytics workflow sample.

Demonstrates:
- Query execution
- Polars DataFrame materialization
- Parquet export
- DuckDB SQL querying

Targets Python 3.14+.
"""

from __future__ import annotations

from pathlib import Path

from intermine314.webservice import Service

SERVICE_ROOT = "https://www.flymine.org/query/service"
OUTPUT_BASE = Path("samples/output/polars_parquet_duckdb")


def main() -> None:
    service = Service(SERVICE_ROOT)

    query = service.new_query("Gene")
    query.add_view("Gene.symbol", "Gene.length", "Gene.organism.shortName")

    # DataFrame from query results (Polars)
    df = query.dataframe(size=2000, batch_size=1000)
    print("DataFrame shape:", df.shape)
    print(df.head(5))

    # Stream query results to Parquet partition files
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_BASE / "results_parquet"
    query.to_parquet(str(parquet_path), size=2000, batch_size=1000)
    print("Parquet path:", parquet_path)

    # Query Parquet using DuckDB
    con = query.to_duckdb(str(parquet_path), table="results")

    print("Rows:", con.execute("select count(*) from results").fetchall())
    print(
        "By organism:",
        con.execute(
            'select "Gene.organism.shortName", count(*) from results group by 1 order by 2 desc limit 10'
        ).fetchall(),
    )


if __name__ == "__main__":
    main()
