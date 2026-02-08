"""Modern sample: query alleles and run analytics with Polars + DuckDB.

This sample targets Python 3.14+ and the intermine314 package line.
"""

from __future__ import annotations

from samples.common import (
    DEFAULT_PREVIEW_LIMIT,
    DEFAULT_SERVICE_ROOT,
    export_parquet_and_open_duckdb,
    parquet_head,
    preview_rows,
    sample_output_dir,
)
from intermine314.webservice import Service

GENE_SYMBOLS = ["zen", "eve", "bib", "h"]
OUTPUT_SUBDIR = "alleles"
RESULT_SIZE = 5_000


def build_query(service: Service):
    query = service.new_query("Gene")
    query.add_view(
        "Gene.symbol",
        "Gene.name",
        "Gene.length",
        "Gene.alleles.symbol",
        "Gene.alleles.alleleClass",
    )
    query.add_constraint("Gene.symbol", "ONE OF", GENE_SYMBOLS)
    query.add_constraint("Gene.alleles.symbol", "IS NOT NULL")
    query.add_sort_order("Gene.symbol", "asc")
    return query


def main() -> None:
    service = Service(DEFAULT_SERVICE_ROOT)
    query = build_query(service)
    output_dir = sample_output_dir(OUTPUT_SUBDIR)

    print("Parallel preview:")
    preview_rows(query, limit=DEFAULT_PREVIEW_LIMIT)

    parquet_path, con = export_parquet_and_open_duckdb(
        query,
        output_dir=output_dir,
        parquet_name="parquet",
        table_name="alleles",
        result_size=RESULT_SIZE,
    )
    total_rows = con.execute("select count(*) from alleles").fetchone()[0]
    print("\nExported rows:", total_rows)
    print("\nParquet path:", parquet_path)
    print("\nPolars head(10):")
    print(parquet_head(parquet_path, limit=10))

    # DuckDB SQL analytics over Parquet files
    top_classes = con.execute(
        """
        select
          "Gene.alleles.alleleClass" as allele_class,
          count(*) as n
        from alleles
        group by 1
        order by n desc
        limit 10
        """
    ).fetchall()
    print("\nTop allele classes:")
    for row in top_classes:
        print(row)


if __name__ == "__main__":
    main()
