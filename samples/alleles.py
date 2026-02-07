"""Modern sample: query alleles and run analytics with Polars + DuckDB.

This sample targets Python 3.14+ and the intermine314 package line.
"""

from __future__ import annotations

from pathlib import Path

from intermine314.webservice import Service

SERVICE_ROOT = "https://www.flymine.org/query/service"
GENE_SYMBOLS = ["zen", "eve", "bib", "h"]
OUTPUT_DIR = Path("samples/output/alleles")


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


def preview_parallel(query, limit: int = 20) -> None:
    print("Parallel preview:")
    for idx, row in enumerate(query.run_parallel(row="dict", page_size=2000, max_workers=4, prefetch=4), start=1):
        print(row)
        if idx >= limit:
            break


def main() -> None:
    service = Service(SERVICE_ROOT)
    query = build_query(service)

    preview_parallel(query)

    # Polars DataFrame materialization
    df = query.dataframe(batch_size=5000)
    print("\nDataFrame shape:", df.shape)
    print(df.head(10))

    # Parquet export
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_dir = OUTPUT_DIR / "parquet"
    query.to_parquet(str(parquet_dir), batch_size=5000)
    print("\nParquet directory:", parquet_dir)

    # DuckDB SQL analytics over Parquet files
    con = query.to_duckdb(str(parquet_dir), table="alleles")
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
