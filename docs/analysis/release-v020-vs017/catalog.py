"""Regenerate the summary and a relocatable local DuckDB evidence catalog."""

from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    with duckdb.connect(str(HERE / "analysis.duckdb")) as connection:
        connection.execute("SET threads=2")
        connection.execute("SET file_search_path=" + literal(HERE))
        for path in sorted(HERE.glob("*.parquet")):
            if path.stem == "summary":
                continue
            name = '"' + path.stem.replace('"', '""') + '"'
            connection.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM read_parquet({literal(path.name)})"
            )
        summary = connection.execute(
            """SELECT "case", variant, count(*) AS observations,
                      median(ns) / 1e6 AS median_ms,
                      quantile_cont(ns, 0.25) / 1e6 AS q1_ms,
                      quantile_cont(ns, 0.75) / 1e6 AS q3_ms,
                      median(python_peak_bytes) AS python_peak_bytes,
                      median(requests) AS requests,
                      median(data_bytes) AS data_bytes,
                      median(total_bytes) AS total_bytes,
                      median(parts) AS parts,
                      median(row_groups) AS row_groups
               FROM trials GROUP BY "case", variant ORDER BY "case", variant"""
        ).to_arrow_table()
        import pyarrow.parquet as pq

        pq.write_table(summary, HERE / "summary.parquet", compression="zstd")
        connection.execute(
            "CREATE OR REPLACE VIEW summary AS SELECT * FROM read_parquet('summary.parquet')"
        )
        print(connection.execute("SELECT * FROM summary").fetchall())
    print(
        "Open analysis.duckdb from this evidence directory, or set file_search_path to it."
    )


if __name__ == "__main__":
    main()
