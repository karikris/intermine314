"""Expose the review's canonical Parquet tables through a local DuckDB catalog."""

from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent


def main():
    with duckdb.connect(str(HERE / "analysis.duckdb")) as connection:
        connection.execute("SET threads=2")
        connection.execute(
            "SET file_search_path='" + str(HERE).replace("'", "''") + "'"
        )
        for path in HERE.glob("*.parquet"):
            name = '"' + path.stem.replace('"', '""') + '"'
            file = "'" + path.name.replace("'", "''") + "'"
            connection.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({file})"
            )
    print("Open analysis.duckdb from this directory, or set file_search_path to it.")


if __name__ == "__main__":
    main()
