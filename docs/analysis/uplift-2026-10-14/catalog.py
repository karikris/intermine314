"""Expose local canonical evidence tables through relocatable DuckDB views."""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent


def connect():
    connection = duckdb.connect(str(ROOT / "analysis.duckdb"))
    connection.execute("SET file_search_path = ?", [str(ROOT)])
    for path in sorted(ROOT.glob("*.parquet")):
        name = path.stem.replace('"', '""')
        literal = path.name.replace("'", "''")
        connection.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet(\'{literal}\')')
    return connection


if __name__ == "__main__":
    with connect() as db:
        print(db.execute("SHOW TABLES").fetchall())
