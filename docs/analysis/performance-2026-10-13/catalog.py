"""Open the relocatable evidence catalog with its Parquet search directory.

Usage: python catalog.py 'SELECT count(*) FROM selected_measurements'
The database contains views only; canonical data remains in adjacent Parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def open_catalog(directory=None, *, read_only=True):
    directory = Path(directory or Path(__file__).parent).resolve()
    con = duckdb.connect(str(directory / 'analysis.duckdb'), read_only=read_only,
                         config={'threads': 4, 'memory_limit': '1GB'})
    try:
        con.execute('SET file_search_path = ?', [str(directory)])
    except BaseException:
        con.close()
        raise
    return con


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sql')
    args = parser.parse_args()
    with open_catalog() as con:
        result = con.execute(args.sql)
        columns = [column[0] for column in result.description]
        print(json.dumps([dict(zip(columns, row)) for row in result.fetchall()], default=str))


if __name__ == '__main__':
    main()
