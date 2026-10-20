"""Inspect portable Parquet evidence with temporary DuckDB views.

Run ``python -m benchmarks.evidence DIRECTORY SQL``. Nested table names use
relative path components joined by underscores, for example ``storage_trials``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb


def open_catalog(directory):
    directory = Path(directory).resolve(strict=True)
    con = duckdb.connect(config={'threads': 4, 'memory_limit': '1GB'})
    try:
        names = set()
        for path in sorted(directory.rglob('*.parquet')):
            name = re.sub(r'\W', '_', '_'.join(path.relative_to(directory).with_suffix('').parts))
            if name in names:
                raise ValueError(f'Ambiguous evidence table name: {name}')
            names.add(name)
            con.read_parquet(str(path)).create_view(name)
        return con
    except BaseException:
        con.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('sql')
    args = parser.parse_args()
    with open_catalog(args.directory) as con:
        result = con.execute(args.sql)
        print(json.dumps([dict(zip([c[0] for c in result.description], row))
                          for row in result.fetchall()], default=str))


if __name__ == '__main__':
    main()
