"""Append task/verification evidence to Parquet, never to a mutable SQL table."""
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def record(table, **values):
    path = ROOT / f"{table}.parquet"
    rows = pq.read_table(path).to_pylist() if path.exists() else []
    rows.append(values)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("outcome")
    parser.add_argument("evidence")
    args = parser.parse_args()
    record("tasks", task=args.task, outcome=args.outcome, evidence=args.evidence)
