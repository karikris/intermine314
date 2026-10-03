"""Offline smoke check for the supported CSV -> Parquet -> SQL -> Polars workflow."""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from intermine314.export import import_csv, query_parquet


def main() -> None:
    import polars as pl

    source = StringIO("id,code\n9007199254740993,0007\n9007199254740994,0008\n")
    with TemporaryDirectory(prefix="intermine314-smoke-") as temporary:
        target = Path(temporary) / "results.parquet"
        import_csv(source, target, csv_options={"schema_overrides": {"id": pl.Int64, "code": pl.String}})
        frame = query_parquet(target, "SELECT * FROM results ORDER BY id")
        assert isinstance(frame, pl.DataFrame)
        assert frame.to_dicts() == [
            {"id": 9007199254740993, "code": "0007"},
            {"id": 9007199254740994, "code": "0008"},
        ]
        assert not source.closed
    print(json.dumps({"status": "ok", "rows": frame.height, "dataframe": "Polars", "storage": "Parquet"}))


if __name__ == "__main__":
    main()
