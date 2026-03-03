from __future__ import annotations

import re

VALID_PARQUET_COMPRESSIONS = frozenset({"zstd", "snappy", "gzip", "brotli", "lz4", "uncompressed"})
DUCKDB_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
