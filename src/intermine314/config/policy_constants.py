from __future__ import annotations

import re

DEFAULT_HTTP_RETRY_TOTAL = 5
DEFAULT_HTTP_RETRY_BACKOFF_SECONDS = 0.4
DEFAULT_HTTP_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
DEFAULT_HTTP_RETRY_METHODS = ("GET", "POST", "PUT", "DELETE")

DEFAULT_PARQUET_COMPRESSION = "zstd"
VALID_PARQUET_COMPRESSIONS = frozenset({"zstd", "snappy", "gzip", "brotli", "lz4", "uncompressed"})
DUCKDB_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

