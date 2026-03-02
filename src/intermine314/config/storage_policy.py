from __future__ import annotations

from intermine314.config.policy_constants import (
    DEFAULT_PARQUET_COMPRESSION,
    DUCKDB_IDENTIFIER_PATTERN,
    VALID_PARQUET_COMPRESSIONS,
)


def valid_parquet_compressions() -> frozenset[str]:
    return VALID_PARQUET_COMPRESSIONS


def default_parquet_compression() -> str:
    from intermine314.config.runtime_defaults import get_runtime_defaults

    configured = str(get_runtime_defaults().storage_defaults.default_parquet_compression).strip().lower()
    if configured in VALID_PARQUET_COMPRESSIONS:
        return configured
    return DEFAULT_PARQUET_COMPRESSION


def validate_parquet_compression(codec: str | None) -> str:
    if codec is None:
        return default_parquet_compression()
    normalized = str(codec).strip().lower()
    if not normalized:
        return default_parquet_compression()
    if normalized not in VALID_PARQUET_COMPRESSIONS:
        choices = ", ".join(sorted(VALID_PARQUET_COMPRESSIONS))
        raise ValueError(f"Unsupported Parquet compression: {normalized}. Choose one of: {choices}")
    return normalized


def validate_duckdb_identifier(name: str) -> str:
    identifier = str(name)
    if not DUCKDB_IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError("DuckDB identifier must be a valid SQL identifier")
    return identifier
