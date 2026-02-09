from __future__ import annotations

import tomllib
from pathlib import Path


DEFAULT_SMALL_MATRIX_ROWS = (5_000, 10_000, 25_000)
DEFAULT_LARGE_MATRIX_ROWS = (50_000, 100_000, 250_000)
DEFAULT_BATCH_SIZE_TEST_ROWS = 10_000
DEFAULT_BATCH_SIZE_TEST_CHUNK_ROWS = (50, 100, 500, 1_000, 2_500, 5_000, 10_000)
DEFAULT_MATRIX_GROUP_SIZE = 3
AUTO_WORKER_TOKENS = frozenset({"auto", "registry", "mine"})
DEFAULT_PARQUET_COMPRESSION = "zstd"
DEFAULT_MATRIX_STORAGE_DIR = "/tmp/intermine314_matrix_storage"
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "benchmark-constants.toml"


def _parse_rows(value, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return fallback
    tokens: list[str]
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
    elif isinstance(value, (list, tuple)):
        tokens = [str(token).strip() for token in value if str(token).strip()]
    else:
        return fallback
    if not tokens:
        return fallback
    out: list[int] = []
    try:
        for token in tokens:
            number = int(token)
            if number <= 0:
                return fallback
            out.append(number)
    except Exception:
        return fallback
    return tuple(out)


def _load_matrix_rows() -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not _CONFIG_PATH.exists():
        return DEFAULT_SMALL_MATRIX_ROWS, DEFAULT_LARGE_MATRIX_ROWS
    try:
        with _CONFIG_PATH.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return DEFAULT_SMALL_MATRIX_ROWS, DEFAULT_LARGE_MATRIX_ROWS
    small_rows = _parse_rows(data.get("SMALL_MATRIX_ROWS"), DEFAULT_SMALL_MATRIX_ROWS)
    large_rows = _parse_rows(data.get("LARGE_MATRIX_ROWS"), DEFAULT_LARGE_MATRIX_ROWS)
    return small_rows, large_rows


def _load_batch_size_test_defaults() -> tuple[int, tuple[int, ...]]:
    if not _CONFIG_PATH.exists():
        return DEFAULT_BATCH_SIZE_TEST_ROWS, DEFAULT_BATCH_SIZE_TEST_CHUNK_ROWS
    try:
        with _CONFIG_PATH.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return DEFAULT_BATCH_SIZE_TEST_ROWS, DEFAULT_BATCH_SIZE_TEST_CHUNK_ROWS

    rows = DEFAULT_BATCH_SIZE_TEST_ROWS
    raw_rows = data.get("BATCH_SIZE_TEST_ROWS")
    try:
        parsed_rows = int(raw_rows)
        if parsed_rows > 0:
            rows = parsed_rows
    except Exception:
        rows = DEFAULT_BATCH_SIZE_TEST_ROWS

    chunk_rows = _parse_rows(
        data.get("BATCH_SIZE_TEST_CHUNK_ROWS"),
        DEFAULT_BATCH_SIZE_TEST_CHUNK_ROWS,
    )
    return rows, chunk_rows


def rows_to_csv(rows: tuple[int, ...]) -> str:
    return ",".join(str(row) for row in rows)


def resolve_matrix_rows_constant(value: str) -> str:
    token = str(value).strip()
    upper = token.upper()
    if upper == "SMALL_MATRIX_ROWS":
        return rows_to_csv(SMALL_MATRIX_ROWS)
    if upper == "LARGE_MATRIX_ROWS":
        return rows_to_csv(LARGE_MATRIX_ROWS)
    if upper == "BATCH_SIZE_TEST_CHUNK_ROWS":
        return rows_to_csv(BATCH_SIZE_TEST_CHUNK_ROWS)
    return token


SMALL_MATRIX_ROWS, LARGE_MATRIX_ROWS = _load_matrix_rows()
BATCH_SIZE_TEST_ROWS, BATCH_SIZE_TEST_CHUNK_ROWS = _load_batch_size_test_defaults()
