from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


DEFAULT_SMALL_MATRIX_ROWS = (5_000, 10_000, 25_000)
DEFAULT_LARGE_MATRIX_ROWS = (50_000, 100_000, 250_000)
DEFAULT_BATCH_SIZE_TEST_ROWS = 10_000
DEFAULT_BATCH_SIZE_TEST_CHUNK_ROWS = (1_000, 2_500, 5_000, 7_500, 10_000)
DEFAULT_WARMUP_ROWS = 2_000
DEFAULT_PROGRESS_LOG_INTERVAL_ROWS = 100_000
DEFAULT_RETRY_BACKOFF_INITIAL_SECONDS = 2.0
DEFAULT_RETRY_BACKOFF_MAX_SECONDS = 12.0
DEFAULT_MATRIX_GROUP_SIZE = 3
AUTO_WORKER_TOKENS = frozenset({"auto", "registry", "mine"})
DEFAULT_PARQUET_COMPRESSION = "zstd"
DEFAULT_MATRIX_STORAGE_DIR = "/tmp/intermine314_matrix_storage"
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "benchmark-constants.toml"
_CONFIG_CACHE: dict[str, Any] | None = None


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


def _load_config_data() -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    if not _CONFIG_PATH.exists():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE
    try:
        with _CONFIG_PATH.open("rb") as fh:
            parsed = tomllib.load(fh)
    except Exception:
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE
    _CONFIG_CACHE = parsed if isinstance(parsed, dict) else {}
    return _CONFIG_CACHE


def _parse_positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return fallback
    return parsed if parsed > 0 else fallback


def _parse_positive_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return fallback
    return parsed if parsed > 0 else fallback


def _load_matrix_rows() -> tuple[tuple[int, ...], tuple[int, ...]]:
    data = _load_config_data()
    if not _CONFIG_PATH.exists():
        return DEFAULT_SMALL_MATRIX_ROWS, DEFAULT_LARGE_MATRIX_ROWS
    small_rows = _parse_rows(data.get("SMALL_MATRIX_ROWS"), DEFAULT_SMALL_MATRIX_ROWS)
    large_rows = _parse_rows(data.get("LARGE_MATRIX_ROWS"), DEFAULT_LARGE_MATRIX_ROWS)
    return small_rows, large_rows


def _load_batch_size_test_defaults() -> tuple[int, tuple[int, ...]]:
    data = _load_config_data()
    rows = _parse_positive_int(data.get("BATCH_SIZE_TEST_ROWS"), DEFAULT_BATCH_SIZE_TEST_ROWS)
    chunk_rows = _parse_rows(
        data.get("BATCH_SIZE_TEST_CHUNK_ROWS"),
        DEFAULT_BATCH_SIZE_TEST_CHUNK_ROWS,
    )
    return rows, chunk_rows


def _load_runtime_tuning_defaults() -> tuple[int, int, float, float]:
    data = _load_config_data()
    warmup_rows = _parse_positive_int(data.get("WARMUP_ROWS"), DEFAULT_WARMUP_ROWS)
    progress_interval_rows = _parse_positive_int(
        data.get("PROGRESS_LOG_INTERVAL_ROWS"),
        DEFAULT_PROGRESS_LOG_INTERVAL_ROWS,
    )
    backoff_initial_seconds = _parse_positive_float(
        data.get("RETRY_BACKOFF_INITIAL_SECONDS"),
        DEFAULT_RETRY_BACKOFF_INITIAL_SECONDS,
    )
    backoff_max_seconds = _parse_positive_float(
        data.get("RETRY_BACKOFF_MAX_SECONDS"),
        DEFAULT_RETRY_BACKOFF_MAX_SECONDS,
    )
    if backoff_max_seconds < backoff_initial_seconds:
        backoff_max_seconds = backoff_initial_seconds
    return warmup_rows, progress_interval_rows, backoff_initial_seconds, backoff_max_seconds


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
(
    WARMUP_ROWS,
    PROGRESS_LOG_INTERVAL_ROWS,
    RETRY_BACKOFF_INITIAL_SECONDS,
    RETRY_BACKOFF_MAX_SECONDS,
) = _load_runtime_tuning_defaults()
