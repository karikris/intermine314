from __future__ import annotations

_POLARS_MODULE = None
_POLARS_CHECKED = False
_DUCKDB_MODULE = None
_DUCKDB_CHECKED = False


def optional_polars():
    global _POLARS_MODULE, _POLARS_CHECKED
    if not _POLARS_CHECKED:
        try:
            import polars as polars_module
        except ImportError:
            polars_module = None
        _POLARS_MODULE = polars_module
        _POLARS_CHECKED = True
    return _POLARS_MODULE


def require_polars(api_name):
    polars_module = optional_polars()
    if polars_module is None:
        raise ImportError(f"polars is required for {api_name}. Install with: pip install polars")
    return polars_module


def optional_duckdb():
    global _DUCKDB_MODULE, _DUCKDB_CHECKED
    if not _DUCKDB_CHECKED:
        try:
            import duckdb as duckdb_module
        except ImportError:
            duckdb_module = None
        _DUCKDB_MODULE = duckdb_module
        _DUCKDB_CHECKED = True
    return _DUCKDB_MODULE


def require_duckdb(api_name):
    duckdb_module = optional_duckdb()
    if duckdb_module is None:
        raise ImportError(f"duckdb is required for {api_name}. Install with: pip install duckdb")
    return duckdb_module


def quote_sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"
