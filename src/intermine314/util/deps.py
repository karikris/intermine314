from __future__ import annotations

from functools import lru_cache
from importlib import import_module

MISSING_DEP_TEMPLATE = "{pkg} is required for {api_name}. Install with: pip install {pkg}"


@lru_cache(maxsize=4)
def _optional_module(module_name):
    try:
        return import_module(module_name)
    except ImportError:
        return None


def optional_polars():
    return _optional_module("polars")


def require_polars(api_name):
    polars_module = optional_polars()
    if polars_module is None:
        raise ImportError(MISSING_DEP_TEMPLATE.format(pkg="polars", api_name=api_name))
    return polars_module


def optional_duckdb():
    return _optional_module("duckdb")


def require_duckdb(api_name):
    duckdb_module = optional_duckdb()
    if duckdb_module is None:
        raise ImportError(MISSING_DEP_TEMPLATE.format(pkg="duckdb", api_name=api_name))
    return duckdb_module


def quote_sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"
