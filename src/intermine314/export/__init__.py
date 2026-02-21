"""Public export helpers."""

from intermine314.export.parquet import *  # noqa: F401,F403

__all__ = ["to_dataframe", "to_duckdb"]


def to_dataframe(query, **kwargs):
    return query.dataframe(**kwargs)


def to_duckdb(query, **kwargs):
    return query.to_duckdb(**kwargs)
