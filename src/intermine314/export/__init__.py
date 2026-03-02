"""Public export helpers."""

__all__ = ["to_dataframe", "to_duckdb"]


def to_dataframe(query, **kwargs):
    return query.dataframe(**kwargs)


def to_duckdb(query, **kwargs):
    return query.to_duckdb(**kwargs)
