"""Parquet query builders with deferred runtime imports."""

def query_with_rows(rows, columns, *, profile="native"):
    from intermine314.query import Query
    query = Query(compatibility=profile)
    query.add_view(*columns)
    query.iter_rows = lambda **kwargs: iter(rows)
    return query

