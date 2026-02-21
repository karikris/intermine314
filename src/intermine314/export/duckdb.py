import logging

_LEGACY_LOG = logging.getLogger("intermine314.export")
_LEGACY_WARNING_EMITTED = False


def _warn_legacy_wrapper():
    global _LEGACY_WARNING_EMITTED
    if _LEGACY_WARNING_EMITTED:
        return
    _LEGACY_LOG.warning(
        "intermine314.export.duckdb.to_duckdb is a legacy shim. "
        "Use query.to_duckdb(...) or intermine314.export.to_duckdb(...)."
    )
    _LEGACY_WARNING_EMITTED = True


def to_duckdb(query, **kwargs):
    _warn_legacy_wrapper()
    return query.to_duckdb(**kwargs)
