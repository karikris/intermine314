import logging

_LEGACY_LOG = logging.getLogger("intermine314.export")
_LEGACY_WARNING_EMITTED = False


def _warn_legacy_wrapper():
    global _LEGACY_WARNING_EMITTED
    if _LEGACY_WARNING_EMITTED:
        return
    _LEGACY_LOG.warning(
        "intermine314.export.polars_frame.to_dataframe is a legacy shim. "
        "Use query.dataframe(...) or intermine314.export.to_dataframe(...)."
    )
    _LEGACY_WARNING_EMITTED = True


def to_dataframe(query, **kwargs):
    _warn_legacy_wrapper()
    return query.dataframe(**kwargs)
