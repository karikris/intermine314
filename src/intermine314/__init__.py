from intermine314._version import VERSION, __version__

__all__ = [
    "VERSION",
    "__version__",
    "fetch_from_mine",
]


def fetch_from_mine(*args, **kwargs):
    from intermine314.export.fetch import fetch_from_mine as _fetch_from_mine

    return _fetch_from_mine(*args, **kwargs)
