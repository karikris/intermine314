VERSION = "0.1.4"


def fetch_from_mine(*args, **kwargs):
    from intermine314.fetch import fetch_from_mine as _fetch_from_mine

    return _fetch_from_mine(*args, **kwargs)
