from __future__ import annotations

import os


def resolve_verify_tls(verify_tls):
    if verify_tls is None:
        return True
    if isinstance(verify_tls, bool):
        return verify_tls
    if isinstance(verify_tls, (str, os.PathLike)):
        return verify_tls
    raise TypeError("verify_tls must be a bool, str, pathlib.Path, or None")


def close_resource_quietly(resource) -> None:
    close_fn = getattr(resource, "close", None)
    if not callable(close_fn):
        return
    try:
        close_fn()
    except Exception:
        return


def close_session_quietly(session) -> None:
    close_resource_quietly(session)


def close_response_quietly(response) -> None:
    close_resource_quietly(response)

