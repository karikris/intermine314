"""Standalone source opening for URLs, file paths, XML text and streams."""

from io import BytesIO, StringIO
from os import PathLike, fspath
from urllib import request


def openAnything(source):
    """Return a readable source; borrowed streams retain their ownership."""
    if hasattr(source, "read"):
        return source
    if isinstance(source, PathLike):
        source = fspath(source)
    if not isinstance(source, bytes):
        try:
            return request.urlopen(source)
        except ValueError, TypeError, OSError:
            pass
    try:
        return open(source)
    except ValueError, TypeError, OSError:
        pass
    if isinstance(source, bytes):
        return BytesIO(source)
    return StringIO(str(source))
