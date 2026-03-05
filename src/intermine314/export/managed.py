from __future__ import annotations


class ManagedDuckDBConnection:
    """Context manager wrapper for managed DuckDB lifecycle."""

    def __init__(self, connection, *, close_resource_quietly):
        self._connection = connection
        self._close_resource_quietly = close_resource_quietly
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._close_resource_quietly(self._connection)

    def __enter__(self):
        return self._connection

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        self.close()
        return False
