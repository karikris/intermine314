from io import BytesIO, StringIO
from urllib.parse import urlparse
from urllib.request import urlopen

from intermine314.service.transport import build_session, resolve_proxy_url


def _is_http_source(source) -> bool:
    try:
        parsed = urlparse(str(source))
    except Exception:
        return False
    return parsed.scheme in {"http", "https"}


def openAnything(source, *, session=None, timeout=None, verify_tls=True, proxy_url=None):
    # Already file-like.
    if hasattr(source, "read"):
        return source

    # Route HTTP(S) through the shared session transport to honor SOCKS/proxy settings.
    if _is_http_source(source):
        http_session = session or build_session(proxy_url=resolve_proxy_url(proxy_url), user_agent=None)
        response = http_session.get(str(source), timeout=timeout, verify=bool(verify_tls))
        response.raise_for_status()
        return BytesIO(response.content)

    # Local file path.
    try:
        return open(source)
    except (ValueError, IOError, OSError, TypeError):
        pass

    # Non-HTTP URL schemes (for example: ftp://, file://).
    try:
        return urlopen(source)
    except (ValueError, IOError, OSError, TypeError):
        pass

    return StringIO(str(source))


class ReadableException(Exception):
    def __init__(self, message, cause=None):
        self.message = message
        self.cause = cause

    def __str__(self):
        if self.cause is None:
            return repr(self.message)
        else:
            return repr(self.message) + repr(self.cause)
