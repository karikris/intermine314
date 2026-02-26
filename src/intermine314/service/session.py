import sys
from contextlib import closing
import os
from urllib.parse import urlencode, urlparse

try:
    import requests
except ImportError:  # pragma: no cover - requests is a declared dependency
    requests = None

from intermine314 import VERSION
from intermine314.config.constants import DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_REQUEST_TIMEOUT_SECONDS
from intermine314.service.auth import build_basic_auth_header, build_token_auth_header
from intermine314.service import iterators as _iterators
from intermine314.service.errors import WebserviceError
from intermine314.service.iterators import (
    _FALLBACK_GET_MAX_PAYLOAD_BYTES,
    _JSON_ERROR_PREVIEW_MAX_CHARS,
    _JSON_STATUS_BUFFER_MAX_CHARS,
    EnrichmentLine,
    FlatFileIterator,
    JSONIterator,
    ResultObject,
    ResultRow,
    TableResultRow,
    _json_loads,
    decode_binary,
    encode_dict,
    encode_str,
)
from intermine314.service.transport import (
    build_session,
    enforce_tor_dns_safe_proxy_url,
    is_tor_proxy_url,
    resolve_proxy_url,
)


class _ResponseBodyAdapter(object):
    def __init__(self, response):
        self._response = response

    def read(self):
        return self._response.content

    def close(self):
        self._response.close()


class _ResponseStreamAdapter(object):
    def __init__(self, response):
        self._response = response
        self._iter = response.iter_lines(decode_unicode=False)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def read(self, size=-1):
        if size is None or size < 0:
            return self._response.content
        return self._response.raw.read(size)

    def close(self):
        self._response.close()


class ResultIterator(_iterators.ResultIterator):
    """
    Compatibility wrapper that keeps session-level monkeypatching behavior.
    """

    def __init__(self, service, path, params, rowformat, view, cld=None):
        super().__init__(service, path, params, rowformat, view, cld=cld)
        self.row = ResultRow if service.version >= 8 else TableResultRow


class InterMineURLOpener(object):
    """
    Specific implementation of FancyURLopener for this client
    ================================================================

    Provides user agent and authentication headers, and handling of errors
    """

    USER_AGENT = "InterMine-Client-{0}/python-{1}".format(VERSION, sys.version_info)
    PLAIN_TEXT = "text/plain"
    JSON = "application/json"

    @staticmethod
    def _resolve_verify_tls(verify_tls):
        if verify_tls is None:
            return True
        if isinstance(verify_tls, bool):
            return verify_tls
        if isinstance(verify_tls, (str, os.PathLike)):
            return verify_tls
        raise TypeError("verify_tls must be a bool, str, pathlib.Path, or None")

    def __init__(
        self,
        credentials=None,
        token=None,
        request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        *,
        session=None,
        timeout=None,
        verify_tls=True,
        proxy_url=None,
        tor_mode=None,
    ):
        """
        Constructor
        ===========

        InterMineURLOpener((username, password)) S{->} InterMineURLOpener

        Return a new url-opener with the appropriate credentials
        """
        self.token = token
        if credentials and len(credentials) == 2:
            self.auth_header = build_basic_auth_header(*credentials)
            self.using_authentication = True
        elif self.token is not None:
            self.auth_header = build_token_auth_header(str(self.token))
            self.using_authentication = True
        else:
            self.using_authentication = False
        self.request_timeout = request_timeout
        resolved_proxy_url = resolve_proxy_url(proxy_url)
        if tor_mode is None:
            self.tor_mode = bool(is_tor_proxy_url(resolved_proxy_url))
        else:
            self.tor_mode = bool(tor_mode)
        self.proxy_url = enforce_tor_dns_safe_proxy_url(
            resolved_proxy_url,
            tor_mode=self.tor_mode,
            context="InterMineURLOpener proxy_url",
        )
        self._timeout = self._normalize_timeout(timeout if timeout is not None else request_timeout)
        self._verify_tls = self._resolve_verify_tls(verify_tls)
        if session is not None:
            self._session = session
        elif requests is not None:
            self._session = build_session(proxy_url=self.proxy_url, user_agent=None, tor_mode=self.tor_mode)
        else:
            self._session = None

    def clone(self):
        clone = InterMineURLOpener(
            request_timeout=self.request_timeout,
            session=self._session,
            timeout=self._timeout,
            verify_tls=self._verify_tls,
            proxy_url=self.proxy_url,
            tor_mode=self.tor_mode,
        )
        clone.token = self.token
        clone.using_authentication = self.using_authentication
        clone._session = self._session
        if self.using_authentication:
            clone.auth_header = self.auth_header
        return clone

    @staticmethod
    def _normalize_timeout(timeout):
        if isinstance(timeout, (tuple, list)):
            if len(timeout) != 2:
                raise ValueError("timeout tuple/list must contain exactly (connect_timeout, read_timeout)")
            return (timeout[0], timeout[1])
        if timeout is None:
            return None
        value = float(timeout)
        if value <= 0:
            raise ValueError("timeout must be > 0")
        connect_timeout = min(float(DEFAULT_CONNECT_TIMEOUT_SECONDS), value)
        return (connect_timeout, value)

    def headers(self, content_type=None, accept=None):
        h = {"User-Agent": self.USER_AGENT}
        if self.using_authentication:
            h["Authorization"] = self.auth_header
        if content_type is not None:
            h["Content-Type"] = content_type
        if accept is not None:
            h["Accept"] = accept
        return h

    def post_plain_text(self, url, body):
        return self.post_content(url, body, InterMineURLOpener.PLAIN_TEXT)

    def post_content(self, url, body, mimetype, charset="utf-8"):
        content_type = "{0}; charset={1}".format(mimetype, charset)

        with closing(self.open(url, body, {"Content-Type": content_type})) as f:
            return f.read()

    def open(self, url, data=None, headers=None, method=None, timeout=None):
        url = self.prepare_url(url)
        effective_timeout = self._timeout if timeout is None else self._normalize_timeout(timeout)
        if data is None:
            buff = None
        elif isinstance(data, (bytes, bytearray)):
            buff = data
        else:
            buff = data.encode("utf-8")
        hs = self.headers()
        if headers is not None:
            hs.update(headers)
        if buff is not None and "Content-Type" not in hs:
            hs["Content-Type"] = "application/x-www-form-urlencoded; charset=utf-8"
        if method is None:
            method = "POST" if buff is not None else "GET"

        if self._session is None:
            if requests is None:  # pragma: no cover - requests is a declared dependency
                raise WebserviceError("Request library unavailable", 0, "requests unavailable", "requests unavailable")
            self._session = build_session(proxy_url=self.proxy_url, user_agent=None, tor_mode=self.tor_mode)

        try:
            resp = self._session.request(
                method,
                url,
                data=buff,
                headers=hs,
                stream=True,
                timeout=effective_timeout,
                verify=self._verify_tls,
            )
        except Exception as e:
            raise WebserviceError("Request failed", 0, str(e), str(e))

        if resp.status_code >= 400:
            fp = _ResponseBodyAdapter(resp)
            args = (
                url,
                fp,
                resp.status_code,
                resp.reason,
                resp.headers,
            )
            handler = {
                400: self.http_error_400,
                401: self.http_error_401,
                403: self.http_error_403,
                404: self.http_error_404,
                500: self.http_error_500,
            }.get(resp.status_code, self.http_error_default)
            handler(*args)
        return _ResponseStreamAdapter(resp)

    def read(self, url, data=None):
        with closing(self.open(url, data)) as conn:
            content = conn.read()
            return decode_binary(content)

    def prepare_url(self, url):
        # Generally unnecessary these days - will be deprecated one of these days.
        if self.token:
            token_param = urlencode(encode_dict(dict(token=self.token)))
            o = urlparse(url)
            if o.query:
                url += "&" + token_param
            else:
                url += "?" + token_param

        return url

    def delete(self, url):
        with closing(self.open(url, method="DELETE")) as f:
            return f.read()

    def http_error_default(self, url, fp, errcode, errmsg, headers):
        """Re-implementation of http_error_default, with content now supplied by default"""
        content = fp.read()
        fp.close()
        raise WebserviceError(errcode, errmsg, content)

    def http_error_400(self, url, fp, errcode, errmsg, headers, data=None):
        """
        Handle 400 HTTP errors, attempting to return informative error messages
        =======================================================================

        400 errors indicate that something about our request was incorrect

        @raise WebserviceError: in all circumstances

        """
        content = fp.read()
        fp.close()
        try:
            message = _json_loads(content)["error"]
        except Exception:
            message = content
        raise WebserviceError("There was a problem with our request", errcode, errmsg, message)

    def http_error_401(self, url, fp, errcode, errmsg, headers, data=None):
        """
        Handle 401 HTTP errors, attempting to return informative error messages
        =======================================================================

        401 errors indicate we don't have sufficient permission for the resource
        we requested - usually a list or a tempate

        @raise WebserviceError: in all circumstances

        """
        content = fp.read()
        fp.close()
        if self.using_authentication:
            auth = self.auth_header
            raise WebserviceError("Insufficient permissions - {0}".format(auth), errcode, errmsg, content)
        else:
            raise WebserviceError("No permissions - not logged in", errcode, errmsg, content)

    def http_error_403(self, url, fp, errcode, errmsg, headers, data=None):
        """
        Handle 403 HTTP errors, attempting to return informative error messages
        =======================================================================

        401 errors indicate we don't have sufficient permission for the resource
        we requested - usually a list or a tempate

        @raise WebserviceError: in all circumstances

        """
        content = fp.read()
        fp.close()
        try:
            message = _json_loads(content)["error"]
        except Exception:
            message = content
        if self.using_authentication:
            raise WebserviceError("Insufficient permissions", errcode, errmsg, message)
        else:
            raise WebserviceError("No permissions - not logged in", errcode, errmsg, message)

    def http_error_404(self, url, fp, errcode, errmsg, headers, data=None):
        """
        Handle 404 HTTP errors, attempting to return informative error messages
        =======================================================================

        404 errors indicate that the requested resource does not exist - usually
        a template that is not longer available.

        @raise WebserviceError: in all circumstances

        """
        content = fp.read()
        fp.close()
        try:
            message = _json_loads(content)["error"]
        except Exception:
            message = content
        raise WebserviceError("Missing resource", errcode, errmsg, message)

    def http_error_500(self, url, fp, errcode, errmsg, headers, data=None):
        """
        Handle 500 HTTP errors, attempting to return informative error messages
        =======================================================================

        500 errors indicate that the server borked during the request - ie: it wasn't
        our fault.

        @raise WebserviceError: in all circumstances

        """
        content = fp.read()
        fp.close()
        try:
            message = _json_loads(content)["error"]
        except Exception:
            message = content
        raise WebserviceError("Internal server error", errcode, errmsg, message)


__all__ = [
    "_FALLBACK_GET_MAX_PAYLOAD_BYTES",
    "_JSON_ERROR_PREVIEW_MAX_CHARS",
    "_JSON_STATUS_BUFFER_MAX_CHARS",
    "EnrichmentLine",
    "FlatFileIterator",
    "InterMineURLOpener",
    "JSONIterator",
    "ResultIterator",
    "ResultObject",
    "ResultRow",
    "TableResultRow",
    "decode_binary",
    "encode_dict",
    "encode_str",
]
