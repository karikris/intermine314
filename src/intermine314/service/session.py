import base64
import json
import logging
import re
import sys
from collections import UserDict
from contextlib import closing
from itertools import groupby
from urllib.parse import urlencode, urlparse
try:
    import requests
except ImportError:  # pragma: no cover - requests is a declared dependency
    requests = None
try:
    import orjson
except ImportError:  # pragma: no cover - optional acceleration
    orjson = None

from intermine314.service.errors import WebserviceError
from intermine314.model import Attribute, Reference, Collection
from intermine314.config.constants import DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_REQUEST_TIMEOUT_SECONDS
from intermine314.service.transport import build_session, resolve_proxy_url

from intermine314 import VERSION

_RESULTS_HEADER_SUFFIX = '"results":['
_STATUS_KEY = '"wasSuccessful"'
_FALLBACK_GET_MAX_PAYLOAD_BYTES = 4096
_JSON_STATUS_BUFFER_MAX_CHARS = 64 * 1024
_JSON_ERROR_PREVIEW_MAX_CHARS = 2048


def _json_loads(payload):
    if orjson is not None:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return orjson.loads(payload)
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def _json_dumps(payload):
    if orjson is not None:
        return orjson.dumps(payload).decode("utf-8")
    return json.dumps(payload)


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


class EnrichmentLine(UserDict):
    """
    An object that represents a result returned from the enrichment service.
    ========================================================================

    These objects operate as dictionaries as well as objects with predefined
    properties.
    """

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return "EnrichmentLine(%s)" % self.data

    def __getattr__(self, name):
        if name is not None:
            key_name = name.replace("_", "-")
            if key_name in self.data:
                return self.data[key_name]
        raise AttributeError(name)


class ResultObject(object):
    """
    An object used to represent result records as returned in jsonobjects format
    ============================================================================

    These objects are backed by a row of data and the class descriptor that
    describes the object. They allow access in standard object style:

        >>> for gene in query.results():
        ...    print(gene.symbol)
        ...    print(map(lambda x: x.name, gene.pathways))

    All objects will have "id" and "type" properties. The type refers to the
    actual type of this object: if it is a subclass of the one requested, the
    subclass name will be returned. The "id" refers to the internal database id
    of the object, and is a guarantor of object identity.

    """

    def __init__(self, data, cld, view=None):
        if view is None:
            view = []
        stripped = [v[v.find(".") + 1 :] for v in view]
        self.selected_attributes = [v for v in stripped if "." not in v]
        self.reference_paths = dict(((k, list(i)) for k, i in groupby(stripped, lambda x: x[: x.find(".") + 1])))
        self._data = data
        # Make sure this object has the most specific class desc. possible
        class_name = data["class"]
        if "class" not in data or cld.name == class_name:
            self._cld = cld
        else:  # this could be a composed class - behave accordingly.
            self._cld = cld.model.get_class(class_name)

        self._attr_cache = {}

    def __str__(self):
        dont_show = set(["objectId", "class"])
        return "%s(%s)" % (
            self._cld.name,
            ",  ".join(
                "%s = %r" % (k, v)
                for k, v in list(self._data.items())
                if not isinstance(v, dict) and not isinstance(v, list) and k not in dont_show
            ),
        )

    def __repr__(self):
        dont_show = set(["objectId", "class"])
        return "%s(%s)" % (
            self._cld.name,
            ", ".join("%s = %r" % (k, getattr(self, k)) for k in self._data if k not in dont_show),
        )

    def __getattr__(self, name):
        if name in self._attr_cache:
            return self._attr_cache[name]

        if name == "type":
            return self._data["class"]

        fld = self._cld.get_field(name)
        attr = None
        if isinstance(fld, Attribute):
            if name in self._data:
                attr = self._data[name]
            if attr is None:
                attr = self._fetch_attr(fld)
        elif isinstance(fld, Reference):
            ref_paths = self._get_ref_paths(fld)
            if name in self._data:
                data = self._data[name]
            else:
                data = self._fetch_reference(fld)
            if isinstance(fld, Collection):
                if data is None:
                    attr = []
                else:
                    attr = [ResultObject(x, fld.type_class, ref_paths) for x in data]
            else:
                if data is None:
                    attr = None
                else:
                    attr = ResultObject(data, fld.type_class, ref_paths)
        else:
            raise WebserviceError("Inconsistent model - This should never happen")
        self._attr_cache[name] = attr
        return attr

    def _get_ref_paths(self, fld):
        if fld.name + "." in self.reference_paths:
            return self.reference_paths[fld.name + "."]
        else:
            return []

    @property
    def id(self):
        """Return the internal DB identifier of this object. Or None if this is not an InterMine object"""
        return self._data.get("objectId")

    def _fetch_attr(self, fld):
        if fld.name in self.selected_attributes:
            return None  # Was originally selected - no point asking twice
        c = self._cld
        if "id" not in c:
            return None  # Cannot reliably fetch anything without access to the objectId.
        q = c.model.service.query(c, fld).where(id=self.id)
        r = q.first()
        return r._data[fld.name] if fld.name in r._data else None

    def _fetch_reference(self, ref):
        if ref.name + "." in self.reference_paths:
            return None  # Was originally selected - no point asking twice.
        c = self._cld
        if "id" not in c:
            return None  # Cannot reliably fetch anything without access to the objectId.
        q = c.model.service.query(ref).outerjoin(ref).where(id=self.id)
        r = q.first()
        return r._data[ref.name] if ref.name in r._data else None


class ResultRow(object):
    """
    An object for representing a row of data received back from the server.
    =======================================================================

    ResultRows provide access to the fields of the row through index lookup. However,
    for convenience both list indexes and dictionary keys can be used. So the
    following all work:

        >>> # Assuming the view is "Gene.symbol", "Gene.organism.name":
        >>> row[0] == row["symbol"] == row["Gene.symbol"] == row(0) == row("symbol")
        ... True

    """

    def __init__(self, data, views):
        self.data = data
        self.views = views
        self.index_map = None

    def __len__(self):
        """Return the number of cells in this row"""
        return len(self.data)

    def __iter__(self):
        """Return the list view of the row, so each cell can be processed"""
        return iter(self.to_l())

    def _get_index_for(self, key):
        if self.index_map is None:
            self.index_map = {}
            for i in range(len(self.views)):
                view = self.views[i]
                headless_view = re.sub("^[^.]+.", "", view)
                self.index_map[view] = i
                self.index_map[headless_view] = i

        return self.index_map[key]

    def __str__(self):
        root = re.sub(r"\..*$", "", self.views[0])
        parts = [root + ":"]
        for view in self.views:
            short_form = re.sub("^[^.]+.", "", view)
            value = self[view]
            parts.append(short_form + "=" + repr(value))
        return " ".join(parts)

    def __call__(self, name):
        return self.__getitem__(name)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return self.data[key]
        else:
            index = self._get_index_for(key)
            return self.data[index]

    def to_l(self):
        """Return a list view of this row"""
        return list(self.data)

    def to_d(self):
        """Return a dictionary view of this row"""
        return dict(self.items())

    def items(self):
        return [(view, self[view]) for view in self.views]

    def keys(self):
        return list(self.views)

    def values(self):
        return self.to_l()


class TableResultRow(ResultRow):
    """
    A class for parsing results from the jsonrows data format.
    """

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]["value"]
        elif isinstance(key, slice):
            return [x["value"] for x in self.data[key]]
        else:
            index = self._get_index_for(key)
            return self.data[index]["value"]

    def to_l(self):
        """Return a list view of this row"""
        return [x["value"] for x in self.data]


def encode_str(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value
    return str(value)


def decode_binary(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def encode_dict(input_d):
    return {encode_str(k): encode_str(v) for k, v in input_d.items()}


def _is_blank_query_param_error(exc):
    text = str(exc).lower()
    return "query" in text and "must not be blank" in text


def _append_capped_text(parts, chunk, current_size, max_size):
    if current_size >= max_size:
        return current_size
    remaining = max_size - current_size
    if remaining <= 0:
        return current_size
    if len(chunk) > remaining:
        chunk = chunk[:remaining]
    parts.append(chunk)
    return current_size + len(chunk)


def _join_parts(parts):
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "".join(parts)


def _preview_for_error(text, max_chars=_JSON_ERROR_PREVIEW_MAX_CHARS):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def _extract_status_fragment(footer):
    idx = footer.find(_STATUS_KEY)
    if idx < 0:
        return None
    fragment = "{" + footer[idx:]
    closing_brace = fragment.rfind("}")
    if closing_brace < 0:
        return None
    return fragment[: closing_brace + 1]


class ResultIterator(object):
    """
    A facade over the internal iterator object
    ==========================================

    These objects handle the iteration over results
    in the formats requested by the user. They are responsible
    for generating an appropriate parser,
    connecting the parser to the results, and delegating
    iteration appropriately.
    """

    PARSED_FORMATS = frozenset(["rr", "list", "dict"])
    STRING_FORMATS = frozenset(["tsv", "csv", "count"])
    JSON_FORMATS = frozenset(["jsonrows", "jsonobjects", "json"])
    ROW_FORMATS = PARSED_FORMATS | STRING_FORMATS | JSON_FORMATS

    def __init__(self, service, path, params, rowformat, view, cld=None):
        """
        Constructor
        ===========

        Services are responsible for getting result iterators. You will
        not need to create one manually.

        @param root: The root path (eg: "https://www.flymine.org/query/service")
        @type root: string
        @param path: The resource path (eg: "/query/results")
        @type path: string
        @param params: The query parameters for this request
        @type params: dict
        @param rowformat: One of "rr", "object", "count", "dict", "list", "tsv", "csv", "jsonrows", "jsonobjects", "json"
        @type rowformat: string
        @param view: The output columns
        @type view: list
        @param opener: A url opener (user-agent)
        @type opener: urllib.URLopener

        @raise ValueError: if the row format is incorrect
        @raise WebserviceError: if the request is unsuccessful
        """
        if rowformat.startswith("object"):  # Accept "object", "objects", "objectformat", etc...
            rowformat = "jsonobjects"  # these are synonymous
        if rowformat not in self.ROW_FORMATS:
            raise ValueError(
                "'%s' is not one of the valid row formats (%s)" % (rowformat, repr(list(self.ROW_FORMATS)))
            )

        self.row = ResultRow if service.version >= 8 else TableResultRow

        if rowformat in self.PARSED_FORMATS:
            if service.version >= 8:
                params.update({"format": "json"})
            else:
                params.update({"format": "jsonrows"})
        elif rowformat == "tsv":
            params.update({"format": "tab"})
        else:
            params.update({"format": rowformat})

        self.url = service.root + path
        self.data = urlencode(encode_dict(params), True).encode("utf-8")
        self._payload_size = len(self.data)
        self.view = view
        self.opener = service.opener
        self.cld = cld
        self.rowformat = rowformat
        self._modern_json_rows = service.version >= 8
        self._it = None

    def _extract_row_values(self, payload):
        if self._modern_json_rows:
            return payload
        if isinstance(payload, list):
            return [cell.get("value") if isinstance(cell, dict) else cell for cell in payload]
        return payload

    def _row_as_list(self, payload):
        values = self._extract_row_values(payload)
        if isinstance(values, list):
            return list(values)
        if isinstance(values, tuple):
            return list(values)
        return self.row(payload, self.view).to_l()

    def _row_as_dict(self, payload):
        values = self._extract_row_values(payload)
        if isinstance(values, (list, tuple)) and len(values) == len(self.view):
            return dict(zip(self.view, values))
        return self.row(payload, self.view).to_d()

    def __len__(self):
        """
        Return the number of items in this iterator
        ===========================================

        Note that this requires iterating over the full result set, making the
        request in the process.
        """
        c = 0
        for x in self:
            c += 1
        return c

    def __iter__(self):
        """
        Return an iterator over the results
        ===================================

        Returns the internal iterator object.
        """
        try:
            con = self.opener.open(self.url, self.data)
        except WebserviceError as post_error:
            if _is_blank_query_param_error(post_error):
                raise
            if self._payload_size > _FALLBACK_GET_MAX_PAYLOAD_BYTES:
                raise post_error
            join_char = "&" if "?" in self.url else "?"
            fallback_url = self.url + join_char + self.data.decode("utf-8")
            try:
                con = self.opener.open(fallback_url, method="GET")
            except WebserviceError:
                raise post_error
        identity = lambda x: x
        if self.rowformat in {"tsv", "csv", "count"}:
            return FlatFileIterator(con, identity)
        if self.rowformat in {"json", "jsonrows"}:
            return JSONIterator(con, identity)
        if self.rowformat == "list":
            return JSONIterator(con, self._row_as_list)
        if self.rowformat == "rr":
            return JSONIterator(con, lambda x: self.row(x, self.view))
        if self.rowformat == "dict":
            return JSONIterator(con, self._row_as_dict)
        if self.rowformat == "jsonobjects":
            return JSONIterator(con, lambda x: ResultObject(x, self.cld, self.view))
        raise ValueError("Couldn't get iterator for " + self.rowformat)

    def __next__(self):
        if self._it is None:
            self._it = iter(self)
        try:
            return next(self._it)
        except StopIteration:
            self._it = None
            raise

    def next(self):
        return self.__next__()


class FlatFileIterator(object):
    """
    An iterator for handling results returned as a flat file (TSV/CSV).
    ===================================================================

    This iterator can be used as the sub iterator in a ResultIterator
    """

    def __init__(self, connection, parser):
        """
        Constructor
        ===========

        @param connection: The source of data
        @type connection: socket.socket
        @param parser: a handler for each row of data
        @type parser: Parser
        """
        self.connection = connection
        self.parser = parser

    def __iter__(self):
        return self

    def __next__(self):
        line = decode_binary(next(self.connection)).strip()
        if line.startswith("[ERROR]"):
            raise WebserviceError(line)
        return self.parser(line)

    def next(self):
        return self.__next__()


class JSONIterator(object):
    """
    An iterator for handling results returned in the JSONRows format
    ================================================================

    This iterator can be used as the sub iterator in a ResultIterator
    """

    LOG = logging.getLogger("JSONIterator")

    def __init__(self, connection, parser):
        """
        Constructor
        ===========

        @param connection: The source of data
        @type connection: socket.socket
        @param parser: a handler for each row of data
        @type parser: Parser
        """
        self.connection = connection
        self.parser = parser
        self.header = ""
        self.footer = ""
        self._header_parts = []
        self._footer_parts = []
        self._header_size = 0
        self._footer_size = 0
        self.parse_header()
        self._is_finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._is_finished:
            raise StopIteration
        return self.get_next_row_from_connection()

    def next(self):
        return self.__next__()

    def parse_header(self):
        """Reads out the header information from the connection"""
        self.LOG.debug("Connection = {0}".format(self.connection))
        try:
            while True:
                line = decode_binary(next(self.connection)).strip()
                self._header_size = _append_capped_text(
                    self._header_parts, line, self._header_size, _JSON_STATUS_BUFFER_MAX_CHARS
                )
                if line.endswith(_RESULTS_HEADER_SUFFIX):
                    self.header = _join_parts(self._header_parts)
                    return
        except StopIteration:
            self.header = _join_parts(self._header_parts)
            raise WebserviceError("The connection returned a bad header: " + _preview_for_error(self.header))

    def check_return_status(self):
        """
        Perform status checks
        =====================

        The footer containts information as to whether the result
        set was successfully transferred in its entirety. This
        method makes sure we don't silently accept an
        incomplete result set.

        @raise WebserviceError: if the footer indicates there was an error
        """
        self.footer = _join_parts(self._footer_parts)
        status_fragment = _extract_status_fragment(self.footer)
        if status_fragment is None:
            raise WebserviceError("Error parsing JSON status fragment: " + _preview_for_error(self.footer))
        try:
            info = _json_loads(status_fragment)
        except Exception as exc:
            raise WebserviceError(
                "Error parsing JSON status fragment: "
                + _preview_for_error(status_fragment)
                + " - "
                + str(exc)
            )

        if not info.get("wasSuccessful"):
            raise WebserviceError(info.get("statusCode"), info.get("error"))

    def get_next_row_from_connection(self):
        """
        Reads the connection to get the next row, and sends it to the parser

        @raise WebserviceError: if the connection is interrupted
        """
        next_row = None
        try:
            line = decode_binary(next(self.connection))
            if line.startswith("]"):
                self._footer_size = _append_capped_text(
                    self._footer_parts, line, self._footer_size, _JSON_STATUS_BUFFER_MAX_CHARS
                )
                for otherline in self.connection:
                    self._footer_size = _append_capped_text(
                        self._footer_parts,
                        decode_binary(otherline),
                        self._footer_size,
                        _JSON_STATUS_BUFFER_MAX_CHARS,
                    )
                self.check_return_status()
            else:
                line = line.strip().strip(",")
                if len(line) > 0:
                    try:
                        row = _json_loads(line)
                    except Exception as e:
                        raise WebserviceError(
                            "Error parsing line from results: '"
                            + _preview_for_error(line)
                            + "' - "
                            + str(e)
                        )
                    next_row = self.parser(row)
        except StopIteration:
            raise WebserviceError("Connection interrupted")

        if next_row is None:
            self._is_finished = True
            raise StopIteration
        else:
            return next_row


def encode_headers(headers):
    return {encode_header_value(k): encode_header_value(v) for k, v in headers.items()}


def encode_header_value(value):
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


class InterMineURLOpener(object):
    """
    Specific implementation of FancyURLopener for this client
    ================================================================

    Provides user agent and authentication headers, and handling of errors
    """

    USER_AGENT = "InterMine-Client-{0}/python-{1}".format(VERSION, sys.version_info)
    PLAIN_TEXT = "text/plain"
    JSON = "application/json"

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
    ):
        """
        Constructor
        ===========

        InterMineURLOpener((username, password)) S{->} InterMineURLOpener

        Return a new url-opener with the appropriate credentials
        """
        self.token = token
        if credentials and len(credentials) == 2:
            encoded = "{0}:{1}".format(*credentials).encode("utf8")
            base64string = "Basic {0}".format(base64.encodebytes(encoded)[:-1].decode("ascii"))
            self.auth_header = base64string
            self.using_authentication = True
        elif self.token is not None:
            token_header = "Token {0}".format(self.token)
            self.auth_header = token_header
            self.using_authentication = True
        else:
            self.using_authentication = False
        self.request_timeout = request_timeout
        self.proxy_url = resolve_proxy_url(proxy_url)
        self._timeout = self._normalize_timeout(timeout if timeout is not None else request_timeout)
        self._verify_tls = bool(verify_tls)
        if session is not None:
            self._session = session
        elif requests is not None:
            self._session = build_session(proxy_url=self.proxy_url, user_agent=None)
        else:
            self._session = None

    def clone(self):
        clone = InterMineURLOpener(
            request_timeout=self.request_timeout,
            session=self._session,
            timeout=self._timeout,
            verify_tls=self._verify_tls,
            proxy_url=self.proxy_url,
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
            self._session = build_session(proxy_url=self.proxy_url, user_agent=None)

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
