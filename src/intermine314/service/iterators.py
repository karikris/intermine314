import json
import logging
import re
from collections import UserDict
from itertools import groupby
from urllib.parse import urlencode

try:
    import orjson
except ImportError:  # pragma: no cover - optional acceleration
    orjson = None

from intermine314.model import Attribute, Collection, Reference
from intermine314.service.errors import WebserviceError

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
        if rowformat.startswith("object"):
            rowformat = "jsonobjects"
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
        c = 0
        for _x in self:
            c += 1
        return c

    def __iter__(self):
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
    def __init__(self, connection, parser):
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
    LOG = logging.getLogger("JSONIterator")

    def __init__(self, connection, parser):
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
        return next_row


__all__ = [
    "_FALLBACK_GET_MAX_PAYLOAD_BYTES",
    "_JSON_ERROR_PREVIEW_MAX_CHARS",
    "_JSON_STATUS_BUFFER_MAX_CHARS",
    "EnrichmentLine",
    "FlatFileIterator",
    "JSONIterator",
    "ResultIterator",
    "ResultObject",
    "ResultRow",
    "TableResultRow",
    "decode_binary",
    "encode_dict",
    "encode_str",
]
