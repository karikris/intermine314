import logging
from collections import UserDict
from contextlib import closing
from urllib.parse import urlencode

from intermine314.service.errors import WebserviceError
from intermine314.service.resource_utils import close_resource_quietly as _close_resource_quietly
from intermine314.util.json import json_loads as _json_loads

_RESULTS_HEADER_SUFFIX = '"results":['
_STATUS_KEY = '"wasSuccessful"'
_FALLBACK_GET_MAX_PAYLOAD_BYTES = 4096
_JSON_STATUS_BUFFER_MAX_CHARS = 64 * 1024
_JSON_ERROR_PREVIEW_MAX_CHARS = 2048


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

    PARSED_FORMATS = frozenset(["dict"])
    STRING_FORMATS = frozenset(["count"])
    ROW_FORMATS = PARSED_FORMATS | STRING_FORMATS

    def __init__(self, service, path, params, rowformat, view, cld=None):
        if rowformat not in self.ROW_FORMATS:
            raise ValueError(
                "'%s' is not one of the valid row formats (%s)" % (rowformat, repr(list(self.ROW_FORMATS)))
            )

        if rowformat in self.PARSED_FORMATS:
            if service.version >= 8:
                params.update({"format": "json"})
            else:
                params.update({"format": "jsonrows"})
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

    def _row_as_dict(self, payload):
        values = self._extract_row_values(payload)
        if isinstance(values, (list, tuple)) and len(values) == len(self.view):
            return dict(zip(self.view, values))
        if isinstance(values, dict):
            if all(column in values for column in self.view):
                return {column: values.get(column) for column in self.view}
            return dict(values)
        raise ValueError(f"Unexpected row payload type for dict mode: {type(values).__name__}")

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
        try:
            if self.rowformat == "count":
                inner = FlatFileIterator(con, identity)
            elif self.rowformat == "dict":
                inner = JSONIterator(con, self._row_as_dict)
            else:
                raise ValueError("Couldn't get iterator for " + self.rowformat)
        except Exception:
            _close_resource_quietly(con)
            raise

        def _iter_rows():
            with closing(con):
                for item in inner:
                    yield item

        return _iter_rows()

    def __next__(self):
        if self._it is None:
            self._it = iter(self)
        try:
            return next(self._it)
        except StopIteration:
            self._it = None
            raise
        except Exception:
            self._it = None
            raise

    def close(self):
        iterator = self._it
        self._it = None
        if iterator is None:
            return
        close_fn = getattr(iterator, "close", None)
        if callable(close_fn):
            close_fn()

    def __del__(self):  # pragma: no cover - non-deterministic GC timing
        try:
            self.close()
        except Exception:
            return

    def next(self):
        return self.__next__()


class FlatFileIterator(object):
    def __init__(self, connection, parser):
        self.connection = connection
        self.parser = parser

    def __iter__(self):
        return self

    def __next__(self):
        try:
            line = decode_binary(next(self.connection)).strip()
        except StopIteration:
            _close_resource_quietly(self.connection)
            raise
        except Exception:
            _close_resource_quietly(self.connection)
            raise
        if line.startswith("[ERROR]"):
            _close_resource_quietly(self.connection)
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
            _close_resource_quietly(self.connection)
            raise StopIteration
        try:
            return self.get_next_row_from_connection()
        except StopIteration:
            _close_resource_quietly(self.connection)
            raise
        except Exception:
            _close_resource_quietly(self.connection)
            raise

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
    "decode_binary",
    "encode_dict",
    "encode_str",
]
