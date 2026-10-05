"""Managed server list CRUD, adapted from InterMine 1.13.0."""

from __future__ import annotations

import json
import weakref
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode

from intermine314.compatibility import class_name
from intermine314.lists._identifiers import quote_identifier
from intermine314.lists.list import List
from intermine314.service.errors import WebserviceError
from intermine314.service.transport import open_readonly

__all__ = ["ListManager", "ListServiceError", "safe_key"]


def safe_key(maybe_unicode):
    """Python 3 dictionary keys already support Unicode."""
    return maybe_unicode


class ListServiceError(WebserviceError):
    """A list endpoint returned a malformed or unsuccessful response."""


class ListManager:
    """Lazy list discovery and management using the service's configured opener.

    Anonymous name allocation is local to this manager and is not thread safe.
    Context cleanup deletes only unnamed temporary lists owned by this manager.
    """

    INTERSECTION_PATH = "/lists/intersect/json"
    UNION_PATH = "/lists/union/json"
    DIFFERENCE_PATH = "/lists/diff/json"
    SUBTRACTION_PATH = "/lists/subtract/json"

    DEFAULT_LIST_NAME = "my_list"
    DEFAULT_DESCRIPTION = "List created with Python client library"

    def __init__(self, service):
        self.service = weakref.proxy(service)
        self.lists = None
        self._temp_lists = set()

    @staticmethod
    def safe_dict(d):
        return {safe_key(key): value for key, value in d.items()} if isinstance(d, dict) else d

    @staticmethod
    def _body_to_json(body):
        try:
            data = json.loads(body)
        except (ValueError, TypeError, UnicodeError) as exc:
            # Do not expose identifier payloads in parsing errors or logs.
            raise ListServiceError("Error parsing response") from exc
        if not isinstance(data, dict):
            raise ListServiceError("Error parsing response: expected an object")
        if not data.get("wasSuccessful"):
            raise ListServiceError(data.get("error"))
        return data

    def refresh_lists(self):
        uri = self.service.root + self.service.LIST_PATH
        with closing(open_readonly(self.service.opener, uri)) as response:
            data = self._body_to_json(response.read())
        try:
            lists = {info["name"]: List(service=self.service, manager=self, **self.safe_dict(info))
                     for info in data["lists"]}
        except (KeyError, TypeError, ValueError) as exc:
            raise ListServiceError("Error parsing response: invalid list metadata") from exc
        # Publish only a complete successful refresh.
        self.lists = lists

    def get_list(self, name):
        if self.lists is None:
            self.refresh_lists()
        return self.lists.get(name)

    def l(self, name):  # noqa: E743 - original public alias
        return self.get_list(name)

    def get_all_lists(self):
        if self.lists is None:
            self.refresh_lists()
        return self.lists.values()

    def get_all_list_names(self):
        if self.lists is None:
            self.refresh_lists()
        return self.lists.keys()

    def get_list_count(self):
        return len(self.get_all_list_names())

    def get_unused_list_name(self):
        self.refresh_lists()
        counter = 1
        while True:
            name = f"{self.DEFAULT_LIST_NAME}_{counter}"
            if name not in self.lists and name not in self._temp_lists:
                self._temp_lists.add(name)
                return name
            counter += 1

    @staticmethod
    def _is_queryable(content):
        return isinstance(content, List) or callable(getattr(content, "to_query", None)) or hasattr(content, "model")

    @staticmethod
    def _identifier_text(content, *, preserve_raw=False):
        if ListManager._is_queryable(content):
            raise TypeError("Queryable content requires the query upload endpoint")
        if callable(getattr(content, "read", None)):
            identifiers = content.read()
        elif isinstance(content, Path):
            identifiers = content.read_text(encoding="utf-8")
        elif isinstance(content, str):
            try:
                identifiers = Path(content).read_text(encoding="utf-8")
            except (OSError, ValueError):
                identifiers = content if preserve_raw else content.strip()
        else:
            try:
                tokens = []
                for value in iter(content):
                    if ListManager._is_queryable(value):
                        raise TypeError("Queryable content requires the query upload endpoint")
                    tokens.append(quote_identifier(value))
                identifiers = "\n".join(tokens)
            except TypeError as exc:
                raise TypeError("Cannot create list from the supplied content") from exc
        if isinstance(identifiers, bytes):
            identifiers = identifiers.decode("utf-8")
        if not isinstance(identifiers, str):
            raise TypeError("Identifier read() must return text or UTF-8 bytes")
        return identifiers

    def create_list(self, content=None, list_type='', name=None, description=None,
                    tags=(), add=(), organism=None, *, csv_input=None,
                    csv_column=None, csv_options=None):
        """Upload identifiers as plain text, optionally reading a CSV column.

        CSV requires an explicit class and column. Identifiers are read as
        Polars Strings through temporary Parquet and a DuckDB Arrow reader.
        Null/empty identifiers and CR/LF within an identifier raise ValueError
        before any upload. Borrowed streams remain open, including on failure.
        """
        if csv_input is None:
            if csv_column is not None or csv_options is not None:
                raise ValueError("csv_column and csv_options require csv_input")
            if organism:
                query = self.service.new_query(list_type)
                if isinstance(organism, list):
                    query.add_constraint(f"{list_type}.organism.name", "ONE OF", organism)
                else:
                    query.add_constraint("organism", "LOOKUP", organism)
                if isinstance(content, list):
                    query.add_constraint("symbol", "ONE OF", content)
                content = query
            if self._is_queryable(content):
                return self._create_list_from_queryable(content, name, description, tags)
            identifiers = self._identifier_text(content)
        else:
            if content is not None or organism is not None:
                raise ValueError("CSV input cannot be combined with content or organism filters")
            if not isinstance(csv_column, str) or not csv_column:
                raise ValueError("CSV input requires an explicit csv_column")
            if not list_type:
                raise ValueError("CSV input requires an explicit list_type")
            from intermine314.lists._csv import identifier_text

            identifiers = identifier_text(csv_input, csv_column, csv_options)
        if not identifiers:
            print("Lists must have one or more elements - the current list has 0")
            print("Please create a valid list with at least one element and create the list again.")
            return None
        if name is None:
            name = self.get_unused_list_name()
        if description is None:
            description = self.DEFAULT_DESCRIPTION
        params = {"name": name, "type": list_type, "description": description, "tags": ";".join(tags)}
        additions = [value.lower() for value in add if value]
        if additions:
            params["add"] = additions
        uri = self.service.root + self.service.LIST_CREATION_PATH + "?" + urlencode(params, doseq=True)
        body = self.service.opener.post_plain_text(uri, identifiers)
        return self.parse_list_upload_response(body)

    def _validate_service_root(self, source_service):
        if source_service is None or source_service.root.rstrip("/") != self.service.root.rstrip("/"):
            raise ValueError("List and query operands must be bound to a Service with the same root as this list manager")

    def _get_listable_query(self, queryable):
        """Project an independent query while retaining profile and filters.

        Different mine roots and unbound queries cannot be uploaded through
        this manager. Same-root clients use this manager's configured opener.
        """
        query = queryable.to_query()
        self._validate_service_root(getattr(query, "service", None))
        query = query.clone()
        if not query.views:
            root = class_name(query.root)
            if not root:
                raise ValueError("List upload requires a root; select one entity path")
            query.add_view(root + ".id")
        else:
            parents = {view.rsplit(".", 1)[0] for view in query.views}
            if len(parents) != 1:
                raise ValueError("List upload has ambiguous views; select one entity path")
            query = query.select(parents.pop() + ".id")
        # Endpoints and authentication always come from the target manager.
        query.service = self.service
        query._list_upload_params()  # Validate server capability before any operand upload.
        return query

    def _create_list_from_queryable(self, queryable, name, description, tags):
        query = self._get_listable_query(queryable)
        if name is None:
            name = self.get_unused_list_name()
        if description is None:
            description = self.DEFAULT_DESCRIPTION
        params = query._list_upload_params()
        params.update(listName=name, description=description, tags=";".join(tags))
        with closing(self.service.opener.open(query.get_list_upload_uri(), urlencode(params, doseq=True))) as response:
            body = response.read()
        return self.parse_list_upload_response(body)

    def _append_queryable(self, queryable, name):
        query = self._get_listable_query(queryable)
        params = query._list_upload_params(append=True)
        params.update(listName=name)
        with closing(self.service.opener.open(query.get_list_append_uri(), urlencode(params, doseq=True))) as response:
            body = response.read()
        return self.parse_list_upload_response(body)

    def _validate_operation_inputs(self, lists):
        """Validate every operand before query uploads can change server state."""
        validated = []
        for item in lists:
            if isinstance(item, List):
                self._validate_service_root(item._service)
                item = item.name
            elif self._is_queryable(item):
                item = self._get_listable_query(item)
            else:
                # Raw names always refer to this manager's server.
                item = str(item)
            if isinstance(item, str) and ";" in item:
                raise ValueError("List names containing a semicolon cannot be used in set operations; rename the list first")
            validated.append(item)
        return validated

    def _make_validated_list_names(self, lists):
        return [item if isinstance(item, str) else self.create_list(item).name for item in lists]

    def make_list_names(self, lists):
        """Validate all operands, then upload queries without iterating rows."""
        return self._make_validated_list_names(self._validate_operation_inputs(lists))

    def intersect(self, lists, name=None, description=None, tags=()):
        return self._do_operation(self.INTERSECTION_PATH, "Intersection", lists, name, description, tags)

    def union(self, lists, name=None, description=None, tags=()):
        return self._do_operation(self.UNION_PATH, "Union", lists, name, description, tags)

    def xor(self, lists, name=None, description=None, tags=()):
        return self._do_operation(self.DIFFERENCE_PATH, "Difference", lists, name, description, tags)

    def _do_operation(self, path, operation, lists, name, description, tags):
        names = self.make_list_names(lists)
        if description is None:
            description = operation + " of " + " and ".join(names)
        return self._operation_request(path, name, description, tags, lists=";".join(names))

    def subtract(self, lefts, rights, name=None, description=None, tags=()):
        # Both sides must be valid before either side uploads a query.
        lefts = self._validate_operation_inputs(lefts)
        rights = self._validate_operation_inputs(rights)
        left_names = self._make_validated_list_names(lefts)
        right_names = self._make_validated_list_names(rights)
        if description is None:
            description = "Subtraction of " + " and ".join(right_names) + " from " + " and ".join(left_names)
        return self._operation_request(self.SUBTRACTION_PATH, name, description, tags,
                                       references=";".join(left_names), subtract=";".join(right_names))

    def _operation_request(self, path, name, description, tags, **params):
        if name is None:
            name = self.get_unused_list_name()
        params.update(name=name, description=description, tags=";".join(tags))
        uri = self.service.root + path + "?" + urlencode(params)
        with closing(self.service.opener.open(uri)) as response:
            body = response.read()
        return self.parse_list_upload_response(body)

    def parse_list_upload_response(self, response):
        data = self._body_to_json(response)
        try:
            name = data["listName"]
        except KeyError as exc:
            raise ListServiceError("Error parsing response: missing listName") from exc
        self.refresh_lists()
        item = self.get_list(name)
        if item is None:
            raise ListServiceError("Uploaded list is absent from refreshed list metadata")
        item._add_failed_matches(data.get("unmatchedIdentifiers"))
        return item

    def delete_lists(self, lists):
        items = tuple(lists)
        self.refresh_lists()
        for item in items:
            name = item.name if isinstance(item, List) else str(item)
            if name not in self.lists:
                continue
            uri = self.service.root + self.service.LIST_PATH + "?" + urlencode({"name": name})
            self._body_to_json(self.service.opener.delete(uri))
            self._temp_lists.discard(name)
        self.refresh_lists()

    def add_tags(self, to_tag, tags):
        """Add semicolon-separated tags and return the server's current tags."""
        uri = self.service.root + self.service.LIST_TAG_PATH
        form = urlencode({"name": to_tag.name, "tags": ";".join(tags)})
        with closing(self.service.opener.open(uri, form)) as response:
            return self._body_to_json(response.read())["tags"]

    def remove_tags(self, to_remove_from, tags):
        """Remove semicolon-separated tags and return the server's current tags."""
        uri = self.service.root + self.service.LIST_TAG_PATH + "?" + urlencode({
            "name": to_remove_from.name, "tags": ";".join(tags),
        })
        return self._body_to_json(self.service.opener.delete(uri))["tags"]

    def get_tags(self, im_list):
        """Fetch current tags without mutating the supplied List."""
        uri = self.service.root + self.service.LIST_TAG_PATH + "?" + urlencode({"name": im_list.name})
        with closing(self.service.opener.open(uri)) as response:
            return self._body_to_json(response.read())["tags"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        try:
            self.delete_temporary_lists()
        except Exception as cleanup_error:
            if exc_val is None:
                raise
            exc_val.add_note(f"Temporary list cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}")
        # Cleanup interrupts propagate; ordinary cleanup failures retain the
        # body's exception (including an interrupt) with an observable note.

    def delete_temporary_lists(self):
        """Delete this manager's unnamed lists, retaining failed names for retry."""
        names = tuple(self._temp_lists)
        if names:
            self.delete_lists(names)
            # delete_lists retires confirmed deletions individually. Missing
            # names can be retired only after the entire operation succeeds.
            self._temp_lists.difference_update(names)
