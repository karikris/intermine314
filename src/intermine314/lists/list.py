"""Server list metadata and object access, adapted from InterMine 1.13.0."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode

from intermine314.service.resource_utils import close_resource_quietly

__all__ = ["List"]


class List:
    """A stored server collection; obtain instances through a ListManager."""

    def __init__(self, **args):
        try:
            self._service = args["service"]
            self._manager = weakref.proxy(args["manager"])
            self._name = args["name"]
            self._title = args["title"]
            self._list_type = args["type"]
            self._size = int(args["size"])
        except KeyError as exc:
            raise ValueError("Missing argument") from exc
        self._description = args.get("description")
        self._date_created = args.get("dateCreated")
        self._is_authorized = args.get("authorized")
        if self._is_authorized is None:
            self._is_authorized = True
        self._status = args.get("status")
        self._tags = frozenset(args.get("tags", []))
        self.unmatched_identifiers = set()

    @property
    def date_created(self):
        return self._date_created

    @property
    def tags(self):
        return self._tags

    @property
    def description(self):
        return self._description

    @property
    def title(self):
        return self._title

    @property
    def status(self):
        return self._status

    @property
    def is_authorized(self):
        return self._is_authorized

    @property
    def list_type(self):
        return self._list_type

    def get_name(self):
        return self._name

    def set_name(self, new_name):
        """Rename on the server and retain an explicitly named temporary list."""
        if self._name == new_name:
            return
        old_name = self._name
        uri = self._service.root + self._service.LIST_RENAME_PATH + "?" + urlencode({
            "oldname": old_name, "newname": new_name,
        })
        with closing(self._service.opener.open(uri)) as response:
            body = response.read()
        renamed = self._manager.parse_list_upload_response(body)
        self._name = renamed.name
        self._size = renamed.size
        self.unmatched_identifiers.update(renamed.unmatched_identifiers)
        self._manager.lists.pop(old_name, None)
        self._manager.lists[self._name] = self
        self._manager._temp_lists.discard(old_name)

    def del_name(self):
        raise AttributeError("List names cannot be deleted, only changed")

    name = property(get_name, set_name, del_name, "The name of this list")

    @property
    def size(self):
        return self._size

    @property
    def count(self):
        return self.size

    def __len__(self):
        return self.size

    def _add_failed_matches(self, ids):
        if ids is not None:
            self.unmatched_identifiers.update(ids)

    def __str__(self):
        value = f"{self.name} ({self.size} {self.list_type})"
        if self.date_created:
            value += " " + self.date_created
        if self.description:
            value += " " + self.description
        return value

    def delete(self):
        self._manager.delete_lists([self])

    def append(self, appendix=None, *, csv_input=None, csv_column=None, csv_options=None):
        """Append identifiers, optionally from an explicit CSV String column.

        Queryables use query endpoints; collections of queryables are unioned
        first. Dispatch never retries failures. Borrowed streams remain open.
        """
        if csv_input is None:
            if csv_column is not None or csv_options is not None:
                raise ValueError("csv_column and csv_options require csv_input")
            if self._manager._is_queryable(appendix):
                return self._finish_append(self._manager._append_queryable(appendix, self.name))
            if not isinstance(appendix, (str, bytes, Path)) and not callable(getattr(appendix, "read", None)):
                try:
                    appendix = list(appendix)
                except TypeError as exc:
                    raise TypeError("Cannot append the supplied content") from exc
                queryables = [self._manager._is_queryable(item) for item in appendix]
                if any(queryables):
                    if not all(queryables):
                        raise TypeError("Cannot mix queryables and identifiers in an appendix")
                    union = self._manager.union(appendix)
                    return self._finish_append(self._manager._append_queryable(union, self.name))
            identifiers = self._manager._identifier_text(appendix, preserve_raw=True)
        else:
            if appendix is not None:
                raise ValueError("CSV input cannot be combined with appendix")
            if not isinstance(csv_column, str) or not csv_column:
                raise ValueError("CSV input requires an explicit csv_column")
            from intermine314.lists._csv import identifier_text

            identifiers = identifier_text(csv_input, csv_column, csv_options)
        uri = self._service.root + self._service.LIST_APPENDING_PATH + "?" + urlencode({"name": self.name})
        body = self._service.opener.post_plain_text(uri, identifiers)
        return self._finish_append(self._manager.parse_list_upload_response(body))

    def _finish_append(self, updated):
        self.unmatched_identifiers.update(updated.unmatched_identifiers)
        self._size = updated.size
        return self

    def to_query(self):
        """Return a factory query with explicit named-list membership."""
        return self._contents_query()

    def make_list_constraint(self, path, op):
        from intermine314.model import ConstraintNode

        return ConstraintNode(path, op, self.name)

    def __or__(self, other):
        return self._manager.union([self, other])

    def __add__(self, other):
        return self._manager.union([self, other])

    def __and__(self, other):
        return self._manager.intersect([self, other])

    def __xor__(self, other):
        return self._manager.xor([self, other])

    def __sub__(self, other):
        return self._manager.subtract([self], [other])

    def __iadd__(self, other):
        return self.append(other)

    def _replace_with_operation(self, operation, other):
        old_name = self.name
        temporary = old_name in self._manager._temp_lists
        args = ([self], [other]) if operation == "subtract" else ([self, other],)
        result = getattr(self._manager, operation)(*args, description=self.description, tags=self.tags)
        self.delete()
        result.name = old_name
        # Internal replacement is not an explicit user rename.
        if temporary:
            self._manager._temp_lists.add(old_name)
        return result

    def __iand__(self, other):
        return self._replace_with_operation("intersect", other)

    def __ixor__(self, other):
        return self._replace_with_operation("xor", other)

    def __isub__(self, other):
        return self._replace_with_operation("subtract", other)

    def add_tags(self, *tags):
        """Add tags on the server and store its returned immutable tag set."""
        self._tags = frozenset(self._manager.add_tags(self, tags))

    def remove_tags(self, *tags):
        """Remove tags on the server and store its returned immutable tag set."""
        self._tags = frozenset(self._manager.remove_tags(self, tags))

    def update_tags(self, *tags):
        """Refresh tags from the server; upstream ignores optional arguments."""
        self._tags = frozenset(self._manager.get_tags(self))

    def _contents_query(self):
        """Shared foundation for object access and public conversion."""
        from intermine314.query.constraints import ListConstraint

        query = self._service.new_query(self.list_type)
        # Explicitly select named membership, independent of native scalar IN.
        query.add_constraint(ListConstraint(self.list_type, "IN", self.name))
        return query

    def __iter__(self):
        """Use profile defaults: native dictionaries and legacy objects."""
        return iter(self._contents_query())

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise IndexError(f"Expected an integer key - got {index}")
        offset = self.size + index if index < 0 else index
        if offset < 0 or offset >= self.size:
            raise IndexError(f"{index} is not a valid index for a list of size {self.size}")
        return self._contents_query().first(start=offset, row="jsonobjects")

    def display(self):
        """Print actual selected fields without parsing a row's string form."""
        from intermine314._result_object import ResultObject

        stream = iter(self)
        try:
            for number, row in enumerate(stream, 1):
                print(f"Row {number}:")
                if isinstance(row, Mapping):
                    fields = row.items()
                elif isinstance(row, ResultObject):
                    fields = row._data.items()
                elif callable(getattr(row, "items", None)):
                    fields = row.items()
                else:
                    fields = [("value", row)]
                for name, value in fields:
                    if name not in ("class", "objectId"):
                        print(f"{name.removeprefix(self.list_type + '.')} = {value}")
                print()
        finally:
            close_resource_quietly(stream)
