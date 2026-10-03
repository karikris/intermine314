"""Model-backed object results, adapted from InterMine 1.13.0 (BSD-2-Clause)."""
from reprlib import recursive_repr

from intermine314.model import (
    Attribute,
    Class,
    Collection,
    ComposedClass,
    ModelError,
    Reference,
)


class ResultObject:
    """A returned object with cached fields and lazy access through its model service.

    Loaded/selected nulls never trigger another request. Missing identifiers make
    unselected fields unavailable. Nested objects share an identity map keyed by
    the input mapping, so even cyclic in-memory payloads have bounded reprs.
    """

    def __init__(self, data, cld, view=()):
        self._initialize(data, cld, view, {})

    def _initialize(self, data, cld, view, objects):
        if not isinstance(cld, Class):
            raise ModelError("Object results require a valid model Class descriptor")
        if not isinstance(data, dict):
            raise TypeError("Object results require a JSON object payload")
        self._data = data
        class_name = data.get("class")
        self._cld = cld if not class_name or cld.name == class_name else cld.model.get_class(class_name)
        self.selected_attributes = []
        self.reference_paths = {}
        for path in view:
            stripped = str(path).split(".", 1)[-1]
            if "." in stripped:
                prefix = stripped.split(".", 1)[0] + "."
                self.reference_paths.setdefault(prefix, []).append(stripped)
            else:
                self.selected_attributes.append(stripped)
        self._attr_cache = {}
        self._service = cld.model.service
        self._objects = objects
        objects[id(data)] = self

    @property
    def id(self):
        """The internal database identifier, or None when it was not returned."""
        return self._data.get("objectId")

    @property
    def type(self):
        return self._data.get("class") or self._cld.name

    def __str__(self):
        values = (f"{key} = {value!r}" for key, value in self._data.items()
                  if key not in {"objectId", "class"} and not isinstance(value, (dict, list)))
        return f"{self._cld.name}({',  '.join(values)})"

    @recursive_repr()
    def __repr__(self):
        # Only already returned fields are inspected: repr must never fetch.
        values = (f"{key} = {getattr(self, key)!r}" for key in self._data
                  if key not in {"objectId", "class"})
        return f"{self._cld.name}({', '.join(values)})"

    def _get_ref_paths(self, field):
        return self.reference_paths.get(field.name + ".", [])

    def _nested(self, data, field):
        if not isinstance(data, dict):
            raise TypeError("Object references require a JSON object payload")
        cached = self._objects.get(id(data))
        if cached is not None:
            return cached
        obj = object.__new__(ResultObject)
        obj._initialize(data, field.type_class, self._get_ref_paths(field), self._objects)
        obj._service = self._service
        return obj

    def __getattr__(self, name):
        cache = object.__getattribute__(self, "_attr_cache")
        if name in cache:
            return cache[name]
        field = object.__getattribute__(self, "_cld").get_field(name)
        if name in self._data:
            data = self._data[name]
            if isinstance(field, Collection):
                value = [] if data is None else [self._nested(item, field) for item in data]
            elif isinstance(field, Reference):
                value = None if data is None else self._nested(data, field)
            elif isinstance(field, Attribute):
                value = data
            else:
                raise ModelError(f"Unsupported field type for {name}")
        else:
            value = self._fetch(field)
        self._attr_cache[name] = value
        return value

    def _fetch(self, field):
        empty = [] if isinstance(field, Collection) else None
        if (field.name in self.selected_attributes or field.name + "." in self.reference_paths
                or self.id is None or "id" not in self._cld):
            return empty
        if self._service is None:
            return empty
        from intermine314.query import Query

        # Retain the exact query model even when it differs from Service.model.
        root = field.declared_in if isinstance(self._cld, ComposedClass) else self._cld
        query = Query(self._cld.model, service=self._service, root=root)
        path = root.name + "." + field.name
        query.add_view(path + ".*" if isinstance(field, Reference) else path)
        if isinstance(field, Reference):
            query.outerjoin(path)
        result = query.where(id=self.id)._first_object()
        if result is None or field.name not in result._data:
            return empty
        return getattr(result, field.name)
