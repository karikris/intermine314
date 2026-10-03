"""InterMine model descriptors, validated paths and column expression primitives.

Adapted from intermine-ws-python 1.13.0 (d888b779), copyright InterMine and
University of Cambridge, under BSD-2-Clause. See LICENSE-BSD and NOTICE.
Column query execution uses the service protocol; client integration is separate.
"""

import logging
import re
import weakref
from functools import reduce
from xml.dom import minidom

from intermine314.util import ReadableException, openAnything

__all__ = [
    "Model",
    "Field",
    "Attribute",
    "Reference",
    "Collection",
    "Class",
    "ComposedClass",
    "Path",
    "Column",
    "ConstraintTree",
    "ConstraintNode",
    "CodelessNode",
    "ModelError",
    "PathParseError",
    "ModelParseError",
]
__author__ = "Alex Kalderimis"
__organization__ = "InterMine"
__license__ = "BSD-2-Clause"
__contact__ = "dev@intermine.org"


class Field:
    def __init__(self, name, type_name, class_origin):
        self.name = name
        self.type_name = type_name
        self.type_class = None
        self.declared_in = class_origin

    def __repr__(self):
        return self.name + " is a " + self.type_name

    def __str__(self):
        return self.name

    @property
    def fieldtype(self):
        raise Exception("Fields should never be directly instantiated")


class Attribute(Field):
    @property
    def fieldtype(self):
        return "attribute"


class Reference(Field):
    def __init__(self, name, type_name, class_origin, reverse_ref=None):
        self.reverse_reference_name = reverse_ref
        super().__init__(name, type_name, class_origin)
        self.reverse_reference = None

    def __repr__(self):
        s = super().__repr__()
        if self.reverse_reference is None:
            return s
        else:
            return s + ", which links back to this as " + self.reverse_reference.name

    @property
    def fieldtype(self):
        return "reference"


class Collection(Reference):
    def __repr__(self):
        ret = super().__repr__().replace(" is a ", " is a group of ")
        if self.reverse_reference is None:
            return ret + " objects"
        else:
            return ret.replace(", which links", " objects, which link")

    @property
    def fieldtype(self):
        return "collection"


class Class:
    def __init__(self, name, parents, model, interface=True):
        self.name = name
        self.parents = parents
        self.model = model
        self.parent_classes = []
        self.is_interface = interface
        self.field_dict = {}
        self.has_id = "Object" not in parents
        if self.has_id:
            id_field = Attribute("id", "Integer", self)
            self.field_dict["id"] = id_field

    def __repr__(self):
        return "<{}.{} {}.{}>".format(
            self.__module__,
            self.__class__.__name__,
            self.model.package_name
            if hasattr(self.model, "package_name")
            else "__test__",
            self.name,
        )

    @property
    def fields(self):
        return sorted(list(self.field_dict.values()), key=lambda field: field.name)

    def __iter__(self):
        yield from list(self.field_dict.values())

    def __contains__(self, item):
        if isinstance(item, Field):
            return item in list(self.field_dict.values())
        else:
            return str(item) in self.field_dict

    @property
    def attributes(self):
        return [x for x in self.fields if isinstance(x, Attribute)]

    @property
    def references(self):

        def isRef(x):
            return isinstance(x, Reference) and (not isinstance(x, Collection))

        return list(filter(isRef, self.fields))

    @property
    def collections(self):
        return [x for x in self.fields if isinstance(x, Collection)]

    def get_field(self, name):
        if name in self.field_dict:
            return self.field_dict[name]
        else:
            raise ModelError(f"There is no field called {name} in {self.name}")

    def isa(self, other):
        if isinstance(other, Class):
            other_name = other.name
        else:
            other_name = other
        if self.name == other_name:
            return True
        if other_name in self.parents:
            return True
        for p in self.parent_classes:
            if p.isa(other):
                return True
        return False


class ComposedClass(Class):
    def __init__(self, parts, model):
        self.is_interface = True
        self.parts = parts
        self.model = weakref.proxy(model)

    @property
    def parents(self):
        return reduce(lambda ps, cls: ps + cls.parents, self.parts, [])

    @property
    def name(self):
        return "_".join(c.name for c in self.parts)

    @property
    def has_id(self):
        return "Object" not in self.parents

    @property
    def field_dict(self):
        fields = {}
        if self.has_id:
            fields["id"] = Attribute("id", "Integer", self)
        for p in self.parts:
            fields.update(p.field_dict)
        return fields

    @property
    def parent_classes(self):
        all_parents = []
        for p in self.parts:
            all_parents.extend(pc for pc in p.parent_classes if pc not in all_parents)
        return all_parents + self.parts


class Path:
    def __init__(self, path, model, subclasses=None):
        self.model = weakref.proxy(model)
        self.subclasses = {} if subclasses is None else subclasses
        if isinstance(path, Class):
            self._string = path.name
            self.parts = [path]
        else:
            self._string = str(path)
            self.parts = model.parse_path_string(str(path), self.subclasses)

    def __str__(self):
        return self._string

    def __repr__(self):
        return (
            "<"
            + self.__module__
            + "."
            + self.__class__.__name__
            + ": "
            + self._string
            + ">"
        )

    def prefix(self):
        parts = list(self.parts)
        parts.pop()
        if len(parts) < 1:
            raise PathParseError(str(self) + " does not have a prefix")
        s = ".".join([x.name for x in parts])
        return Path(s, self.model._unproxied(), self.subclasses)

    def append(self, *elements):
        s = str(self) + "." + ".".join(elements)
        return Path(s, self.model._unproxied(), self.subclasses)

    @property
    def root(self):
        return self.parts[0]

    @property
    def end(self):
        return self.parts[-1]

    def get_class(self):
        if self.is_class():
            return self.end
        elif self.is_reference():
            if str(self) in self.subclasses:
                return self.model.get_class(self.subclasses[str(self)])
            return self.end.type_class
        else:
            return None

    end_class = property(get_class)

    def is_reference(self):
        return isinstance(self.end, Reference)

    def is_class(self):
        return isinstance(self.end, Class)

    def is_attribute(self):
        return isinstance(self.end, Attribute)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        # Equality is deliberately path-string based, even with subclass overrides.
        return hash(str(self))


def _logic_codes(start):
    """Generate A through Z, AA through AZ, and so on from a starting code."""
    if (
        not isinstance(start, str)
        or not start
        or any(c < "A" or c > "Z" for c in start)
    ):
        raise ValueError("Constraint codes must be uppercase alphabetic strings")
    number = 0
    for char in start:
        number = number * 26 + ord(char) - ord("A") + 1
    while True:
        remainder = number
        letters = []
        while remainder:
            remainder, digit = divmod(remainder - 1, 26)
            letters.append(chr(ord("A") + digit))
        yield "".join(reversed(letters))
        number += 1


class ConstraintTree:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __and__(self, other):
        return ConstraintTree("AND", self, other)

    def __or__(self, other):
        return ConstraintTree("OR", self, other)

    def __iter__(self):
        for n in [self.left, self.right]:
            yield from n

    def as_logic(self, codes=None, start="A"):
        if codes is None:
            codes = _logic_codes(start)
        left, right = self.left.as_logic(codes), self.right.as_logic(codes)
        if not left or not right:
            return left or right
        return f"({left} {self.op} {right})"


class ConstraintNode(ConstraintTree):
    def __init__(self, *args, **kwargs):
        self.vargs = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self

    def as_logic(self, codes=None, start="A"):
        if codes is None:
            codes = _logic_codes(start)
        return next(codes)


class CodelessNode(ConstraintNode):
    def as_logic(self, code=None, start="A"):
        return ""


class Column:
    def __init__(self, path, model, subclasses=None, query=None, parent=None):
        self._model = model
        self._query = query
        self._subclasses = (
            (path.subclasses if isinstance(path, Path) else {})
            if subclasses is None
            else subclasses
        )
        self._parent = parent
        if isinstance(path, Path):
            self._path = path
        else:
            self._path = model.make_path(path, self._subclasses)
        self._branches = {}

    def select(self, *cols):
        q = self._model.service.new_query(str(self))
        if len(cols):
            q.select(*cols)
        else:
            q.select(self)
        return q

    def where(self, *args, **kwargs):
        q = self.select()
        return q.where(*args, **kwargs)

    filter = where

    def __len__(self):
        return self.select().count()

    def __iter__(self):
        q = self.select()
        if self._path.is_attribute():
            for row in q.rows():
                yield row[0]
        else:
            yield from q

    def __getattr__(self, name):
        if name in self._branches:
            return self._branches[name]
        cld = (
            self._model.get_class(self._subclasses[str(self)])
            if str(self) in self._subclasses
            else self._path.get_class()
        )
        if cld is not None:
            try:
                cld.get_field(name)
                branch = Column(
                    str(self) + "." + name,
                    self._model,
                    self._subclasses,
                    self._query,
                    self,
                )
                self._branches[name] = branch
                return branch
            except ModelError as e:
                raise AttributeError(str(e)) from e
        raise AttributeError("No attribute '" + name + "'")

    def __str__(self):
        return str(self._path)

    def __mod__(self, other):
        if isinstance(other, tuple):
            return ConstraintNode(str(self), "LOOKUP", *other)
        else:
            return ConstraintNode(str(self), "LOOKUP", str(other))

    def __rshift__(self, other):
        return CodelessNode(str(self), str(other))

    __lshift__ = __rshift__

    def __eq__(self, other):
        if other is None:
            return ConstraintNode(str(self), "IS NULL")
        elif isinstance(other, Column):
            return ConstraintNode(str(self), "IS", str(other))
        elif hasattr(other, "make_list_constraint"):
            return other.make_list_constraint(str(self), "IN")
        elif isinstance(other, list):
            return ConstraintNode(str(self), "ONE OF", other)
        else:
            return ConstraintNode(str(self), "=", other)

    def __ne__(self, other):
        if other is None:
            return ConstraintNode(str(self), "IS NOT NULL")
        elif isinstance(other, Column):
            return ConstraintNode(str(self), "IS NOT", str(other))
        elif hasattr(other, "make_list_constraint"):
            return other.make_list_constraint(str(self), "NOT IN")
        elif isinstance(other, list):
            return ConstraintNode(str(self), "NONE OF", other)
        else:
            return ConstraintNode(str(self), "!=", other)

    def __xor__(self, other):
        if hasattr(other, "make_list_constraint"):
            return other.make_list_constraint(str(self), "NOT IN")
        elif isinstance(other, list):
            return ConstraintNode(str(self), "NONE OF", other)
        raise TypeError(f"Invalid argument for xor: {other!r}")

    def in_(self, other):
        if hasattr(other, "make_list_constraint"):
            return other.make_list_constraint(str(self), "IN")
        elif isinstance(other, list):
            return ConstraintNode(str(self), "ONE OF", other)
        raise TypeError(f"Invalid argument for in_: {other!r}")

    def __lt__(self, other):
        if isinstance(other, Column):
            self._subclasses[str(self)] = str(other)
            self._branches = {}
            if self._parent is not None:
                self._parent._branches = {}
            return CodelessNode(str(self), str(other))
        try:
            return self.in_(other)
        except TypeError:
            return ConstraintNode(str(self), "<", other)

    def __le__(self, other):
        if isinstance(other, Column):
            return CodelessNode(str(self), str(other))
        try:
            return self.in_(other)
        except TypeError:
            return ConstraintNode(str(self), "<=", other)

    def __gt__(self, other):
        return ConstraintNode(str(self), ">", other)

    def __ge__(self, other):
        return ConstraintNode(str(self), ">=", other)


class Model:
    NUMERIC_TYPES = frozenset(
        [
            "int",
            "Integer",
            "float",
            "Float",
            "double",
            "Double",
            "long",
            "Long",
            "short",
            "Short",
        ]
    )
    LOG = logging.getLogger("Model")

    def __init__(self, source, service=None):
        assert source is not None
        self.source = source
        if service is not None:
            self.service = weakref.proxy(service)
        else:
            self.service = None
        self.classes = {}
        self.parse_model(source)
        self.vivify()

    def parse_model(self, source):
        io = src = doc = None
        owned = not hasattr(source, "read")
        try:
            io = openAnything(source)
            src = io.read()
            if hasattr(src, "decode"):
                src = src.decode("utf8")
            self.LOG.debug(f"model = [{src}]")
            doc = minidom.parseString(src)
            nodes = doc.getElementsByTagName("model")
            if len(nodes) != 1:
                raise ValueError(
                    "More than one model element" if nodes else "No model element"
                )
            node = nodes[0]
            self.name = node.getAttribute("name")
            self.package_name = node.getAttribute("package")
            if not self.name or not self.package_name:
                raise ValueError("No model name or package name")
            parsed_names = set()
            for c in doc.getElementsByTagName("class"):
                class_name = c.getAttribute("name")
                if not class_name:
                    raise ValueError("Name not defined in" + c.toxml())
                if class_name in parsed_names:
                    raise ValueError("Duplicate class " + class_name)
                parsed_names.add(class_name)

                def strip_java_prefix(x):
                    return re.sub(".*\\.", "", x)

                parents = [
                    strip_java_prefix(p)
                    for p in c.getAttribute("extends").split(" ")
                    if len(p)
                ]
                interface = c.getAttribute("is-interface") == "true"
                cl = Class(class_name, parents, self, interface)
                self.LOG.debug(f"Created {cl.name}")
                for a in c.getElementsByTagName("attribute"):
                    name = a.getAttribute("name")
                    type_name = strip_java_prefix(a.getAttribute("type"))
                    at = Attribute(name, type_name, cl)
                    cl.field_dict[name] = at
                    self.LOG.debug(f"set {cl.name}.{at.name}")
                for r in c.getElementsByTagName("reference"):
                    name = r.getAttribute("name")
                    type_name = r.getAttribute("referenced-type")
                    linked_field_name = r.getAttribute("reverse-reference")
                    ref = Reference(name, type_name, cl, linked_field_name)
                    cl.field_dict[name] = ref
                    self.LOG.debug(f"set {cl.name}.{ref.name}")
                for co in c.getElementsByTagName("collection"):
                    name = co.getAttribute("name")
                    type_name = co.getAttribute("referenced-type")
                    linked_field_name = co.getAttribute("reverse-reference")
                    col = Collection(name, type_name, cl, linked_field_name)
                    cl.field_dict[name] = col
                    self.LOG.debug(f"set {cl.name}.{col.name}")
                self.classes[class_name] = cl
        except Exception as error:
            model_src = src if src is not None else source
            raise ModelParseError("Error parsing model", model_src, error) from error
        finally:
            if doc is not None:
                doc.unlink()
            if io is not None and owned:
                io.close()

    def vivify(self):
        """Resolve ancestry before inheritance, then link all relationship fields."""
        for c in list(self.classes.values()):
            c.parent_classes = self.to_ancestry(c)
        complete = set()

        def inherit(c):
            if c.name in complete:
                return
            for pc in c.parent_classes:
                inherit(pc)
                for name, field in pc.field_dict.items():
                    existing = c.field_dict.get(name)
                    if existing is not None:
                        if existing.fieldtype != field.fieldtype:
                            raise ModelError(
                                f"Invalid inherited field {c.name}.{name}: incompatible field types"
                            )
                        if (
                            isinstance(field, Reference)
                            and existing.reverse_reference_name
                            != field.reverse_reference_name
                        ):
                            raise ModelError(
                                f"Invalid inherited field {c.name}.{name}: incompatible reverse references"
                            )
                        if isinstance(field, Attribute):
                            aliases = {
                                "Integer": "int",
                                "Boolean": "boolean",
                                "Float": "float",
                                "Double": "double",
                                "Long": "long",
                                "Short": "short",
                                "Byte": "byte",
                                "Character": "char",
                            }
                            if aliases.get(
                                existing.type_name, existing.type_name
                            ) != aliases.get(field.type_name, field.type_name):
                                raise ModelError(
                                    f"Invalid inherited field {c.name}.{name}: incompatible attribute types"
                                )
                        elif existing.type_name != field.type_name:
                            current_type = self.get_class(existing.type_name)
                            parent_type = self.get_class(field.type_name)
                            if current_type.isa(parent_type):
                                continue  # Preserve a legitimate narrowed reference.
                            if not parent_type.isa(current_type):
                                raise ModelError(
                                    f"Invalid inherited field {c.name}.{name}: incompatible reference types"
                                )
                    c.field_dict[name] = field
            complete.add(c.name)

        for c in self.classes.values():
            inherit(c)
        for c in self.classes.values():
            for f in c.fields:
                f.type_class = self.classes.get(f.type_name)
                if isinstance(f, Reference) and f.type_class is None:
                    raise ModelError(f"'{f.type_name}' is not a class in this model")
        for c in self.classes.values():
            for f in c.fields:
                if isinstance(f, Reference) and f.reverse_reference_name:
                    rrn = f.reverse_reference_name
                    reverse = f.type_class.get_field(rrn)
                    if not isinstance(reverse, Reference) or not (
                        f.declared_in.isa(reverse.type_class)
                        or reverse.type_class.isa(f.declared_in)
                    ):
                        raise ModelError(
                            f"Invalid reverse reference {f.type_class.name}.{rrn}"
                        )
                    f.reverse_reference = reverse

    def to_ancestry(self, cd):
        def visit(current, active):
            if current.name in active:
                raise ModelError(
                    "Inheritance cycle: " + " -> ".join((*active, current.name))
                )
            lineage = [
                self.classes[name] for name in current.parents if name in self.classes
            ]
            ancestors = list(lineage)
            for parent in lineage:
                ancestors.extend(visit(parent, (*active, current.name)))
            return ancestors

        # Unknown external Java ancestors (including Object) are ignored upstream.
        return visit(cd, ())

    def to_classes(self, classnames):
        return list(map(self.get_class, classnames))

    def column(self, path, *rest):
        return Column(path, self, *rest)

    table = column

    def __getattr__(self, name):
        return self.column(name)

    def get_class(self, name):
        if name.find(",") != -1:
            names = name.split(",")
            classes = [self.get_class(n) for n in names]
            return ComposedClass(classes, self)
        elif name.find(".") != -1:
            path = self.make_path(name)
            if path.is_attribute():
                raise ModelError("'" + str(path) + "' is not a class")
            else:
                return path.get_class()
        elif name in self.classes:
            return self.classes[name]
        else:
            raise ModelError("'" + name + "' is not a class in this model")

    def make_path(self, path, subclasses=None):
        return Path(path, self, subclasses)

    def validate_path(self, path_string, subclasses=None):
        try:
            self.parse_path_string(path_string, subclasses)
            return True
        except PathParseError as e:
            raise PathParseError(
                f"Error parsing '{path_string}' (subclasses: {str(subclasses)})",
                e,
            )

    def parse_path_string(self, path_string, subclasses=None):
        subclasses = {} if subclasses is None else subclasses
        descriptors = []
        names = path_string.split(".")
        root_name = names.pop(0)
        root_descriptor = self.get_class(root_name)
        descriptors.append(root_descriptor)
        if root_name in subclasses:
            current_class = self.get_class(subclasses[root_name])
        else:
            current_class = root_descriptor
        for field_name in names:
            if current_class is None:
                raise PathParseError(
                    "Cannot extend attribute path '{}'".format(
                        ".".join(x.name for x in descriptors)
                    )
                )
            field = current_class.get_field(field_name)
            descriptors.append(field)
            if isinstance(field, Reference):
                key = ".".join([x.name for x in descriptors])
                if key in subclasses:
                    current_class = self.get_class(subclasses[key])
                else:
                    current_class = field.type_class
            else:
                current_class = None
        return descriptors

    def _unproxied(self):
        return self


class ModelError(ReadableException):
    pass


class PathParseError(ModelError):
    pass


class ModelParseError(ModelError):
    def __init__(self, message, source, cause=None):
        self.source = source
        super().__init__(message, cause)

    def __str__(self):
        base = repr(self.message) + ":" + repr(self.source)
        if self.cause is None:
            return base
        else:
            return base + repr(self.cause)
