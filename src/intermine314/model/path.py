import weakref
from typing import Mapping, Optional

from .class_ import Class
from .errors import PathParseError
from .fields import Attribute, Reference
from .helpers import _get_extension_attr, _get_extensions_map, _set_slot_or_extension


class Path(object):
    """
    A class representing a validated dotted string path
    ===================================================

    A path represents a connection between records and fields

    SYNOPSIS
    --------

        >>> service = Service("https://www.flymine.org/query/service")
            model = service.model
            path = model.make_path("Gene.organism.name")
            path.is_attribute()
        ... True
        >>> path2 = model.make_path("Gene.proteins")
            path2.is_attribute()
        ... False
        >>> path2.is_reference()
        ... True
        >>> path2.get_class()
        ... <intermine314.model.Class: gene>

    OVERVIEW
    --------

    This class is used for performing validation on dotted path strings.
    The simple act of parsing it into existence will validate the path
    to some extent, but there are additional methods for verifying certain
    relationships as well
    """

    __slots__ = ("model", "subclasses", "_string", "parts", "_extensions")

    @classmethod
    def _from_validated_parts(cls, model, path_string: str, parts, subclasses):
        instance = object.__new__(cls)
        object.__setattr__(instance, "model", weakref.proxy(model))
        object.__setattr__(instance, "subclasses", {} if subclasses is None else dict(subclasses))
        object.__setattr__(instance, "_string", str(path_string))
        object.__setattr__(instance, "parts", list(parts))
        object.__setattr__(instance, "_extensions", None)
        return instance

    def __init__(self, path, model, subclasses: Optional[Mapping[str, str]] = None):
        """
        Constructor
        ===========

          >>> path = Path("Gene.name", model)

        You will not need to use this constructor directly. Instead,
        use the "make_path" method on the model to construct paths for you.

        @param path: the dotted path string (eg: Gene.proteins.name)
        @type path: str
        @param model: the model to validate the path against
        @type model: L{Model}
        @param subclasses: a dict which maps
                           subclasses (defaults to an empty dict)
        @type subclasses: dict
        """
        self.model = weakref.proxy(model)
        self.subclasses = {} if subclasses is None else dict(subclasses)
        self._extensions = None
        if isinstance(path, Class):
            self._string = path.name
            self.parts = [path]
        else:
            self._string = str(path)
            self.parts = model.parse_path_string(str(path), self.subclasses)

    def __str__(self):
        return self._string

    def __repr__(self):
        return "<" + self.__module__ + "." + self.__class__.__name__ + ": " + self._string + ">"

    def prefix(self):
        """
        The path one step above this path.
        ==================================

          >>> p1 = Path("Gene.exons.name", model)
          >>> p2 = p1.prefix()
          >>> print(p2)
          ... Gene.exons

        """
        parts = self.parts[:-1]
        if len(parts) < 1:
            raise PathParseError(str(self) + " does not have a prefix")
        s = ".".join([x.name for x in parts])
        return type(self)._from_validated_parts(self.model._unproxied(), s, parts, self.subclasses)

    def append(self, *elements):
        """
        Construct a new path by adding elements to the end of this one.
        ===============================================================

          >>> p1 = Path("Gene.exons", model)
          >>> p2 = p1.append("name")
          >>> print(p2)
          ... Gene.exons.name

        This is the inverse of prefix.
        """
        suffix = ".".join(elements)
        s = str(self) + "." + suffix
        model = self.model._unproxied()
        subclasses_map = dict(self.subclasses)
        if not elements:
            return Path(s, model, subclasses_map)

        current_class = self.get_class()
        path_prefix = str(self)
        parts = list(self.parts)
        for field_name in elements:
            field = current_class.get_field(field_name)
            parts.append(field)
            path_prefix = f"{path_prefix}.{field_name}"
            if isinstance(field, Reference):
                override_name = subclasses_map.get(path_prefix)
                if override_name is not None:
                    current_class = model.get_class(override_name)
                else:
                    current_class = field.type_class
            else:
                current_class = None
        return type(self)._from_validated_parts(model, s, parts, subclasses_map)

    @property
    def root(self):
        """
        The descriptor for the first part of the string.
        This should always a class descriptor.

        @rtype: L{intermine314.model.Class}
        """
        return self.parts[0]

    @property
    def end(self):
        """
        The descriptor for the last part of the string.

        @rtype: L{model.Class} or L{model.Field}
        """
        return self.parts[-1]

    def get_class(self):
        """
        Return the class object for this path, if it refers to a class
        or a reference. Attribute paths return None

        @rtype: L{model.Class}
        """
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
        """
        Return true if the path is a reference,
        eg: Gene.organism or Gene.proteins
        Note: Collections are ALSO references

        @rtype: boolean
        """
        return isinstance(self.end, Reference)

    def is_class(self):
        """
        Return true if the path just refers to a class, eg: Gene

        @rtype: boolean
        """
        return isinstance(self.end, Class)

    def is_attribute(self):
        """
        Return true if the path refers to an attribute, eg: Gene.length

        @rtype: boolean
        """
        return isinstance(self.end, Attribute)

    def __eq__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        return self._string == other._string and self._normalized_subclasses() == other._normalized_subclasses()

    def __hash__(self):
        return hash((self._string, self._normalized_subclasses()))

    def _normalized_subclasses(self):
        return tuple(sorted(self.subclasses.items()))

    @property
    def extensions(self):
        return _get_extensions_map(self)

    def __setattr__(self, name, value):
        _set_slot_or_extension(self, name, value)

    def __getattr__(self, name):
        return _get_extension_attr(self, name)
