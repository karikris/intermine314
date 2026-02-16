import weakref
from functools import reduce
from typing import Mapping, Optional

from .class_ import Class
from .errors import PathParseError
from .fields import Attribute, Reference
from .helpers import _copy_subclasses


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
        self.subclasses = _copy_subclasses(subclasses)
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
        parts = list(self.parts)
        parts.pop()
        if len(parts) < 1:
            raise PathParseError(str(self) + " does not have a prefix")
        s = ".".join([x.name for x in parts])
        return Path(s, self.model._unproxied(), self.subclasses)

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
        s = str(self) + "." + ".".join(elements)
        return Path(s, self.model._unproxied(), self.subclasses)

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
        return str(self) == str(other)

    def __hash__(self):
        i = hash(str(self))
        return reduce(lambda a, b: a ^ b, [hash(k) ^ hash(v) for k, v in list(self.subclasses.items())], i)
