import weakref
from functools import reduce
from types import MappingProxyType
from typing import Dict, Iterator, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union

from .constants import _ID_FIELD_NAME, _ID_FIELD_TYPE, _ROOT_OBJECT_CLASS
from .errors import ModelError
from .fields import Attribute, Collection, Field, Reference

if TYPE_CHECKING:
    from .model import Model


class Class:
    """
    An abstraction of database tables in the data model
    ===================================================

    These objects refer to the table objects in the
    InterMine ORM layer.

    SYNOPSIS
    --------

    >>>  service = Service("https://www.flymine.org/query/service")
    >>>  model = service.model
    >>>
    >>>  if "Gene" in model.classes:
    ...      gene_cd = model.get_class("Gene")
    ...      print("Gene has", len(gene_cd.fields), "fields")
    ...      for field in gene_cd.fields:
    ...          print(" - ", field.name)

    OVERVIEW
    --------

    Each class can have attributes (columns) of various types,
    and can have references to other classes (tables), on either
    a one-to-one (references) or one-to-many (collections) basis

    Classes should not be instantiated by hand, but rather used
    as part of the model they belong to.

    """

    def __init__(self, name: str, parents: List[str], model: "Model", interface: bool = True) -> None:
        """
        Constructor - Creates a new Class descriptor
        ============================================

        >>> cd = intermine314.model.Class("Gene", ["SequenceFeature"])
        <intermine314.model.Class: Gene>

        This constructor is called when deserialising the
        model - you should have no need to create Classes by hand

        @param name: The name of this class
        @param parents: a list of parental names

        """
        self.name: str = name
        self.parents: List[str] = parents
        self.model: "Model" = model
        self.parent_classes: List["Class"] = []
        self.is_interface: bool = interface
        self.field_dict: Dict[str, Field] = {}
        self._fields_sorted_cache: Optional[Tuple[Field, ...]] = None
        self._attributes_cache: Optional[Tuple["Attribute", ...]] = None
        self._references_cache: Optional[Tuple["Reference", ...]] = None
        self._collections_cache: Optional[Tuple["Collection", ...]] = None
        self._fields_cache_hits = 0
        self._fields_cache_misses = 0
        self.has_id: bool = _ROOT_OBJECT_CLASS not in parents
        if self.has_id:
            # All InterMineObject classes have an id attribute.
            id_field = Attribute(_ID_FIELD_NAME, _ID_FIELD_TYPE, self)
            self.add_field(id_field)

    def _invalidate_fields_cache(self) -> None:
        self._fields_sorted_cache = None
        self._attributes_cache = None
        self._references_cache = None
        self._collections_cache = None

    def add_field(self, field: Field) -> None:
        self.field_dict[str(field.name)] = field
        self._invalidate_fields_cache()

    def update_fields(self, fields: Mapping[str, Field]) -> None:
        if not fields:
            return
        self.field_dict.update(fields)
        self._invalidate_fields_cache()

    def _sorted_fields(self) -> Tuple[Field, ...]:
        cached = self._fields_sorted_cache
        if cached is None:
            self._fields_cache_misses += 1
            cached = tuple(sorted(self.field_dict.values(), key=lambda field: field.name))
            self._fields_sorted_cache = cached
        else:
            self._fields_cache_hits += 1
        return cached

    @property
    def fields_cache_hits(self) -> int:
        return int(self._fields_cache_hits)

    @property
    def fields_cache_misses(self) -> int:
        return int(self._fields_cache_misses)

    def __repr__(self) -> str:
        return "<%s.%s %s.%s>" % (
            self.__module__,
            self.__class__.__name__,
            self.model.package_name if hasattr(self.model, "package_name") else "__test__",
            self.name,
        )

    @property
    def fields(self) -> Tuple[Field, ...]:
        """
        The fields of this class
        ========================

        The fields are returned sorted by name. Fields
        includes all Attributes, References and Collections

        @rtype: tuple(L{Field})
        """
        return self._sorted_fields()

    def __iter__(self) -> Iterator[Field]:
        return iter(self._sorted_fields())

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Field):
            return item in self.field_dict.values()
        else:
            return str(item) in self.field_dict

    @property
    def attributes(self) -> Tuple[Attribute, ...]:
        """
        The fields of this class which contain data
        ===========================================

        @rtype: tuple(L{Attribute})
        """
        if self._attributes_cache is None:
            self._attributes_cache = tuple(x for x in self._sorted_fields() if isinstance(x, Attribute))
        return self._attributes_cache

    @property
    def references(self) -> Tuple[Reference, ...]:
        """
        fields which reference other objects
        ====================================

        @rtype: tuple(L{Reference})
        """
        if self._references_cache is None:
            self._references_cache = tuple(
                x for x in self._sorted_fields() if isinstance(x, Reference) and not isinstance(x, Collection)
            )
        return self._references_cache

    @property
    def collections(self) -> Tuple[Collection, ...]:
        """
        fields which reference many other objects
        =========================================

        @rtype: tuple(L{Collection})
        """
        if self._collections_cache is None:
            self._collections_cache = tuple(x for x in self._sorted_fields() if isinstance(x, Collection))
        return self._collections_cache

    def get_field(self, name: str) -> Field:
        """
        Get a field by name
        ===================

        The standard way of retrieving a field

        @raise ModelError: if the Class does not have such a field

        @rtype: subclass of L{intermine314.model.Field}
        """
        if name in self.field_dict:
            return self.field_dict[name]
        else:
            raise ModelError("There is no field called %s in %s" % (name, self.name))

    def isa(self, other: Union["Class", str]) -> bool:
        """
        Check if self is, or inherits from other
        ========================================

        This method validates statements about inheritance.
        Returns true if the "other" is, or is within the
        ancestry of, this class

        Other can be passed as a name (str), or as the class object itself

        @rtype: boolean
        """
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
    """
    An abstraction of dynamic objects that are in two classes
    ==========================================================

    These objects are structural unions of two or more different data-types.
    """

    def __init__(self, parts, model):
        self.is_interface = True
        self.parts = parts
        self.model = weakref.proxy(model)
        self._field_dict_cache: Optional[Dict[str, Field]] = None
        self._field_dict_view: Optional[Mapping[str, Field]] = None

    @property
    def parents(self):
        return reduce(lambda ps, cls: ps + cls.parents, self.parts, [])

    @property
    def name(self):
        return "_".join(c.name for c in self.parts)

    @property
    def has_id(self):
        return _ROOT_OBJECT_CLASS not in self.parents

    @property
    def field_dict(self):
        """The combined field dictionary of all parts"""
        if self._field_dict_view is not None:
            return self._field_dict_view

        fields: Dict[str, Field] = {}
        if self.has_id:
            # All InterMineObject classes have an id attribute.
            fields[_ID_FIELD_NAME] = Attribute(_ID_FIELD_NAME, _ID_FIELD_TYPE, self)
        for p in self.parts:
            fields.update(p.field_dict)
        self._field_dict_cache = fields
        self._field_dict_view = MappingProxyType(fields)
        return self._field_dict_view

    @property
    def parent_classes(self):
        """The flattened list of parent classes, with the parts"""
        for p in self.parts:
            all_parents = [pc for pc in p.parent_classes]
        return all_parents + self.parts
