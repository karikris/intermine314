import logging
import weakref
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .class_ import Class, ComposedClass
from .constants import NUMERIC_TYPES, NUMERIC_TYPES_NORMALIZED, _MODEL_PARSE_ERROR_MESSAGE
from .errors import ModelError, ModelParseError, PathParseError
from .fields import Reference
from .helpers import _split_path_segments
from .operators import Column
from .path import Path
from .xmlparse import parse_model_xml


class Model:
    """
    A class for representing the data model of an InterMine datawarehouse
    =====================================================================

    An abstraction of the database schema

    SYNOPSIS
    --------

        >>> service = Service("https://www.flymine.org/query/service")
        >>> model = service.model
        >>> model.get_class("Gene")
        <intermine314.model.Class: Gene>

    OVERVIEW
    --------

    This class represents the data model  - ie. an abstraction
    of the database schema. It can be used to introspect what
    data is available and how it is inter-related
    """

    NUMERIC_TYPES = NUMERIC_TYPES
    NUMERIC_TYPES_NORMALIZED = NUMERIC_TYPES_NORMALIZED

    LOG = logging.getLogger(__name__)

    def __init__(self, source: Any, service: Optional[Any] = None) -> None:
        """
        Constructor
        ===========

          >>> model = Model(xml)

        You will most like not need to create a model directly,
        instead get one from the Service object:

        @see: L{intermine314.webservice.Service}

        @param source: the model.xml, as a local file, string, or url
        """
        if source is None:
            raise ModelParseError(_MODEL_PARSE_ERROR_MESSAGE, "<none>", "Model source cannot be None")
        self.source: Any = source

        if service is not None:
            self.service: Optional[Any] = weakref.proxy(service)
        else:
            self.service = None

        self.classes: Dict[str, Class] = {}
        self._ancestry_cache: Dict[str, Tuple[Class, ...]] = {}
        self.parse_model(source)
        self.vivify()

        # Make sugary aliases
        self.table = self.column

    def parse_model(self, source: Any) -> None:
        """
        Create classes, attributes, references and
        collections from the model.xml
        =====================================================================

        The xml can be provided as a file, url or string. This method
        is called during instantiation - it does not need to be called
        directly.

        @param source:  the model.xml, as a local file, string, or url
        @raise ModelParseError: if there is a problem parsing the source
        """
        parse_model_xml(self, source)

    def vivify(self):
        """
        Make names point to instances and insert inherited fields
        =========================================================

        This method ensures the model is internally consistent. This method
        is called during instantiaton. It does not need to be called
        directly.

        @raise ModelError: if the names point to non-existent objects
        """
        self._ancestry_cache.clear()
        for c in list(self.classes.values()):
            ancestry = self._to_ancestry_tuple(c)
            c.parent_classes = list(ancestry)
            self.LOG.debug("{0.name} < {0.parent_classes}".format(c))
            for pc in ancestry:
                c.update_fields(pc.field_dict)
            for f in c.fields:
                f.type_class = self.classes.get(f.type_name)
                if hasattr(f, "reverse_reference_name") and f.reverse_reference_name != "":
                    rrn = f.reverse_reference_name
                    f.reverse_reference = f.type_class.field_dict[rrn]

    def _to_ancestry_tuple(self, cd: Class) -> Tuple[Class, ...]:
        """
        Internal ancestry resolver that memoizes immutable tuples.

        Uses iterative traversal with mutable path state to avoid
        allocation-heavy list/set copying in deep hierarchies.
        """
        cached = self._ancestry_cache.get(cd.name)
        if cached is not None:
            return cached

        self.LOG.debug("{0} < {1}".format(cd.name, cd.parents))
        ancestry: List[Class] = []
        seen_names: set[str] = set()
        stack: List[Tuple[Class, int]] = [(cd, 0)]
        path_names: List[str] = [cd.name]
        path_set: set[str] = {cd.name}

        while stack:
            current, parent_index = stack[-1]
            parents = current.parents
            if parent_index >= len(parents):
                stack.pop()
                ended = path_names.pop()
                path_set.remove(ended)
                continue

            parent_name = parents[parent_index]
            stack[-1] = (current, parent_index + 1)

            parent_class = self.classes.get(parent_name)
            if parent_class is None:
                continue

            if parent_class.name in path_set:
                cycle_path = " -> ".join(path_names + [parent_class.name])
                raise ModelError(f"Cycle detected in class ancestry: {cycle_path}")

            if parent_class.name not in seen_names:
                self.LOG.debug("{0} is ancestor of {1}".format(parent_class, cd.name))
                seen_names.add(parent_class.name)
                ancestry.append(parent_class)

            parent_cached = self._ancestry_cache.get(parent_class.name)
            if parent_cached is not None:
                for ancestor in parent_cached:
                    ancestor_name = ancestor.name
                    if ancestor_name in path_set:
                        cycle_path = " -> ".join(path_names + [ancestor_name])
                        raise ModelError(f"Cycle detected in class ancestry: {cycle_path}")
                    if ancestor_name in seen_names:
                        continue
                    self.LOG.debug("{0} is ancestor of {1}".format(ancestor, cd.name))
                    seen_names.add(ancestor_name)
                    ancestry.append(ancestor)
                continue

            stack.append((parent_class, 0))
            path_names.append(parent_class.name)
            path_set.add(parent_class.name)

        resolved = tuple(ancestry)
        self._ancestry_cache[cd.name] = resolved
        return resolved

    def to_ancestry(self, cd: Class) -> List[Class]:
        """
        Returns the lineage of the class
        ================================

            >>> classes = Model.to_ancestry(cd)

        Returns the class' parents, and all the class' parents' parents

        @rtype: list(L{intermine314.model.Class})
        """
        return list(self._to_ancestry_tuple(cd))

    def to_classes(self, classnames: Iterable[str]) -> List[Class]:
        """
        take a list of class names and return a list of classes
        =======================================================

            >>> classes = model.to_classes(["Gene", "Protein", "Organism"])

        This simply maps from a list of strings to a list of
        classes in the calling model.

        @raise ModelError: if the list of class names
                           includes ones that don't exist

        @rtype: list(L{intermine314.model.Class})
        """
        return list(map(self.get_class, classnames))

    def column(self, path: str, *rest: Any) -> Column:
        return Column(path, self, *rest)

    def __getattr__(self, name: str) -> Column:
        return self.column(name)

    def get_class(self, name: str) -> Class:
        """
        Get a class by its name, or by a dotted path
        ============================================

            >>> model = Model("https://www.flymine.org/query/service/model")
            >>> model.get_class("Gene")
            <intermine314.model.Class: Gene>
            >>> model.get_class("Gene.proteins")
            <intermine314.model.Class: Protein>

        This is the recommended way of retrieving a class from
        the model. As well as handling class names, you can also
        pass in a path such as "Gene.proteins" and get the
        corresponding class back (<intermine314.model.Class: Protein>)

        @raise ModelError: if the class name refers to a non-existant object

        @rtype: L{intermine314.model.Class}
        """
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

    def make_path(self, path: str, subclasses: Optional[Mapping[str, str]] = None) -> "Path":
        """
        Return a path object for the given path string
        ==============================================

            >>> path = model.make_path("Gene.organism.name")
            <intermine314.model.Path: Gene.organism.name>

        This is recommended manner of constructing path objects.

        @type path: str
        @type subclasses: dict

        @raise PathParseError: if there is a problem parsing the path string

        @rtype: L{intermine314.model.Path}
        """
        return Path(path, self, subclasses)

    def validate_path(self, path_string, subclasses: Optional[Mapping[str, str]] = None):
        """
        Validate a path
        ===============

            >>> try:
            ...     model.validate_path("Gene.symbol")
            ...     return "path is valid"
            ... except PathParseError:
            ...     return "path is invalid"
            "path is valid"

        When you don't need to interrogate relationships
        between paths, simply using this method to validate
        a path string is enough. It guarantees that there
        is a descriptor for each section of the string,
        with the appropriate relationships

        @raise PathParseError: if there is a problem parsing the path string
        """
        try:
            self.parse_path_string(path_string, subclasses)
            return True
        except PathParseError as e:
            raise PathParseError("Error parsing '%s' (subclasses: %s)" % (path_string, str(subclasses)), e)

    def parse_path_string(self, path_string, subclasses: Optional[Mapping[str, str]] = None):
        """
        Parse a path string into a list of descriptors - one for each section
        =====================================================================

            >>> parts = Model.parse_path_string(string)

        This method is used when making paths from a model, and
        when validating path strings. It probably won't need to
        be called directly.

        @see: L{intermine314.model.Model.make_path}
        @see: L{intermine314.model.Model.validate_path}
        @see: L{intermine314.model.Path}
        """
        descriptors = []
        subclasses_map = {} if subclasses is None else subclasses
        path_text = str(path_string)
        names = _split_path_segments(path_text)
        root_name = names[0]
        path_prefix = root_name

        root_descriptor = self.get_class(root_name)
        descriptors.append(root_descriptor)

        if root_name in subclasses_map:
            current_class = self.get_class(subclasses_map[root_name])
        else:
            current_class = root_descriptor

        for field_name in names[1:]:
            field = current_class.get_field(field_name)
            descriptors.append(field)
            path_prefix = f"{path_prefix}.{field_name}"

            if isinstance(field, Reference):
                override_name = subclasses_map.get(path_prefix)
                if override_name is not None:
                    current_class = self.get_class(override_name)
                else:
                    current_class = field.type_class
            else:
                current_class = None

        return descriptors

    def _unproxied(self):
        return self
