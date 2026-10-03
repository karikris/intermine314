from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree import ElementTree as _ET

from intermine314.compatibility import resolve_compatibility
from intermine314.query.constraints import (
    BinaryConstraint,
    CodedConstraint,
    MultiConstraint,
    SubClassConstraint,
    TernaryConstraint,
)
from intermine314.query.pathfeatures import Join, PathDescription


@dataclass(frozen=True)
class TemplateMetadata:
    """Immutable named-template identity and saved presentation metadata."""

    user_name: str = ""
    title: str = ""
    view_types: tuple[str, ...] = ()
    comment: str = ""


@dataclass(frozen=True)
class QuerySpec:
    root_class: str | None = None
    views: tuple[str, ...] = ()
    constraints: tuple[object, ...] = ()
    joins: tuple[Join, ...] = ()
    sort_order: str = ""
    name: str = ""
    description: str = ""
    model_name: str = ""
    compatibility: str = field(default="native", kw_only=True)
    constraint_logic: str = field(default="", kw_only=True)
    decimal_paths: tuple[str, ...] = field(default=(), kw_only=True)
    path_descriptions: tuple[PathDescription, ...] = field(default=(), kw_only=True)

    template: TemplateMetadata | None = field(default=None, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "compatibility", resolve_compatibility(self.compatibility))


def _xml_attr(value) -> str:
    if value is None:
        return ""
    return str(value)


def _append_join_xml(query, join) -> None:
    element = _ET.SubElement(query, "join")
    element.set("path", _xml_attr(join.path))
    element.set("style", _xml_attr(join.style))


def _append_constraint_xml(query, constraint, *, compatibility="native") -> None:
    element = _ET.SubElement(query, "constraint")
    element.set("path", _xml_attr(constraint.path))

    if isinstance(constraint, SubClassConstraint):
        element.set("type", _xml_attr(constraint.subclass))
        return
    if isinstance(constraint, CodedConstraint):
        # Public dictionaries encode LOOKUP extraValue, loop XML operators,
        # and list names. Multi/range/ISA values are XML child elements.
        for key, value in constraint.to_dict().items():
            if (key == "value" and compatibility == "native"
                    and isinstance(constraint, BinaryConstraint)
                    and not isinstance(constraint, TernaryConstraint)):
                # Existing native scalar XML normalizes raw None to empty;
                # public dictionaries and restored LOOKUP retain str(value).
                value = constraint.value
            if key == "value" and isinstance(constraint, MultiConstraint):
                for item in value:
                    node = _ET.SubElement(element, "value")
                    node.text = _xml_attr(item)
            else:
                element.set(key, _xml_attr(value))
        return

    raise TypeError(
        "Unsupported constraint type for XML encoder: "
        + constraint.__class__.__name__
    )


def query_spec_to_element(spec: QuerySpec):
    query = _ET.Element("query")
    query.set("name", _xml_attr(spec.name))
    query.set("model", _xml_attr(spec.model_name))
    query.set("view", _xml_attr(" ".join(spec.views)))
    query.set("sortOrder", _xml_attr(spec.sort_order))
    query.set("longDescription", _xml_attr(spec.description))
    coded = [constraint for constraint in spec.constraints if isinstance(constraint, CodedConstraint)]
    if len(coded) > 1:
        logic = spec.constraint_logic or " and ".join(constraint.code for constraint in coded)
        query.set("constraintLogic", logic)

    for description in spec.path_descriptions:
        element = _ET.SubElement(query, "pathDescription")
        # Public dictionaries retain the original client's `path` contract;
        # saved/server XML uses the canonical InterMine `pathString` spelling.
        element.set("pathString", _xml_attr(description.path))
        element.set("description", _xml_attr(description.description))
    for join in spec.joins:
        _append_join_xml(query, join)
    for constraint in spec.constraints:
        _append_constraint_xml(query, constraint, compatibility=spec.compatibility)
        if spec.template is not None:
            element = query[-1]
            element.set("editable", "true" if constraint.editable else "false")
            if constraint.optional:
                element.set("switchable", constraint.get_switchable_status())
    if spec.template is not None:
        wrapper = _ET.Element("template", name=spec.name, userName=spec.template.user_name,
                              title=spec.template.title, dataTypes=" ".join(spec.template.view_types))
        if spec.template.comment:
            wrapper.set("comment", spec.template.comment)
        wrapper.append(query)
        return wrapper
    return query


def query_spec_to_xml(spec: QuerySpec) -> str:
    query = query_spec_to_element(spec)
    return _ET.tostring(query, encoding="unicode", short_empty_elements=True)


def query_spec_to_formatted_xml(spec: QuerySpec) -> str:
    query = query_spec_to_element(spec)
    _ET.indent(query, space="  ")
    return _ET.tostring(query, encoding="unicode", short_empty_elements=True)
