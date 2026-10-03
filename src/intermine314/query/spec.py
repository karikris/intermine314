from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree import ElementTree as _ET

from intermine314.compatibility import resolve_compatibility
from intermine314.query.constraints import (
    BinaryConstraint,
    CodedConstraint,
    MultiConstraint,
    SubClassConstraint,
    UnaryConstraint,
)
from intermine314.query.pathfeatures import Join


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


def _append_constraint_xml(query, constraint) -> None:
    element = _ET.SubElement(query, "constraint")
    element.set("path", _xml_attr(constraint.path))

    if isinstance(constraint, SubClassConstraint):
        element.set("type", _xml_attr(constraint.subclass))
        return
    if isinstance(constraint, BinaryConstraint):
        element.set("op", _xml_attr(constraint.op))
        element.set("code", _xml_attr(constraint.code))
        element.set("value", _xml_attr(constraint.value))
        return
    if isinstance(constraint, UnaryConstraint):
        element.set("op", _xml_attr(constraint.op))
        element.set("code", _xml_attr(constraint.code))
        return
    if isinstance(constraint, MultiConstraint):
        element.set("op", _xml_attr(constraint.op))
        element.set("code", _xml_attr(constraint.code))
        for value in constraint.values:
            node = _ET.SubElement(element, "value")
            node.text = _xml_attr(value)
        return

    raise TypeError(
        "Unsupported constraint type for minimal XML encoder: "
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

    for join in spec.joins:
        _append_join_xml(query, join)
    for constraint in spec.constraints:
        _append_constraint_xml(query, constraint)
    return query


def query_spec_to_xml(spec: QuerySpec) -> str:
    query = query_spec_to_element(spec)
    return _ET.tostring(query, encoding="unicode", short_empty_elements=True)


def query_spec_to_formatted_xml(spec: QuerySpec) -> str:
    query = query_spec_to_element(spec)
    _ET.indent(query, space="  ")
    return _ET.tostring(query, encoding="unicode", short_empty_elements=True)
