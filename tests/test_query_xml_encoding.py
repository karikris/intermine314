from __future__ import annotations

from xml.etree import ElementTree as ET

from intermine314.query.builder import Query
from intermine314.query.constraints import (
    BinaryConstraint,
    MultiConstraint,
    SubClassConstraint,
    UnaryConstraint,
)
from intermine314.query.pathfeatures import Join


class _Model:
    name = "test-model"


def _make_query() -> Query:
    query = Query(_Model(), validate=False)
    query.name = "minimal-export-query"
    query.description = "xml encoding smoke test"
    query.views = ["Gene.symbol", "Gene.length"]
    query.add_sort_order("Gene.symbol", "asc")
    query.joins = [Join("Gene.organism", "OUTER")]
    query.constraint_dict = {
        "A": BinaryConstraint("Gene.symbol", "=", "zen", "A"),
        "B": MultiConstraint("Gene.organism.name", "ONE OF", ["human", "mouse"], "B"),
        "C": UnaryConstraint("Gene.length", "IS NOT NULL", "C"),
    }
    query.uncoded_constraints = [SubClassConstraint("Gene.organism", "Organism")]
    return query


def _child_snapshot(xml: str):
    root = ET.fromstring(xml)
    rows = []
    for child in root:
        rows.append(
            {
                "tag": child.tag,
                "attributes": dict(child.attrib),
                "values": [node.text for node in child.findall("value")],
            }
        )
    return root, rows


def test_query_to_xml_uses_minimal_encoder_nodes():
    query = _make_query()
    root, rows = _child_snapshot(query.to_xml())

    assert root.tag == "query"
    assert root.attrib["name"] == "minimal-export-query"
    assert root.attrib["model"] == "test-model"
    assert root.attrib["view"] == "Gene.symbol Gene.length"
    assert root.attrib["sortOrder"] == "Gene.symbol asc"
    assert root.attrib["longDescription"] == "xml encoding smoke test"

    assert [row["tag"] for row in rows] == [
        "join",
        "constraint",
        "constraint",
        "constraint",
        "constraint",
    ]

    join = rows[0]
    assert join["attributes"] == {"path": "Gene.organism", "style": "OUTER"}

    binary = rows[1]
    assert binary["attributes"]["code"] == "A"
    assert binary["attributes"]["op"] == "="
    assert binary["attributes"]["value"] == "zen"
    assert binary["values"] == []

    multi = rows[2]
    assert multi["attributes"]["code"] == "B"
    assert multi["attributes"]["op"] == "ONE OF"
    assert multi["values"] == ["human", "mouse"]

    unary = rows[3]
    assert unary["attributes"]["code"] == "C"
    assert unary["attributes"]["op"] == "IS NOT NULL"
    assert "value" not in unary["attributes"]
    assert unary["values"] == []

    subclass = rows[4]
    assert subclass["attributes"] == {"path": "Gene.organism", "type": "Organism"}
    assert subclass["values"] == []


def test_to_formatted_xml_matches_to_xml_semantics():
    query = _make_query()
    xml_raw = query.to_xml()
    xml_formatted = query.to_formatted_xml()

    assert "\n" in xml_formatted

    raw_root, raw_rows = _child_snapshot(xml_raw)
    formatted_root, formatted_rows = _child_snapshot(xml_formatted)
    assert dict(formatted_root.attrib) == dict(raw_root.attrib)
    assert formatted_rows == raw_rows
