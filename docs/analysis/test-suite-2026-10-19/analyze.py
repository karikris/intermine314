"""Inventory the unchanged suite and evaluate a proposed parameter-only reduction.

Inputs are produced by run.py in an external work directory. Coverage SQLite
reading is pinned to coverage.py 7.16.2; portable outputs are Parquet.
"""

import argparse
import ast
import collections
import hashlib
import json
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def write(name, rows):
    pq.write_table(
        pa.Table.from_pylist(rows), HERE / f"{name}.parquet", compression="zstd"
    )


def proposed_drop(row):
    """Review candidate only: do not edit or delete tests."""
    p, f, values = row["path"], row["function"], row["params"]
    if p == "tests/test_parallel_page_validation.py" and f in {
        "test_invalid_terminal_status_never_releases_full_page",
        "test_successful_empty_short_and_exact_pages_are_closed_once",
    }:
        return values["profile"] == "'legacy'"
    if (
        p == "tests/test_query_summaries.py"
        and f == "test_summary_forces_raw_jsonrows_before_object_augmentation"
    ):
        return (
            "intermine314.service.legacy.Service" in values["client"]
            and values["mode"] != "None"
        )
    if (
        p == "tests/test_copy_profiles.py"
        and f == "test_copy_uses_one_profiled_connection_and_correct_order_policy"
    ):
        return values["profile"] not in {"'cpu2'", "'cpu16'"} and not (
            values["ordered"] == values["single_file"] == "True"
        )
    if (
        p == "tests/test_resource_integration.py"
        and f == "test_profile_reaches_owned_duckdb_and_polars_pipeline"
    ):
        return values["profile"] not in {"'cpu2'", "'cpu16'"}
    return False


def main(work):
    collection = json.loads((work / "baseline-collection.json").read_text())
    audit = json.loads(
        (ROOT / "docs/analysis/intermine-api-final-coverage.json").read_text()
    )
    times = {}
    for case in ET.parse(work / "baseline.xml").getroot().iter("testcase"):
        node = (
            case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        )
        times[node] = float(case.attrib["time"])
    inventory, groups, definitions = (
        [],
        collections.defaultdict(list),
        collections.defaultdict(list),
    )
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        tests = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
        ]
        inventory.append(
            dict(
                path=str(path.relative_to(ROOT)),
                lines=len(source.splitlines()),
                functions=len(tests),
                sha256=hashlib.sha256(source.encode()).hexdigest(),
            )
        )
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
                node.name = "NORMALIZED"
                definitions[ast.dump(node, include_attributes=False)].append(
                    dict(
                        path=str(path.relative_to(ROOT)),
                        name=name,
                        line=node.lineno,
                        end=node.end_lineno,
                    )
                )
    write("inventory", inventory)
    write(
        "exact_duplicates",
        [
            dict(group=i, **r)
            for i, values in enumerate(definitions.values())
            if len(values) > 1
            for r in values
        ],
    )
    drops = {r["nodeid"] for r in collection if proposed_drop(r)}
    (work / "deselect.json").write_text(json.dumps(sorted(drops), indent=2) + "\n")
    for row in collection:
        row["params_json"] = json.dumps(row.pop("params"), sort_keys=True)
        row["seconds"] = times.get(row["nodeid"], 0)
        row["proposed_drop"] = row["nodeid"] in drops
        row["audit_catalog"] = row["nodeid"] in audit["test_catalog"]
        groups[row["path"], row["function"]].append(row)
    write("cases", collection)
    write(
        "parameter_groups",
        [
            dict(
                path=p,
                function=f,
                cases=len(items),
                proposed_drop=sum(r["proposed_drop"] for r in items),
                seconds=sum(r["seconds"] for r in items),
            )
            for (p, f), items in groups.items()
        ],
    )

    coverage = json.loads((work / "coverage.json").read_text())
    write(
        "coverage_files",
        [
            dict(
                path=p,
                **row["summary"],
                missing_line_numbers=row["missing_lines"],
                missing_branch_arcs=json.dumps(row["missing_branches"]),
            )
            for p, row in coverage["files"].items()
        ],
    )
    statements = {
        p: set(r["executed_lines"]) | set(r["missing_lines"])
        for p, r in coverage["files"].items()
    }
    branches = {
        p: set(map(tuple, r["executed_branches"])) for p, r in coverage["files"].items()
    }
    points, point_ids = {}, []

    def point(value):
        if value not in points:
            points[value] = len(point_ids)
            point_ids.append(value)
        return points[value]

    all_hits, call_hits = collections.defaultdict(set), collections.defaultdict(set)
    db = sqlite3.connect(work / ".coverage")
    for context, absolute, start, end in db.execute(
        "SELECT c.context,f.path,a.fromno,a.tono FROM arc a JOIN context c ON c.id=a.context_id JOIN file f ON f.id=a.file_id"
    ):
        path = str(Path(absolute).relative_to(ROOT))
        node, _, phase = context.rpartition("|")
        if not node:
            node = "<collection>"
        hit = set()
        for line in (start, end):
            if line in statements.get(path, ()):
                hit.add(point(("line", path, line, 0)))
        if (start, end) in branches.get(path, ()):
            hit.add(point(("branch", path, start, end)))
        all_hits[node].update(hit)
        if phase == "call":
            call_hits[node].update(hit)
    before = set().union(*all_hits.values())
    after = set().union(*(hits for node, hits in all_hits.items() if node not in drops))
    write(
        "coverage_losses",
        [
            dict(
                kind=point_ids[i][0],
                path=point_ids[i][1],
                start=point_ids[i][2],
                end=point_ids[i][3],
            )
            for i in before - after
        ]
        or [dict(kind="none", path="", start=0, end=0)],
    )
    frequency = collections.Counter(i for hits in call_hits.values() for i in hits)
    signatures = collections.Counter(tuple(sorted(h)) for h in call_hits.values() if h)
    write(
        "case_coverage",
        [
            dict(
                nodeid=r["nodeid"],
                call_points=len(call_hits[r["nodeid"]]),
                unique_call_points=sum(
                    frequency[i] == 1 for i in call_hits[r["nodeid"]]
                ),
                same_coverage_cases=signatures.get(
                    tuple(sorted(call_hits[r["nodeid"]])), 0
                ),
            )
            for r in collection
        ],
    )
    trace = json.loads((work / "api-trace.json").read_text())
    for path, expected in trace["source_hashes"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("API trace is stale for " + path)
    lost_symbols = []
    for symbol, hits in trace["calls"].items():
        call_nodes = {n for n, phases in hits.items() if "call" in phases}
        if call_nodes and not call_nodes - drops:
            lost_symbols.append(symbol)
    summary = dict(
        source_revision="7a1ec32333fd79238f2e93957cdfaeac807f29eb",
        cases=len(collection),
        proposed_drop=len(drops),
        proposed_retain=len(collection) - len(drops),
        touched_functions=sum(
            any(r["proposed_drop"] for r in items) for items in groups.values()
        ),
        audit_catalog_drops=len(drops & audit["test_catalog"].keys()),
        covered_points_before=len(before),
        covered_points_after=len(after),
        lost_points=[point_ids[i] for i in before - after],
        lost_audited_symbols=lost_symbols,
        dropped_baseline_seconds=sum(times.get(n, 0) for n in drops),
        baseline=json.loads((work / "baseline-result.json").read_text()),
        coverage_run=json.loads((work / "coverage-result.json").read_text()),
        totals=coverage["totals"],
        zero_call_coverage=sum(not call_hits[r["nodeid"]] for r in collection),
    )
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    reports = [dict(kind="baseline-analysis", report=json.dumps(summary))]
    if (work / "trial-coverage.json").exists():
        reduced = json.loads((work / "trial-coverage.json").read_text())
        changes = {}
        for change, left, right in (
            ("loss", coverage, reduced),
            ("gain", reduced, coverage),
        ):
            changes[change] = []
            for path, values in left["files"].items():
                for kind in ("executed_lines", "executed_branches"):
                    before = {
                        tuple(v) if isinstance(v, list) else v for v in values[kind]
                    }
                    after = {
                        tuple(v) if isinstance(v, list) else v
                        for v in right["files"][path][kind]
                    }
                    if before - after:
                        changes[change].append(
                            dict(path=path, kind=kind, points=sorted(before - after))
                        )
        comparison = dict(
            trial=json.loads((work / "trial-result.json").read_text()),
            baseline=coverage["totals"],
            reduced=reduced["totals"],
            **changes,
        )
        (work / "trial-comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n"
        )
    if (work / "trial-comparison.json").exists():
        reports.append(
            dict(
                kind="reduced-trial",
                report=(work / "trial-comparison.json").read_text(),
            )
        )
    write("verification", reports)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    main(parser.parse_args().work)
