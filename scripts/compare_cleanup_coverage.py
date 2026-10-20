"""Compare coverage by qualified AST declaration, including moved source lines."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BASE = "5ea5b765d9edfff04fda3aff82cc9a14bc34e977"


def declarations(source):
    result = {}

    def visit(node, scope=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{scope}.{child.name}" if scope else child.name
                if not isinstance(child, ast.ClassDef):
                    result[name] = child
                visit(child, name)
            else:
                visit(child, scope)

    visit(ast.parse(source))
    return result


def points(node, coverage):
    # Match the AST position, never the physical line number, across a move.
    byline = {}

    def walk(value, path=""):
        if isinstance(value, ast.AST):
            if hasattr(value, "lineno"):
                byline.setdefault(value.lineno, []).append(
                    path + ":" + type(value).__name__
                )
            for field, item in ast.iter_fields(value):
                if field not in {
                    "lineno",
                    "col_offset",
                    "end_lineno",
                    "end_col_offset",
                }:
                    walk(item, path + "." + field)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + f"[{index}]")

    walk(node)

    def key(line):
        if line < 0:
            return "<exit>"
        return "|".join(sorted(byline.get(line, ()))) or "<structural-line>"

    lines = {key(n) for n in coverage.get("executed_lines", [])}
    branches = {(key(a), key(b)) for a, b in coverage.get("executed_branches", [])}
    return lines, branches


def compare(before, after, output):
    before = json.loads(Path(before).read_text())["files"]
    after = json.loads(Path(after).read_text())["files"]
    rows = []
    for path, oldcoverage in before.items():
        old = declarations(
            subprocess.check_output(["git", "show", BASE + ":" + path], text=True)
        )
        new = declarations(Path(path).read_text()) if Path(path).exists() else {}
        for name, node in old.items():
            oldhit = oldcoverage["functions"].get(name, {})
            newhit = after.get(path, {}).get("functions", {}).get(name, {})
            current = new.get(name)
            oldhash = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()
            ).hexdigest()
            newhash = (
                hashlib.sha256(
                    ast.dump(current, include_attributes=False).encode()
                ).hexdigest()
                if current
                else None
            )
            a, b = points(node, oldhit)
            c, d = points(current, newhit) if current else (set(), set())
            same = oldhash == newhash
            rows.append(
                dict(
                    path=path,
                    qualified_name=name,
                    before_ast_sha256=oldhash,
                    after_ast_sha256=newhash,
                    disposition="unchanged"
                    if same
                    else "replaced"
                    if current
                    else "retired",
                    before_line_points=json.dumps(sorted(a)),
                    after_line_points=json.dumps(sorted(c)),
                    before_branch_points=json.dumps(sorted(b)),
                    after_branch_points=json.dumps(sorted(d)),
                    before_lines=len(a),
                    after_lines=len(c),
                    before_branches=len(b),
                    after_branches=len(d),
                    lost_unchanged_lines=json.dumps(sorted(a - c)) if same else "[]",
                    lost_unchanged_branches=json.dumps(sorted(b - d)) if same else "[]",
                )
            )
    pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd")
    losses = [
        r
        for r in rows
        if r["lost_unchanged_lines"] != "[]" or r["lost_unchanged_branches"] != "[]"
    ]
    print(json.dumps(losses, indent=2))
    if losses:
        raise RuntimeError("Unexplained coverage losses in unchanged declarations")


if __name__ == "__main__":
    import sys

    compare(*sys.argv[1:])
