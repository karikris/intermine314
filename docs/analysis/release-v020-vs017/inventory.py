"""Compare the hash-verified PyPI sdist with the reviewed checkout."""

import ast
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = Path("/tmp/intermine-v017-review/sdist/intermine314-0.1.7")


def write(name, rows):
    pq.write_table(
        pa.Table.from_pylist(rows), HERE / f"{name}.parquet", compression="zstd"
    )


def main():
    files, symbols = [], []
    for version, root in [("0.1.7", OLD), ("candidate", ROOT)]:
        for path in sorted((root / "src/intermine314").rglob("*.py")):
            source = path.read_text()
            tree = ast.parse(source)
            name = str(path.relative_to(root / "src"))
            files.append(
                dict(
                    version=version,
                    path=name,
                    lines=len(source.splitlines()),
                    bytes=len(source.encode()),
                    sha256=hashlib.sha256(source.encode()).hexdigest(),
                )
            )

            def collect(nodes, prefix):
                for node in nodes:
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        if not node.name.startswith("_") or node.name in (
                            "__init__",
                            "__iter__",
                            "__len__",
                            "__getitem__",
                        ):
                            symbols.append(
                                dict(
                                    version=version,
                                    path=name,
                                    symbol=prefix + node.name,
                                    line=node.lineno,
                                )
                            )
                        if isinstance(node, ast.ClassDef):
                            collect(node.body, prefix + node.name + ".")

            collect(tree.body, "")
    write("code_inventory", files)
    write("declared_symbols", symbols)
    old = {r["path"]: r for r in files if r["version"] == "0.1.7"}
    new = {r["path"]: r for r in files if r["version"] == "candidate"}
    write(
        "module_changes",
        [
            dict(
                path=p,
                status="added"
                if p not in old
                else "removed"
                if p not in new
                else "unchanged"
                if old[p]["sha256"] == new[p]["sha256"]
                else "changed",
                old_lines=old.get(p, {}).get("lines", 0),
                new_lines=new.get(p, {}).get("lines", 0),
            )
            for p in sorted(old.keys() | new.keys())
        ],
    )
    for version in ("0.1.7", "candidate"):
        selected = [r for r in files if r["version"] == version]
        print(
            version,
            len(selected),
            sum(r["lines"] for r in selected),
            sum(r["bytes"] for r in selected),
        )
    a = {(r["path"], r["symbol"]) for r in symbols if r["version"] == "0.1.7"}
    b = {(r["path"], r["symbol"]) for r in symbols if r["version"] == "candidate"}
    print(
        "Removed/moved declarations (not a runtime API availability verdict):",
        sorted(a - b),
    )


if __name__ == "__main__":
    main()
