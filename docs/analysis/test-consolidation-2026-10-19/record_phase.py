"""Record consolidation gates without weakening the original coverage baseline."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def record(phase, work):
    original = json.loads((HERE / "runtime-hashes.json").read_text())
    current = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "src/intermine314").rglob("*.py")
    }
    if current != original:
        raise RuntimeError("Runtime changed during test-only consolidation")
    points = HERE / "coverage-points.parquet"
    if points.exists():
        baseline = {"files": {}}
        for point in pq.read_table(points).to_pylist():
            if not point["baseline_covered"]:
                continue
            entry = baseline["files"].setdefault(
                point["path"], {"executed_lines": [], "executed_branches": []}
            )
            value = (point["line"] if point["kind"] == "executed_lines"
                     else [point["line"], point["arc_to"]])
            entry[point["kind"]].append(value)
    else:
        # Initial phases precede persistence of the original point set.
        baseline = json.loads(
            Path("/tmp/intermine314-consolidation/coverage.json").read_text()
        )
    measured = json.loads((work / "coverage.json").read_text())
    changes = []
    for path, before in baseline["files"].items():
        after = measured["files"][path]
        for kind in ("executed_lines", "executed_branches"):
            old = {tuple(v) if isinstance(v, list) else v for v in before[kind]}
            new = {tuple(v) if isinstance(v, list) else v for v in after[kind]}
            for value in old - new:
                changes.append(dict(path=path, kind=kind, value=str(value)))
    if changes:
        (work / "coverage-losses.json").write_text(json.dumps(changes, indent=2))
        raise RuntimeError(f"Previously covered runtime points lost: {len(changes)}")
    audit = json.loads(
        (ROOT / "docs/analysis/intermine-api-final-coverage.json").read_text()
    )
    if (
        audit["summary"] != {"passing-scoped": 331, "tested-departure": 129}
        or len(audit["departures"]) != 43
    ):
        raise RuntimeError("Historical API obligations changed")
    from scripts.audit_api_coverage import verify_execution_inputs

    verify_execution_inputs(audit["execution_inputs"])
    report = dict(
        phase=phase,
        tested_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        cases=audit["pytest"]["outcomes"],
        scopes=audit["summary"],
        semantic_nodes=len(audit["test_catalog"]),
        coverage=measured["totals"],
        lost_runtime_points=0,
        runtime_unchanged=True,
        test_python_lines=sum(
            len(p.read_text().splitlines()) for p in (ROOT / "tests").rglob("*.py")
        ),
    )
    pq.write_table(
        pa.Table.from_pylist([dict(report=json.dumps(report))]),
        HERE / f"phase{phase}.parquet",
        compression="zstd",
    )
    (HERE / f"phase{phase}.md").write_text(
        f"# Phase {phase} acceptance\n\nTested task head `{report['tested_commit']}`. "
        f"{report['cases']}; {report['semantic_nodes']} semantic nodes; 460 historical symbols "
        "(331 passing scopes, 129 departures / 43 groups). Runtime unchanged and no previously "
        "covered runtime lines/branches lost. Full exact-push CI is required before the next phase.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", type=int)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(ROOT))
    record(args.phase, args.work)
