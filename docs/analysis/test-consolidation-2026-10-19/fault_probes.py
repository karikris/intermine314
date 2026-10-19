"""Reproduce consolidation fault evidence without changing tracked runtime files.

Run with the development environment's Python. Writes Zstandard Parquet receipts.
"""

import argparse
import inspect
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def contract_faults():
    from pytest import MonkeyPatch

    import intermine314
    import tests.test_minimal_contract as contract
    from intermine314.query import Query

    rows = []
    for fault in ("missing-export", "wrong-default", "wrong-kind", "wrong-alias"):
        with MonkeyPatch.context() as patch:
            if fault == "missing-export":
                patch.setattr(intermine314, "__all__", [n for n in intermine314.__all__ if n != "fetch_from_mine"])
                check = contract.test_public_entrypoints_match_contract_exactly
            elif fault == "wrong-alias":
                patch.setattr(Query, "all", Query.first)
                check = contract.test_alias_bindings_are_identical
            else:
                signature = inspect.signature(Query.first)
                parameters = list(signature.parameters.values())
                if fault == "wrong-default":
                    parameters[1] = parameters[1].replace(default="dict")
                else:
                    parameters[1:3] = [p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in parameters[1:3]]
                patch.setattr(Query.first, "__signature__", signature.replace(parameters=parameters), raising=False)
                check = contract.test_exact_signatures_include_defaults_order_and_parameter_kinds
            try:
                check()
            except AssertionError:
                rows.append(dict(fault=fault, detected=True))
            else:
                raise RuntimeError(f"Undetected fault: {fault}")
    return rows


def scheduler_faults():
    faults = [
        ("footer-acceptance", "query/parallel_offset.py", "if next(iterator, end) is not end:", "if False:", "test_parallel_page_validation.py"),
        ("reservation-release", "query/parallel_offset.py", "queue.clear()", "None", "test_parallel_cleanup.py"),
        ("copy-order-policy", "export/runtime.py", "'preserve_insertion_order': bool(ordered)", "'preserve_insertion_order': False", "test_copy_profiles.py"),
        ("summary-format", "query/executor.py", "'jsonrows' if summary_path is not None else self.row", "'dict' if summary_path is not None else self.row", "test_query_summaries.py"),
    ]
    rows = []
    for name, file, before, after, test in faults:
        with tempfile.TemporaryDirectory(prefix="consolidation-fault-") as directory:
            target = Path(directory)
            for tree in ("src", "tests", "benchmarks/contracts"):
                shutil.copytree(ROOT / tree, target / tree, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
            shutil.copy(ROOT / "pyproject.toml", target / "pyproject.toml")
            path = target / "src/intermine314" / file
            source = path.read_text()
            if before not in source:
                raise RuntimeError(f"Fault site changed: {name}")
            path.write_text(source.replace(before, after, 1))
            # Isolated mode avoids editable-install path precedence; bootstrap
            # explicitly chooses this disposable runtime and its test fixtures.
            program = (
                "import sys; sys.path[:0] = " + repr([str(target / "src"), str(target)]) + "; "
                "import pytest; raise SystemExit(pytest.main(['-q', '-x', " + repr("tests/" + test) + "]))"
            )
            result = subprocess.run([sys.executable, "-I", "-c", program], cwd=target,
                                    text=True, capture_output=True, timeout=45)
            if result.returncode != 1 or "FAILED" not in result.stdout or "ERROR collecting" in result.stdout:
                raise RuntimeError(f"Fault did not fail an assertion: {name}\n{result.stdout}\n{result.stderr}")
            rows.append(dict(fault=name, detected=True, exit_code=1, output=result.stdout[-1800:]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output.mkdir(parents=True, exist_ok=True)
    for name, probe in (("contract", contract_faults), ("scheduler", scheduler_faults)):
        rows = probe()
        pq.write_table(pa.Table.from_pylist(rows), args.output / f"{name}-faults.parquet", compression="zstd")
        print(f"{name}: {len(rows)} faults detected")


if __name__ == "__main__":
    main()
