"""Select explicit CI obligations; full source coverage remains the release default."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("base", "plots", "speed", "combined")
FOCUSED = {
    "speed": (
        "test_json_precision.py", "test_result_formats.py", "test_result_objects.py",
        "test_model_query_integration.py", "test_typed_arrow.py",
        "test_parallel_page_validation.py", "test_template_execution.py", "test_cold_imports.py",
    ),
    "plots": ("test_bar_chart_helpers.py", "test_parquet_plots.py", "test_cold_imports.py"),
}


def scope_for(event, install):
    if install not in MODES:
        raise ValueError(f"Unknown installation mode: {install}")
    return install if event == "pull_request" and install in FOCUSED else "full"


def select_tests(scope, root=ROOT):
    if scope == "full":
        paths = ("tests",)
        valid = any((root / "tests").glob("test_*.py"))
    elif scope in FOCUSED:
        paths = tuple("tests/" + name for name in FOCUSED[scope])
        valid = bool(paths) and all((root / path).is_file() for path in paths)
    else:
        raise ValueError(f"Unknown test scope: {scope}")
    if not valid:
        raise RuntimeError(f"Empty or missing test selection: {scope}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("full", *FOCUSED))
    parser.add_argument("--event", default=os.getenv("GITHUB_EVENT_NAME", "local"))
    parser.add_argument("--install", choices=MODES, default="combined")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    paths = select_tests(args.scope or scope_for(args.event, args.install))
    # pyproject already supplies -q. A second one hides the collection total
    # consumed by test_inventory_summary; keep it only for ordinary execution.
    options = ["--collect-only"] if args.collect_only else ["-q"]
    if args.junit:
        options.append("--junitxml=" + str(args.junit))
    return subprocess.run([sys.executable, "-m", "pytest", *options, *paths], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
