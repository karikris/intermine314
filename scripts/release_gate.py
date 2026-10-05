"""Fail closed on release context and exact verified distribution artifacts."""

import argparse
import json
import os
import runpy
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
project_version = runpy.run_path(str(ROOT / "scripts/release_metadata.py"))["project_version"]


def release_context(root, *, event, ref, sha):
    version = project_version(root)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if sha != actual:
        raise RuntimeError("Release checkout differs from triggering commit")
    if event == "workflow_dispatch":
        return {"publish": False, "version": version, "source_revision": sha, "reason": "Manual validation only"}
    if event != "push" or ref != f"refs/tags/v{version}":
        raise RuntimeError("Publication requires a pushed tag matching the project version")
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/master"], cwd=root, capture_output=True)
    if ancestry.returncode != 0:
        raise RuntimeError("Release commit must belong to origin/master")
    return {"publish": True, "version": version, "source_revision": sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = release_context(ROOT, event=os.environ.get("GITHUB_EVENT_NAME", ""),
                             ref=os.environ.get("GITHUB_REF", ""), sha=os.environ.get("GITHUB_SHA", ""))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            stream.write(f"publish={str(report['publish']).lower()}\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
