"""Fail closed on release context and exact verified distribution artifacts."""

import argparse
import hashlib
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


def require_artifact_id(value):
    if not isinstance(value, str) or not value or not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise RuntimeError("A non-empty immutable artifact ID is required")
    return value


def verify_artifacts(root, directory, *, sha):
    version = project_version(root)
    try:
        manifest = json.loads((directory / "artifact-manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError("Artifact manifest is missing or invalid") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("status") != "ok":
        raise RuntimeError("Distribution verification is incomplete")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if manifest.get("version") != version or manifest.get("source_revision") != sha or actual != sha:
        raise RuntimeError("Artifact version or source commit differs")
    if manifest.get("source_dirty") is not False:
        raise RuntimeError("Artifacts must be built from a clean release checkout")
    checkout = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root, capture_output=True)
    if checkout.returncode != 0:
        raise RuntimeError("Release checkout contains changes after checkout")
    verifier = runpy.run_path(str(ROOT / "scripts/verify_distribution.py"))
    if set(manifest.get("modes", [])) != set(verifier["INSTALL_EXTRAS"]) or manifest.get("docs_verified") is not True:
        raise RuntimeError("Required installation modes or documentation were not verified")
    entries = manifest.get("artifacts", [])
    if len(entries) != 2:
        raise RuntimeError("Exactly one wheel and one sdist are required")
    files = []
    for entry in entries:
        relative = Path(entry["file"])
        if len(relative.parts) != 2 or relative.parts[0] != "dist" or ".." in relative.parts:
            raise RuntimeError("Artifact path must be a direct dist member")
        path = directory / relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            raise RuntimeError("Verified artifact is missing or outside its directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("Verified artifact hash differs")
        files.append(path)
    wheels = [p for p in files if p.name.endswith(".whl")]
    sdists = [p for p in files if p.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1 or set((directory / "dist").iterdir()) != set(files):
        raise RuntimeError("Distribution directory differs from the verified file set")
    verifier["inspect_artifacts"](wheels[0], sdists[0], root=root)
    return {"status": "ok", "version": version, "source_revision": sha,
            "artifacts": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--artifact-id")
    parser.add_argument("--check-artifact-id")
    args = parser.parse_args()
    if args.check_artifact_id is not None:
        report = {"artifact_id": require_artifact_id(args.check_artifact_id)}
    elif args.artifacts:
        artifact_id = require_artifact_id(args.artifact_id)
        report = verify_artifacts(ROOT, args.artifacts, sha=os.environ.get("GITHUB_SHA", ""))
        report["artifact_id"] = artifact_id
    else:
        report = release_context(ROOT, event=os.environ.get("GITHUB_EVENT_NAME", ""),
                                 ref=os.environ.get("GITHUB_REF", ""), sha=os.environ.get("GITHUB_SHA", ""))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            if "publish" in report:
                stream.write(f"publish={str(report['publish']).lower()}\n")
            if "artifact_id" in report:
                stream.write(f"artifact_id={report['artifact_id']}\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
