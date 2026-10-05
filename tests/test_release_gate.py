"""Exercise release decisions with real temporary Git history, never publish."""

import hashlib
import json
import shutil
import subprocess

import pytest

from scripts import release_gate
from tests.test_distribution_versions import artifacts


@pytest.fixture
def release_repo(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True, stderr=subprocess.DEVNULL).strip()

    git("init", "-b", "master")
    git("config", "user.email", "tests@example.invalid")
    git("config", "user.name", "Release tests")
    (tmp_path / "src/intermine314").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.2.0"\n')
    (tmp_path / "src/intermine314/_version.py").write_text('VERSION="0.2.0"\n__version__=VERSION\n')
    git("add", ".")
    git("commit", "-m", "candidate")
    sha = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/master", sha)
    return tmp_path, sha, git


def test_matching_tag_on_master_allows_publication(release_repo):
    root, sha, _ = release_repo
    assert release_gate.release_context(root, event="push", ref="refs/tags/v0.2.0", sha=sha)["publish"]


@pytest.mark.parametrize("ref", ["refs/heads/master", "refs/tags/v0.2.0", "refs/tags/v9.9.9"])
def test_manual_runs_only_validate(release_repo, ref):
    root, sha, _ = release_repo
    assert not release_gate.release_context(root, event="workflow_dispatch", ref=ref, sha=sha)["publish"]


@pytest.mark.parametrize("event,ref", [("push", "refs/tags/v0.1.8"), ("push", "refs/heads/master"),
                                       ("pull_request", "refs/tags/v0.2.0"), ("push", "refs/tags/0.2.0")])
def test_non_release_events_and_wrong_tags_are_rejected(release_repo, event, ref):
    root, sha, _ = release_repo
    with pytest.raises(RuntimeError, match="pushed tag matching"):
        release_gate.release_context(root, event=event, ref=ref, sha=sha)


def test_wrong_checkout_sha_is_rejected(release_repo):
    root, _, _ = release_repo
    with pytest.raises(RuntimeError, match="triggering commit"):
        release_gate.release_context(root, event="push", ref="refs/tags/v0.2.0", sha="0" * 40)


def test_tag_outside_master_is_rejected(release_repo):
    root, _, git = release_repo
    git("checkout", "-b", "unmerged")
    (root / "extra").write_text("unmerged change")
    git("add", ".")
    git("commit", "-m", "unmerged")
    with pytest.raises(RuntimeError, match="origin/master"):
        release_gate.release_context(root, event="push", ref="refs/tags/v0.2.0", sha=git("rev-parse", "HEAD"))


@pytest.fixture
def release_artifacts(release_repo):
    root, sha, _ = release_repo
    for name in ("LICENSE", "LICENSE-BSD", "NOTICE"):
        (root / name).write_text(name + " license\n")
    wheel, sdist = artifacts(root, "0.2.0")
    directory = root / "release-artifacts"
    (directory / "dist").mkdir(parents=True)
    for path in (wheel, sdist):
        shutil.copy2(path, directory / "dist" / path.name)
    manifest = {"schema_version": 1, "status": "ok", "version": "0.2.0", "source_revision": sha,
                "source_dirty": False, "docs_verified": True,
                "modes": ["base", "plots", "analytics", "sdist", "speed", "combined"],
                "artifacts": [{"file": "dist/" + p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in (wheel, sdist)]}
    (directory / "artifact-manifest.json").write_text(json.dumps(manifest))
    return root, sha, directory, manifest


def test_exact_verified_artifact_set_passes(release_artifacts):
    root, sha, directory, _ = release_artifacts
    assert release_gate.verify_artifacts(root, directory, sha=sha)["status"] == "ok"


@pytest.mark.parametrize("damage", ["changed", "missing", "extra", "version", "sha", "dirty", "incomplete", "modes", "docs", "path", "checkout"])
def test_unverified_or_changed_artifacts_are_rejected(release_artifacts, damage):
    root, sha, directory, manifest = release_artifacts
    path = directory / manifest["artifacts"][0]["file"]
    if damage == "changed":
        path.write_bytes(b"altered after validation")
    elif damage == "missing":
        path.unlink()
    elif damage == "extra":
        (directory / "dist/extra.whl").write_bytes(b"unexpected")
    elif damage in {"version", "sha", "dirty", "incomplete"}:
        key, value = {"version": ("version", "9.9.9"), "sha": ("source_revision", "0" * 40),
                      "dirty": ("source_dirty", True), "incomplete": ("status", "incomplete")}[damage]
        manifest[key] = value
    elif damage == "modes":
        manifest["modes"] = ["base"]
    elif damage == "docs":
        manifest["docs_verified"] = False
    elif damage == "path":
        manifest["artifacts"][0]["file"] = "dist/../" + path.name
    elif damage == "checkout":
        source = root / "src/intermine314/_version.py"
        source.write_text(source.read_text() + "# modified after checkout\n")
    (directory / "artifact-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        release_gate.verify_artifacts(root, directory, sha=sha)


@pytest.mark.parametrize("value", ["", None, "0", "-1", "../123", "１２３", 123])
def test_empty_or_invalid_artifact_ids_are_rejected(value):
    with pytest.raises(RuntimeError, match="immutable artifact ID"):
        release_gate.require_artifact_id(value)


@pytest.mark.parametrize("state", ["missing", "invalid"])
def test_missing_or_invalid_manifest_is_rejected(release_artifacts, state):
    root, sha, directory, _ = release_artifacts
    path = directory / "artifact-manifest.json"
    if state == "missing":
        path.unlink()
    else:
        path.write_text("incomplete JSON")
    with pytest.raises(RuntimeError, match="manifest is missing or invalid"):
        release_gate.verify_artifacts(root, directory, sha=sha)
