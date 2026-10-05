"""Exercise release decisions with real temporary Git history, never publish."""

import subprocess

import pytest

from scripts import release_gate


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
