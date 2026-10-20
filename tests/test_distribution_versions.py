"""Future release versions and malformed artifacts exercise real verifier paths."""

import tomllib
from pathlib import Path

import pytest

from scripts import verify_distribution as distribution
from scripts.release_metadata import project_version
from tests.support.artifacts import artifacts


def test_minimum_runtime_matrix_matches_declared_requirements():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    requirements = list(project['dependencies'])
    for extra in ('plots', 'speed', 'proxy', 'pandas'):
        requirements.extend(project['optional-dependencies'][extra])
    expected = {entry.split(',')[0].replace('>=', '==').lower() for entry in requirements}
    actual = {line.lower() for line in (root / 'benchmarks/contracts/minimum-dependencies.txt').read_text().splitlines()
              if line and not line.startswith('#')}
    assert actual == expected


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "src/intermine314").mkdir(parents=True)
    for license in ("LICENSE", "LICENSE-BSD", "NOTICE"):
        (root / license).write_text(license + " license\n")
    monkeypatch.setattr(distribution, "ROOT", root)
    return root




@pytest.mark.parametrize("version", ["0.1.8", "0.2.0", "0.2.0rc1"])
def test_archive_versions_are_derived(candidate, version):
    wheel, sdist = artifacts(candidate, version)
    assert distribution.inspect_artifacts(wheel, sdist)["version"] == version


def test_project_runtime_mismatch_rejected_before_build(candidate):
    artifacts(candidate, "0.2.0")
    (candidate / "src/intermine314/_version.py").write_text('VERSION="0.1.8"\n__version__=VERSION\n')
    with pytest.raises(RuntimeError, match="Project/runtime versions differ"):
        project_version(candidate)
    with pytest.raises(RuntimeError, match="Project/runtime versions differ"):
        distribution.verify(candidate.parent / "output", ["base"], False)
    assert not (candidate.parent / "output").exists()


def test_distribution_metadata_mismatch_rejected(candidate):
    wheel, sdist = artifacts(candidate, "0.2.0", metadata_version="0.1.8")
    with pytest.raises(RuntimeError, match="Distribution version differs"):
        distribution.inspect_artifacts(wheel, sdist)


@pytest.mark.parametrize("state", ["missing", "duplicate"])
def test_missing_or_duplicate_sdist_metadata_rejected(candidate, state):
    wheel, sdist = artifacts(candidate, "0.2.0", pkg_info=state)
    with pytest.raises(RuntimeError, match="exactly one top-level PKG-INFO"):
        distribution.inspect_artifacts(wheel, sdist)
