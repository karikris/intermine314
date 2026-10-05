"""Future release versions and malformed artifacts exercise real verifier paths."""

import io
import tarfile
import zipfile

import pytest

from scripts import verify_distribution as distribution
from scripts.release_metadata import project_version


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "src/intermine314").mkdir(parents=True)
    for license in ("LICENSE", "LICENSE-BSD", "NOTICE"):
        (root / license).write_text(license + " license\n")
    monkeypatch.setattr(distribution, "ROOT", root)
    return root


def artifacts(root, version, *, metadata_version=None, pkg_info="normal"):
    (root / "pyproject.toml").write_text(f'[project]\nversion="{version}"\n')
    (root / "src/intermine314/_version.py").write_text(f'VERSION="{version}"\n__version__=VERSION\n')
    metadata = (f"Metadata-Version: 2.4\nName: intermine314\nVersion: {metadata_version or version}\n"
                "Requires-Python: >=3.14.5\nLicense-Expression: MIT AND BSD-2-Clause\n"
                "License-File: LICENSE\nLicense-File: LICENSE-BSD\nLicense-File: NOTICE\n"
                "Provides-Extra: analytics\nRequires-Dist: requests>=2.34.2\n\n").encode()
    wheel = root / f"intermine314-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"intermine314-{version}.dist-info/METADATA", metadata)
        for license in ("LICENSE", "LICENSE-BSD", "NOTICE"):
            archive.writestr(f"intermine314-{version}.dist-info/licenses/{license}", (root / license).read_bytes())
        for module in distribution.MODULES:
            archive.writestr("intermine314/" + module.replace(".", "/") + ".py", b"")
        archive.writestr("intermine314/config/runtime-defaults.toml", b"")
    sdist = root / f"intermine314-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        def add(name, body):
            entry = tarfile.TarInfo(f"intermine314-{version}/" + name)
            entry.size = len(body)
            archive.addfile(entry, io.BytesIO(body))

        for license in ("LICENSE", "LICENSE-BSD", "NOTICE"):
            add(license, (root / license).read_bytes())
        if pkg_info != "missing":
            add("PKG-INFO", metadata)
        if pkg_info == "duplicate":
            add("PKG-INFO", metadata)
    return wheel, sdist


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
