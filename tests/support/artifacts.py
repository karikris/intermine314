"""Synthetic distribution artifacts for policy failures."""
import io
import tarfile
import zipfile

from scripts import verify_distribution as distribution


def artifacts(root, version, *, metadata_version=None, pkg_info="normal"):
    (root / "pyproject.toml").write_text(f'[project]\nversion="{version}"\n')
    (root / "src/intermine314/_version.py").write_text(f'VERSION="{version}"\n__version__=VERSION\n')
    metadata = (f"Metadata-Version: 2.4\nName: intermine314\nVersion: {metadata_version or version}\n"
                "Requires-Python: >=3.14.5\nLicense-Expression: MIT AND BSD-2-Clause\n"
                "License-File: LICENSE\nLicense-File: LICENSE-BSD\nLicense-File: NOTICE\n"
                'Provides-Extra: analytics\nProvides-Extra: pandas\nRequires-Dist: requests>=2.34.2\n'
                'Requires-Dist: pandas<4,>=3.0.6; extra == "pandas"\n\n').encode()
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

