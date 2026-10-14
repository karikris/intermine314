"""Fetch the exact audited PyPI source archive, verify it and extract safely."""

import hashlib
import json
import tarfile
from pathlib import Path
from urllib.request import urlopen

DESTINATION = Path("/tmp/intermine-v017-review")
ARCHIVE = "intermine314-0.1.7.tar.gz"
SHA256 = "18974b4013016ac524cf6e784ee565e5cee19022c63c8349453582b29c9e2ee6"


def main():
    with urlopen(
        "https://pypi.org/pypi/intermine314/0.1.7/json", timeout=30
    ) as response:
        metadata = json.load(response)
    artifact = next(row for row in metadata["urls"] if row["filename"] == ARCHIVE)
    if artifact["digests"]["sha256"] != SHA256:
        raise RuntimeError("PyPI artifact differs from the audited baseline")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = DESTINATION / ARCHIVE
    if not path.exists():
        with urlopen(artifact["url"], timeout=30) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise RuntimeError("Downloaded baseline failed SHA-256 verification")
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("Existing baseline archive failed SHA-256 verification")
    with tarfile.open(path) as archive:
        archive.extractall(DESTINATION / "sdist", filter="data")
    print(DESTINATION / "sdist/intermine314-0.1.7")


if __name__ == "__main__":
    main()
