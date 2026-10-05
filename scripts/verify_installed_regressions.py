"""Run release regressions using a wheel installation outside the checkout.

Install pytest in the wheel's virtual environment, then run this script with
that environment's Python -I from a directory outside the source checkout.
"""

import argparse
import json
import sys
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "test_transport_retry_policy.py", "test_json_precision.py", "test_legacy_module_facades.py",
    "test_distribution_versions.py", "test_release_gate.py", "test_parallel_export_status.py",
    "test_compressed_metadata_http.py", "test_parallel_memory_bounds.py",
    "test_parallel_cleanup.py", "test_parallel_page_validation.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    if Path.cwd().is_relative_to(ROOT):
        raise RuntimeError("Run installed regressions outside the checkout")
    import intermine314

    package = Path(intermine314.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()) or package.is_relative_to(ROOT):
        raise RuntimeError("Regression package must be an installed wheel")
    import pytest

    code = pytest.main(["-c", str(ROOT / "pyproject.toml"), "-o", "pythonpath=",
                        "--junitxml=" + str(args.junit), *[str(ROOT / "tests" / name) for name in TESTS]])
    modules = {name: module.__file__ for name, module in sys.modules.items()
               if name.startswith("intermine314") and getattr(module, "__file__", None)}
    if not all(Path(path).resolve().is_relative_to(Path(sys.prefix).resolve()) for path in modules.values()):
        raise RuntimeError("Source package leaked into the regression process")
    suite = ElementTree.parse(args.junit).getroot()[0]
    report = {"status": "ok" if code == 0 else "failed", "package_version": intermine314.VERSION,
              "python": sys.version, "package_file": str(package), "installed_modules": modules,
              "test_files": list(TESTS), "junit": suite.attrib,
              "scope": "Main regression process uses installed modules. Cold-import subprocesses explicitly inspect checkout facades. Tooling tests use controlled source metadata and temporary Git repositories. No live mutations."}
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
