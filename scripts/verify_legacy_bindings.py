"""Verify supplemental historical bindings against the installed package."""

import argparse
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(contract):
    import intermine314

    results = []
    for binding in contract["bindings"]:
        module = importlib.import_module(binding["module"])
        target = importlib.import_module(binding["target"])
        expected = target if binding["kind"] == "module" else getattr(target, binding["name"])
        if (getattr(module, binding["name"]) is not expected
                or binding["name"] not in module.__all__ or binding["name"] not in dir(module)):
            raise RuntimeError("Historical binding differs: " + binding["module"] + "." + binding["name"])
        results.append(binding["module"] + "." + binding["name"])
    return {"package_version": intermine314.VERSION, "package_file": intermine314.__file__,
            "upstream": contract["upstream"], "scope": contract["scope"],
            "verified_bindings": results, "status": "ok"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT / "docs/analysis/legacy-reexport-contract.json")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = verify(json.loads(args.contract.read_text()))
    text = json.dumps(report, indent=2) + "\n"
    if args.json_out:
        args.json_out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
