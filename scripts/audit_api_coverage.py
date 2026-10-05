"""Record scoped pytest execution for every historical public API symbol.

This is evidence instrumentation, not an equivalence oracle. Python call tracing
cannot distinguish aliases sharing code or prove an assertion covers every branch.
The generated report keeps those limits and explicit assertion scopes visible.
Run as ``python -m scripts.audit_api_coverage`` from the checkout with
--trace /tmp/audit.json, then render with
--render /tmp/audit.json after reviewing docs/analysis/api-audit-scopes.json.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "docs/analysis"
EXECUTION_INPUT_POLICY = {
    "trees": ["src/intermine314", "tests", "scripts", "samples", "docs/source",
              "benchmarks/profiles", "benchmarks/contracts", ".github/workflows"],
    "globs": ["benchmarks/**/*.py"],
    "files": ["pyproject.toml", "pytest.ini", ".pytest.ini", "setup.cfg", "tox.ini", "conftest.py",
              "Makefile", "MANIFEST.in", "README.md", "BENCHMARK.md", "LICENSE", "LICENSE-BSD", "NOTICE",
              "benchmarks/asv.conf.json", "docs/analysis/legacy-reexport-contract.json"],
    "excluded_directories": ["__pycache__", ".pytest_cache", ".ruff_cache"],
    "excluded_suffixes": [".pyc", ".pyo"],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / "src/intermine314").rglob("*.py"))}


def execution_inputs():
    """Snapshot controlled owned inputs, including absent/present membership.

    Only explicit roots are traversed. Generated analysis/build outputs and the
    rest of the checkout are outside this scope. This binds owned test inputs,
    not the interpreter, installed dependencies or a live server environment.
    """
    policy = EXECUTION_INPUT_POLICY
    paths = {ROOT / name for name in policy["files"] if (ROOT / name).exists()}
    for name in policy["trees"]:
        paths.update((ROOT / name).rglob("*"))
    for pattern in policy["globs"]:
        paths.update(ROOT.glob(pattern))
    files = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        if any(part in policy["excluded_directories"] for part in relative.parts) or path.suffix in policy["excluded_suffixes"]:
            continue
        if path.is_symlink():
            raise RuntimeError(f"Execution inputs cannot contain symlinks: {relative}")
        if path.is_file():
            files[str(relative)] = digest(path)
    return {"schema_version": 1, "scope": policy, "files": files}


def verify_execution_inputs(recorded):
    """Fail before publication if execution used different bytes or membership."""
    if not recorded or recorded.get("schema_version") != 1 or recorded.get("scope") != EXECUTION_INPUT_POLICY:
        raise RuntimeError("Execution inputs manifest missing or stale; rerun the trace")
    current = execution_inputs()["files"]
    previous = recorded["files"]
    added = sorted(current.keys() - previous.keys())
    removed = sorted(previous.keys() - current.keys())
    changed = sorted(name for name in current.keys() & previous.keys() if current[name] != previous[name])
    if added or removed or changed:
        raise RuntimeError(f"Execution inputs changed after capture; rerun the trace: added={added}, removed={removed}, changed={changed}")


def resolve(name):
    parts = name.split(".")
    for index in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:index]))
            break
        except ModuleNotFoundError as error:
            if error.name != ".".join(parts[:index]):
                # A non-package parent also raises for the requested full name.
                if not ".".join(parts[:index]).startswith(error.name + "."):
                    raise
    parent = None
    for part in parts[index:]:
        parent = obj
        obj = getattr(obj, part) if inspect.ismodule(obj) else inspect.getattr_static(obj, part)
        if isinstance(obj, (classmethod, staticmethod)):
            obj = obj.__func__
    return parent, obj


def location(obj):
    try:
        path = Path(inspect.getsourcefile(obj)).resolve()
        return f"{path.relative_to(ROOT)}:{inspect.getsourcelines(obj)[1]}"
    except (TypeError, ValueError):
        return "Python builtin inherited implementation"


class Audit:
    def __init__(self, rows):
        self.mapping, self.metadata, self.calls = {}, {}, {}
        self.outcomes, self.tests = {}, {}
        self.current, self.phase = None, None
        for row in rows:
            name = row["expected_intermine314_symbol"]
            delegated = row["kind"] == "delegated method"
            lookup = "intermine314.lists.listmanager.ListManager." + name.rsplit(".", 1)[1] if delegated else name
            parent, obj = resolve(lookup)
            owner = parent if inspect.isclass(parent) else None
            target = obj.fget if isinstance(obj, property) else obj
            if inspect.isclass(obj):
                target, owner = obj.__init__, obj
            target = inspect.unwrap(target)
            try:
                signature = str(inspect.signature(obj.fget if isinstance(obj, property) else obj))
            except (TypeError, ValueError):
                signature = "not introspectable (builtin constructor)"
            defining_owner = next((base for base in parent.__mro__ if name.rsplit(".", 1)[1] in vars(base)), None) if inspect.isclass(parent) else None
            self.metadata[name] = {
                "name_status": "available-dynamic-instance" if delegated else "available",
                "signature": signature,
                "signature_scope": "unbound callable; property getter; delegated manager method before binding; class constructor",
                "signature_comparison": "same textual signature" if signature == row["upstream_signature"] else "different textual signature; supported calls are scoped by tests, not inferred from spelling",
                "binding": "Service instance __getattr__ -> cached ListManager bound method" if delegated else ("property" if isinstance(obj, property) else "class member" if owner and not inspect.isclass(obj) else "module export"),
                "defining_class": defining_owner.__module__ + "." + defining_owner.__qualname__ if defining_owner else None,
                "implementation": getattr(target, "__module__", "builtins") + "." + getattr(target, "__qualname__", type(target).__name__),
                "source": location(obj if inspect.isclass(obj) else target),
                "execution_source": location(target),
            }
            code = getattr(target, "__code__", None)
            if code:
                self.mapping.setdefault(code, []).append((name, owner))
        # Dynamic availability is actually bound on a configured concrete instance.
        from intermine314.webservice import Service
        from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession
        session = FixtureSession.service()
        with Service(SERVICE_ROOT, session=session) as service:
            before = len(session.requests)
            for row in rows:
                if row["kind"] == "delegated method":
                    name = row["expected_intermine314_symbol"]
                    bound = getattr(service, name.rsplit(".", 1)[1])
                    if not callable(bound) or bound.__self__ is not service._list_manager:
                        raise RuntimeError(f"Incorrect delegated binding: {name}")
                    self.metadata[name]["bound_signature"] = str(inspect.signature(bound))
            if len(session.requests) != before or session.close_calls:
                raise RuntimeError("Binding changed transport lifecycle")

    def profile(self, frame, event, arg):
        if event != "call" or self.current is None:
            return
        matches = self.mapping.get(frame.f_code)
        if not matches:
            return
        instance = frame.f_locals.get("self", frame.f_locals.get("cls"))
        for name, owner in matches:
            if owner is not None and instance is not None:
                if not (isinstance(instance, owner) or (inspect.isclass(instance) and issubclass(instance, owner))):
                    continue
            self.calls.setdefault(name, {}).setdefault(self.current, set()).add(self.phase)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        self.current, self.phase = item.nodeid, "setup"
        previous = sys.getprofile()
        sys.setprofile(self.profile)
        try:
            yield
        finally:
            sys.setprofile(previous)
            self.current = None

    def pytest_runtest_call(self, item):
        self.phase = "call"
        source = inspect.getsource(item.function)
        parsed = ast.parse(inspect.cleandoc(source) if source.startswith(" ") else source)
        assertions = [ast.unparse(node.test) for node in ast.walk(parsed) if isinstance(node, ast.Assert)]
        raises = [ast.unparse(node) for node in ast.walk(parsed) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "raises"]
        fixtures = {}
        for name, definition in item._request._fixture_defs.items():
            loc = location(definition.func)
            if loc.startswith("tests/"):
                fixtures[name] = loc
        self.tests[item.nodeid] = {
            "source": location(item.function), "assertions": assertions,
            "exception_assertions": raises, "fixtures": fixtures,
            "test_file_sha256": digest(Path(item.path)),
        }

    def pytest_runtest_teardown(self, item):
        self.phase = "teardown"

    def pytest_runtest_logreport(self, report):
        # A teardown failure must not be certified as passing.
        if report.failed or report.when == "call":
            self.outcomes[report.nodeid] = report.outcome


def render(trace, rows):
    verify_execution_inputs(trace.get("execution_inputs"))
    if trace["source_hashes"] != source_hashes():
        raise RuntimeError("Package source changed after execution; rerun the trace")
    for node, details in trace["tests"].items():
        if digest(ROOT / node.split("::")[0]) != details["test_file_sha256"]:
            raise RuntimeError(f"Test source changed after execution: {node}")
    rules = json.loads((ANALYSIS / "api-audit-scopes.json").read_text())
    result, catalog = [], {}
    for row in rows:
        name = row["expected_intermine314_symbol"]
        candidates = trace["calls"].get(name, {})
        contract = rules["contracts"].get(name)
        selectors = contract["tests"] if contract else []
        for prefix in selectors:
            if not any(node == prefix or node.startswith(prefix + "[") for node in trace["tests"]):
                raise RuntimeError(f"Unexecuted assertion override: {prefix}")
        # Only curated semantic assertions can justify a passing scope. Call
        # tracing is supplementary execution evidence, never an assertion oracle.
        # In particular, a random assertion in a caller does not test this symbol.
        selected = list(dict.fromkeys(node for prefix in selectors for node in trace["tests"]
                                      if node == prefix or node.startswith(prefix + "[")))
        if contract and contract.get("case_contains"):
            selected = [node for node in selected if contract["case_contains"] in node.split("[", 1)[-1]]
            if not selected:
                raise RuntimeError(f"No executed parameter case for {name}")
        evidence = []
        for node in selected:
            if trace["outcomes"].get(node) != "passed":
                raise RuntimeError(f"Unexecuted or failed assertion override: {node}")
            details = trace["tests"][node]
            if not details["assertions"] and not details["exception_assertions"]:
                continue
            catalog[node] = details
            evidence.append({
                "test": node, "outcome": "passed", "execution_phases": candidates.get(node, []),
                "scope": contract["scope"],
                "binding_limit": "Shared code does not distinguish alias spelling; availability is independently inspected.",
            })
        departures = [rule for rule in rules["departures"] if name in rule["symbols"]]
        for departure in departures:
            for prefix in departure["tests"]:
                matches = [node for node in trace["tests"] if (node == prefix or node.startswith(prefix + "[")) and trace["outcomes"].get(node) == "passed"]
                if not matches:
                    raise RuntimeError(f"Unexecuted departure test: {prefix}")
                # Keep all executed parameter cases for precise departure evidence.
                for node in matches:
                    catalog[node] = trace["tests"][node]
        result.append({
            "upstream_symbol": row["upstream_symbol"], "kind": row["kind"], "owner_task": row["owner_task"],
            "upstream_signature": row["upstream_signature"], "upstream_source": row["upstream_source"],
            "current_symbol": name, "current": trace["metadata"][name],
            "behavior_status": "tested-departure" if departures and evidence else "passing-scoped" if evidence else "unverified",
            "evidence": evidence, "deviations": [rule["id"] for rule in departures],
            "unverified": ["Exhaustive upstream equivalence, unexecuted call combinations and live-server behavior are not certified."] if evidence else ["No executed assertion scope was linked; name availability alone supplies no behavioral evidence."],
        })
    report = {
        "schema_version": 1, "package_version": tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"], "source_base_commit": trace["source_base_commit"], "upstream": json.loads((ANALYSIS / "implementation-ledger.json").read_text())["upstream"],
        "historical_inventory": {"path": "docs/analysis/intermine-api-inventory.parquet", "sha256": digest(ANALYSIS / "intermine-api-inventory.parquet"), "scope": "Unmodified historical 11-column baseline at 41c363e"},
        "policy": "Passing-scoped requires curated semantic assertions about the named symbol and successful execution of the linked nodes. Call tracing alone never promotes a row to passing. This is not blanket signature, branch or server equivalence. Tested departures link caller impact, rationale, recommendation and executed tests. Shared implementations and fixture phases remain explicit. Task and review completion are owned by the ledger.",
        "command": trace["command"],
        "scope_rules_sha256": digest(ANALYSIS / "api-audit-scopes.json"),
        "audit_script_sha256": trace["execution_inputs"]["files"]["scripts/audit_api_coverage.py"],
        "execution_inputs": trace["execution_inputs"],
        "execution_inputs_sha256": hashlib.sha256(json.dumps(trace["execution_inputs"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "pytest": {"exit_code": trace["pytest_exit"], "outcomes": dict(Counter(trace["outcomes"].values()))},
        "summary": dict(Counter(row["behavior_status"] for row in result)),
        "symbols": result, "test_catalog": catalog, "departures": rules["departures"],
        "source_hashes": trace["source_hashes"],
        "installation_evidence": "docs/analysis/remediation-distribution.json", "dependency_evidence": "docs/analysis/remediation-research.json",
        "remediation_validation": "docs/analysis/remediation-validation.json",
    }
    if trace["pytest_exit"] or len(result) != 460 or len({r["current_symbol"] for r in result}) != 460:
        raise RuntimeError("Final audit requires a passing suite and all 460 distinct rows")
    (ANALYSIS / "intermine-api-final-coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    import polars as pl
    flat = [{**{k: value for k, value in row.items() if k != "current"}, **{"current_" + k: value for k, value in row["current"].items()}} for row in result]
    # Explicit JSON columns retain nested evidence losslessly across Parquet tools.
    for row in flat:
        row["execution_inputs_sha256"] = report["execution_inputs_sha256"]
        row["audit_script_sha256"] = report["audit_script_sha256"]
        for key in ("evidence", "deviations", "unverified"):
            row[key] = json.dumps(row[key], sort_keys=True)
    pl.DataFrame(flat, infer_schema_length=None).write_parquet(ANALYSIS / "intermine-api-final-coverage.parquet")
    print(json.dumps({"rows": len(result), "summary": report["summary"], "tests": len(catalog)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trace", type=Path)
    group.add_argument("--render", type=Path)
    args = parser.parse_args()
    rows = json.loads((ANALYSIS / "implementation-ledger.json").read_text())["symbol_assignments"]
    if args.render:
        render(json.loads(args.render.read_text()), rows)
        return
    # Capture before importing API/fixture helpers or invoking pytest. Recheck
    # afterward so changes during the run cannot yield a publishable trace.
    inputs = execution_inputs()
    audit = Audit(rows)
    hashes = source_hashes()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    code = pytest.main(["-o", "addopts=--strict-markers -ra", "-q"], plugins=[audit])
    verify_execution_inputs(inputs)
    args.trace.write_text(json.dumps({
        "metadata": audit.metadata, "calls": {key: {node: sorted(phases) for node, phases in calls.items()} for key, calls in audit.calls.items()},
        "outcomes": audit.outcomes, "tests": audit.tests, "pytest_exit": int(code),
        "source_hashes": hashes, "source_base_commit": commit, "execution_inputs": inputs,
        "command": f".venv/bin/python -m scripts.audit_api_coverage --trace {args.trace}",
    }, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
