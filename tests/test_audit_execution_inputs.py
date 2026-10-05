"""Publication must reject stale execution inputs before replacing artifacts."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import audit_api_coverage as audit


@pytest.fixture
def audit_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    analysis = tmp_path / "docs/analysis"
    analysis.mkdir(parents=True)
    monkeypatch.setattr(audit, "ANALYSIS", analysis)
    inputs = {
        "src/intermine314/example.py": "value = 1\n",
        "src/intermine314/config/runtime-defaults.toml": "value = 1\n",
        "tests/conftest.py": "fixture = 1\n",
        "tests/fixtures/compatibility/__init__.py": "helper = 1\n",
        "tests/fixtures/compatibility/model.xml": "<model/>\n",
        "tests/fixtures/compatibility/rows.json": "[]\n",
        "scripts/audit_api_coverage.py": "instrumentation = 1\n",
        "pyproject.toml": '[project]\nversion = "7.8.9"\n[tool.pytest.ini_options]\n',
        "benchmarks/policy_loader.py": "policy = 1\n",
        "benchmarks/profiles/mines.toml": "mines = []\n",
        "samples/common.py": "sample = 1\n",
        "docs/source/conf.py": "project = 'example'\n",
        ".github/workflows/im-build.yml": "name: CI\n",
        "docs/analysis/legacy-reexport-contract.json": '{}\n',
    }
    for name, content in inputs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (analysis / "api-audit-scopes.json").write_text(json.dumps({"contracts": {}, "departures": []}))
    (analysis / "implementation-ledger.json").write_text(json.dumps({"upstream": {}}))
    (analysis / "intermine-api-inventory.parquet").write_bytes(b"historical baseline")
    rows = [{"expected_intermine314_symbol": f"example.symbol_{i}", "upstream_symbol": f"upstream.symbol_{i}",
             "kind": "function", "owner_task": "fixture", "upstream_signature": "()", "upstream_source": "fixture"}
            for i in range(460)]
    (analysis / "implementation-ledger.json").write_text(json.dumps({"upstream": {}, "symbol_assignments": rows}))
    # Construct a pre-execution snapshot independently, so old instrumentation
    # demonstrates the defect by publishing after a fixture/config change.
    manifest = {"schema_version": 1, "scope": audit.EXECUTION_INPUT_POLICY, "files": {
        name: hashlib.sha256(content.encode()).hexdigest() for name, content in sorted(inputs.items())
    }}
    trace = {"source_hashes": audit.source_hashes(), "tests": {}, "calls": {}, "outcomes": {},
             "metadata": {row["expected_intermine314_symbol"]: {} for row in rows},
             "source_base_commit": "fixture", "command": "fixture pytest", "pytest_exit": 0,
             "execution_inputs": manifest}
    return tmp_path, analysis, trace, rows


@pytest.mark.parametrize("name", [
    "tests/fixtures/compatibility/model.xml",
    "tests/fixtures/compatibility/rows.json",
    "tests/conftest.py",
    "tests/fixtures/compatibility/__init__.py",
    "src/intermine314/config/runtime-defaults.toml",
    "pyproject.toml",
    "scripts/audit_api_coverage.py",
    "benchmarks/policy_loader.py",
    "benchmarks/profiles/mines.toml",
    "samples/common.py",
    "docs/source/conf.py",
    ".github/workflows/im-build.yml",
    "docs/analysis/legacy-reexport-contract.json",
])
def test_changed_execution_input_blocks_publication(audit_checkout, name):
    root, analysis, trace, rows = audit_checkout
    outputs = [analysis / f"intermine-api-final-coverage.{suffix}" for suffix in ("json", "parquet")]
    for path in outputs:
        path.write_bytes(b"previous artifact")
    (root / name).write_text("changed after pytest\n")
    with pytest.raises(RuntimeError, match="Execution inputs changed"):
        audit.render(trace, rows)
    assert all(path.read_bytes() == b"previous artifact" for path in outputs)


@pytest.mark.parametrize("operation", ["add-fixture", "remove-fixture", "add-helper", "add-pytest-config"])
def test_execution_input_membership_blocks_new_artifacts(audit_checkout, operation):
    root, analysis, trace, rows = audit_checkout
    if operation == "remove-fixture":
        (root / "tests/fixtures/compatibility/model.xml").unlink()
    else:
        name = {"add-fixture": "tests/fixtures/compatibility/added.xml",
                "add-helper": "tests/fixtures/compatibility/added.py",
                "add-pytest-config": "pytest.ini"}[operation]
        (root / name).write_text("added after pytest\n")
    with pytest.raises(RuntimeError, match="Execution inputs changed"):
        audit.render(trace, rows)
    assert not (analysis / "intermine-api-final-coverage.json").exists()
    assert not (analysis / "intermine-api-final-coverage.parquet").exists()


def test_unchanged_execution_inputs_allow_repeat_render(audit_checkout):
    root, analysis, trace, rows = audit_checkout
    # Generated outputs/cache and curated render-time rules are deliberately not
    # execution inputs. Neither rerendering nor pytest bytecode invalidates them.
    cache = root / "tests/__pycache__"
    cache.mkdir()
    (cache / "conftest.cpython-314.pyc").write_bytes(b"cache")
    audit.render(trace, rows)
    first = (analysis / "intermine-api-final-coverage.json").read_bytes()
    published = json.loads(first)
    assert published["package_version"] == "7.8.9"
    assert published["execution_inputs"] == trace["execution_inputs"]
    assert published["audit_script_sha256"] == trace["execution_inputs"]["files"]["scripts/audit_api_coverage.py"]
    assert published["execution_inputs_sha256"] == hashlib.sha256(json.dumps(trace["execution_inputs"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    import polars as pl
    parquet = pl.read_parquet(analysis / "intermine-api-final-coverage.parquet")
    assert parquet["execution_inputs_sha256"].unique().to_list() == [published["execution_inputs_sha256"]]
    assert parquet["audit_script_sha256"].unique().to_list() == [published["audit_script_sha256"]]
    audit.render(trace, rows)
    assert (analysis / "intermine-api-final-coverage.json").read_bytes() == first
    assert (analysis / "intermine-api-final-coverage.parquet").is_file()


@pytest.mark.parametrize("state", ["missing", "old-schema", "stale-policy"])
def test_old_trace_inputs_rejected_before_publication(audit_checkout, state):
    _, analysis, trace, rows = audit_checkout
    if state == "missing":
        del trace["execution_inputs"]
    elif state == "old-schema":
        trace["execution_inputs"]["schema_version"] = 0
    else:
        trace["execution_inputs"]["scope"] = {"trees": ["tests"]}
    with pytest.raises(RuntimeError, match="Execution inputs manifest missing or stale"):
        audit.render(trace, rows)
    assert not (analysis / "intermine-api-final-coverage.json").exists()
    assert not (analysis / "intermine-api-final-coverage.parquet").exists()


@pytest.mark.parametrize("change_stage", ["api-initialization", "pytest", None])
def test_trace_captures_inputs_before_initialization_and_pytest(audit_checkout, monkeypatch, change_stage):
    root, _, expected, _ = audit_checkout
    trace_path = root / "trace.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--trace", str(trace_path)])
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *args, **kwargs: "fixture\n")

    def change_input():
        (root / "tests/fixtures/compatibility/model.xml").write_text("changed during run")

    def initialize(rows):
        if change_stage == "api-initialization":
            change_input()
        return SimpleNamespace(metadata={}, calls={}, outcomes={}, tests={})

    def run_pytest(args, *, plugins):
        if change_stage == "pytest":
            change_input()
        return 0

    monkeypatch.setattr(audit, "Audit", initialize)
    monkeypatch.setattr(audit.pytest, "main", run_pytest)
    if change_stage:
        with pytest.raises(RuntimeError, match="Execution inputs changed"):
            audit.main()
        assert not trace_path.exists()
    else:
        with pytest.raises(SystemExit) as exit_info:
            audit.main()
        assert exit_info.value.code == 0
        captured = json.loads(trace_path.read_text())
        assert captured["execution_inputs"] == expected["execution_inputs"]
        assert captured["execution_inputs"]["files"]["scripts/audit_api_coverage.py"] == hashlib.sha256(b"instrumentation = 1\n").hexdigest()
