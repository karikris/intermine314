from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _run_probe(script: str):
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=env,
    )
    return json.loads(completed.stdout)


def test_runtime_imports_do_not_pull_benchmark_modules():
    loaded_benchmarks = _run_probe(
        "import importlib, json, sys; "
        "importlib.import_module('intermine314'); "
        "importlib.import_module('intermine314.query.builder'); "
        "importlib.import_module('intermine314.service.transport'); "
        "mods=[n for n in sys.modules if n=='benchmarks' or n.startswith('benchmarks.')]; "
        "print(json.dumps(mods))"
    )
    assert loaded_benchmarks == []


def test_removed_legacy_root_alias_modules_are_not_available():
    loaded = _run_probe(
        "import importlib, json; "
        "importlib.import_module('intermine314'); "
        "names=('intermine314.query_manager','intermine314.query_export','intermine314.query_parallel'); "
        "state={}; "
        "\nfor name in names:\n"
        "  \n  try:\n"
        "    importlib.import_module(name)\n"
        "    state[name]=True\n"
        "  except ModuleNotFoundError:\n"
        "    state[name]=False\n"
        "print(json.dumps(state, sort_keys=True))"
    )
    assert loaded == {
        "intermine314.query_export": False,
        "intermine314.query_manager": False,
        "intermine314.query_parallel": False,
    }
