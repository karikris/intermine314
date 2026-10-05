"""Build fresh distributions and verify independent installs outside the checkout.

Run with Python >=3.14.5. Package-index access is needed only for installation;
all runtime checks and the Sphinx build use local inputs and offline fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
import runpy
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from email.parser import BytesParser
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
project_version = runpy.run_path(str(ROOT / "scripts/release_metadata.py"))["project_version"]
MODULES = (
    "webservice", "registry", "query_manager", "bar_chart", "model", "results",
    "constraints", "pathfeatures", "lists.list", "lists.listmanager", "idresolution",
    "query.template", "query.executor", "query.parallel_runtime", "service.session",
)
HEAVY = {"polars", "duckdb", "pyarrow", "numpy", "matplotlib", "pandas", "intermine"}


def require(condition, message):
    # These checks must remain active when Python runs with optimization enabled.
    if not condition:
        raise RuntimeError(message)


def smoke(mode, fixture_dir):
    """Exercise the installed package; never insert checkout paths into sys.path."""
    require(not Path.cwd().is_relative_to(ROOT), "Smoke must run outside the checkout")
    require(not any(Path(p).resolve() == ROOT / "src" for p in sys.path), "Source path leaked")
    for name in ("intermine", "pandas") + (() if mode == "plots" else ("matplotlib",)):
        require(importlib.util.find_spec(name) is None, f"Unexpected dependency: {name}")

    class BlockHeavy(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in HEAVY:
                raise RuntimeError(f"Eager heavy import: {fullname}")

    blocker = BlockHeavy()
    sys.meta_path.insert(0, blocker)
    import intermine314
    import intermine314.webservice

    require(intermine314.VERSION == intermine314.__version__ == importlib.metadata.version("intermine314"),
            "Installed runtime/metadata versions differ")

    require("intermine314.service.service" not in sys.modules, "Facade is not lazy")
    from intermine314.query import Query, Template
    from intermine314.service import Registry as NativeRegistry
    from intermine314.service import Service as NativeService
    from intermine314.webservice import Registry, Service

    for name in MODULES:
        importlib.import_module("intermine314." + name)
    require(issubclass(Service, NativeService), "Service runtime differs")
    require(issubclass(Registry, NativeRegistry), "Registry runtime differs")
    require(issubclass(Template, Query), "Template query runtime differs")
    require(not HEAVY.intersection(sys.modules), "Analytics imported before use")
    sys.meta_path.remove(blocker)

    package_path = Path(intermine314.__file__).resolve()
    require(package_path.is_relative_to(Path(sys.prefix).resolve()), "Package is not installed in this venv")
    require(not package_path.is_relative_to(ROOT), "Checkout package imported")
    # Reuse only transport helpers and data, never source-package imports.
    fixtures = runpy.run_path(str(fixture_dir / "__init__.py"))
    fixture_session = fixtures["FixtureSession"]
    fixture_bytes = fixtures["fixture_bytes"]
    rows = b'{"results":[\n["Alice",42,true],\n["Bob",21,false]\n],"wasSuccessful":true}\n'
    expected = [
        {"Employee.name": "Alice", "Employee.age": 42, "Employee.fullTime": True},
        {"Employee.name": "Bob", "Employee.age": 21, "Employee.fullTime": False},
    ]
    import duckdb
    import polars as pl
    import pyarrow as pa

    with tempfile.TemporaryDirectory(prefix="installed-runtime-") as temporary:
        work = Path(temporary)
        for cls, profile in ((NativeService, "native"), (Service, "legacy")):
            session = fixture_session.service(version=30, rows=rows)
            with cls(fixtures["SERVICE_ROOT"], session=session) as service:
                require(service.compatibility == profile, "Wrong facade default")
                query = service.select("Employee.name", "Employee.age", "Employee.fullTime")
                require(type(query) is Query and query.service is service, "Query runtime differs")
                require(list(query.results("dict")) == expected, "Installed query results differ")
                target = work / f"{profile}.parquet"
                query.export(target, size=2, batch_size=2)
                require(pl.read_parquet(target).to_dicts() == expected, "Installed export differs")
                require(query.dataframe(size=2).to_dicts() == expected, "Installed dataframe differs")
            require(session.close_calls == 0, "Borrowed session closed")
            require(all(response.closed for response in session.responses), "Response leaked")
        owned = fixture_session.service(version=30)
        with patch("intermine314.service.session.build_session", return_value=owned):
            with NativeService(fixtures["SERVICE_ROOT"]) as service:
                require(service._owns_session, "Owned-session runtime differs")
            service.close()
        require(owned.close_calls == 1, "Owned session was not closed exactly once")

        from intermine314.export import import_csv, query_parquet

        csv = "id,code\n9007199254740993,0007\n9007199254740994,0008\n"
        csv_options = {"schema_overrides": {"id": pl.Int64, "code": pl.String}}
        source = StringIO(csv)
        target = work / "input.parquet"
        import_csv(source, target, csv_options=csv_options)
        require(not source.closed, "Borrowed CSV stream closed")
        result = query_parquet(target, "SELECT code FROM results WHERE id = ?", parameters=[2**53 + 1])
        require(isinstance(result, pl.DataFrame) and result.to_dicts() == [{"code": "0007"}], "SQL/Arrow/Polars differs")
        query = Query()
        require(query.dataframe(csv_input=StringIO(csv), csv_options=csv_options).height == 2, "CSV dataframe differs")
        managed = query.to_duckdb(
            work / "csv-parts", csv_input=StringIO(csv), csv_options=csv_options, managed=True,
        )
        with managed as connection:
            arrow = connection.execute("SELECT * FROM results ORDER BY id").to_arrow_table()
            require(isinstance(arrow, pa.Table), "DuckDB did not return Arrow")
            require(pl.from_arrow(arrow).to_dicts() == [
                {"id": 2**53 + 1, "code": "0007"}, {"id": 2**53 + 2, "code": "0008"},
            ], "CSV managed SQL differs")
        try:
            connection.execute("SELECT 1")
        except duckdb.ConnectionException:
            pass
        else:
            raise RuntimeError("Managed DuckDB connection leaked")
        require(not list(work.rglob("*.csv")), "Implicit CSV output created")

        from intermine314 import bar_chart

        if mode == "plots":
            import matplotlib

            matplotlib.use("Agg", force=True)
            from matplotlib import pyplot

            session = fixture_session({
                ("GET", "/service/instances"): fixture_bytes("registry-helpers.json"),
                ("GET", "/service/instances/OfflineMine"): fixture_bytes("registry-helpers.json"),
                ("GET", "/custom/service/version/ws"): b"30",
                ("GET", "/service/version/ws"): b"30",
                ("GET", "/custom/service/user/queries"): fixture_bytes("saved-queries.json"),
                ("GET", "/custom/service/query/results"): b"gene\ta\t1\ngene\tb\t2\n",
            })
            shown = []
            try:
                with patch.object(pyplot, "show", side_effect=lambda: shown.append(True)):
                    require(bar_chart.save_mine_and_token("OfflineMine", "offline", session=session) is None, "Plot configuration failed")
                    require(bar_chart.query_to_barchart_log('<query view="Gene.symbol Gene.label Gene.value"/>', "true") is None, "Plot return differs")
                    ax = pyplot.gcf().axes[0]
                    require([bar.get_height() for bar in ax.patches] == [0, 0.69], "Plot values differ")
                    require(ax.get_xlabel() == "Gene.label" and ax.get_ylabel() == "log(Gene.value)", "Plot labels differ")
                    require(shown == [True], "Plot was not shown")
                    pyplot.gcf().canvas.draw()
            finally:
                if bar_chart._state is not None:
                    bar_chart._state.close()
                pyplot.close("all")
            require(session.close_calls == 0 and all(r.closed for r in session.responses), "Plot ownership differs")
        else:
            try:
                bar_chart.plot_go_vs_p("offline")
            except ImportError as error:
                require("intermine314[plots]" in str(error), "Optional plots error is not actionable")
            else:
                raise RuntimeError("Missing Matplotlib was silently accepted")

    installed_modules = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if name.startswith("intermine314") and getattr(module, "__file__", None)
    }
    require(all(Path(path).is_relative_to(Path(sys.prefix).resolve()) for path in installed_modules.values()), "Source module leaked")
    require(importlib.util.find_spec("intermine") is None and importlib.util.find_spec("pandas") is None, "Forbidden dependency appeared")
    return {
        "mode": mode, "package_file": str(package_path), "package_version": intermine314.VERSION, "modules": installed_modules,
        "python": sys.version, "dependencies": {name: importlib.metadata.version(name) for name in ("requests", "urllib3", "polars", "duckdb", "pyarrow")},
        "absent": ["intermine", "pandas"] + ([] if mode == "plots" else ["matplotlib"]),
        "plot_rendered": mode == "plots", "status": "ok",
    }


def inspect_artifacts(wheel, sdist):
    expected_version = project_version(ROOT)
    licenses = ("LICENSE", "LICENSE-BSD", "NOTICE")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        entries = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(len(entries) == 1, "Wheel must contain exactly one METADATA entry")
        metadata = BytesParser().parsebytes(archive.read(entries[0]))
        for name in licenses:
            entry = next(n for n in names if n.endswith(".dist-info/licenses/" + name))
            require(archive.read(entry) == (ROOT / name).read_bytes(), f"Wheel license differs: {name}")
        for module in MODULES:
            require("intermine314/" + module.replace(".", "/") + ".py" in names, f"Missing wheel module: {module}")
        require("intermine314/config/runtime-defaults.toml" in names, "Runtime defaults missing")
    with tarfile.open(sdist, "r:gz") as archive:
        entries = [m for m in archive.getmembers() if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")]
        require(len(entries) == 1 and entries[0].isfile(), "Sdist must contain exactly one top-level PKG-INFO")
        root = entries[0].name.rsplit("/", 1)[0]
        require(root == sdist.name.removesuffix(".tar.gz"), "Sdist root differs from artifact filename")
        require(all(m.name == root or m.name.startswith(root + "/") for m in archive.getmembers()), "Sdist has multiple roots")
        for name in licenses:
            matches = [m for m in archive.getmembers() if m.name == f"{root}/{name}"]
            require(len(matches) == 1 and matches[0].isfile(), f"Missing or duplicate sdist license: {name}")
            member = matches[0]
            require(archive.extractfile(member).read() == (ROOT / name).read_bytes(), f"Sdist license differs: {name}")
        sdist_metadata = BytesParser().parsebytes(archive.extractfile(entries[0]).read())
    for value in (metadata, sdist_metadata):
        require(value["Name"] == "intermine314", "Distribution name differs")
        require(value["Version"] == expected_version, "Distribution version differs")
        require(value["Requires-Python"] == ">=3.14.5", "Python security floor differs")
        require(value["License-Expression"] == "MIT AND BSD-2-Clause", "License metadata differs")
        require(set(value.get_all("License-File")) == set(licenses), "License-file metadata differs")
        require("analytics" in value.get_all("Provides-Extra"), "Analytics alias missing")
        requirements = value.get_all("Requires-Dist")
        require(not any(r.lower().startswith(("pandas", "intermine ", "intermine>", "intermine=")) for r in requirements), "Forbidden runtime requirement")
        require(all("extra ==" in r for r in requirements if r.lower().startswith("matplotlib")), "Matplotlib is not optional")
    return {
        "artifacts": [{"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (wheel, sdist)],
        "version": metadata["Version"], "requires_python": metadata["Requires-Python"],
        "license_expression": metadata["License-Expression"], "license_files": metadata.get_all("License-File"),
        "requires_dist": metadata.get_all("Requires-Dist"),
    }


def verify(output, modes, docs):
    require(sys.version_info >= (3, 14, 5), "Python >=3.14.5 is required")
    expected_version = project_version(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    require(not output.is_relative_to(ROOT), "Verification output must be outside the checkout")
    require(not any(output.iterdir()), "Use an empty output directory for fresh builds")
    environment = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    environment.update(MPLBACKEND="Agg", PIP_DISABLE_PIP_VERSION_CHECK="1")

    def run(arguments, name):
        result = subprocess.run([str(a) for a in arguments], cwd=output, env=environment, capture_output=True, text=True)
        (output / f"{name}.log").write_text(result.stdout + result.stderr)
        require(result.returncode == 0, f"{name} failed; see {output / (name + '.log')}")
        return result.stdout

    def make_env(name):
        directory = output / name
        venv.EnvBuilder(with_pip=True).create(directory)
        return directory / "bin/python"

    builder = make_env("build-env")
    run([builder, "-I", "-m", "pip", "install", "build"], "build-dependencies")
    run([builder, "-I", "-m", "build", ROOT, "--outdir", output / "dist"], "build")
    wheel, = (output / "dist").glob("*.whl")
    sdist, = (output / "dist").glob("*.tar.gz")
    report = inspect_artifacts(wheel, sdist)
    report["installations"] = []
    for mode in modes:
        python = make_env(mode + "-env")
        artifact = sdist if mode == "sdist" else wheel
        extra = f"[{mode}]" if mode in ("plots", "analytics") else ""
        run([python, "-I", "-m", "pip", "install", str(artifact) + extra], f"install-{mode}")
        run([python, "-I", "-m", "pip", "check"], f"pip-check-{mode}")
        result = run([python, "-I", Path(__file__).resolve(), "--smoke", mode], f"smoke-{mode}")
        report["installations"].append(json.loads(result))
        require(report["installations"][-1]["package_version"] == expected_version, "Installed version differs")
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Verified {mode}: {report['installations'][-1]['package_file']}", flush=True)
    if docs:
        require("base" in modes, "Docs verification requires the base mode")
        python = output / "base-env/bin/python"
        run([python, "-I", "-m", "pip", "install", "sphinx"], "docs-dependencies")
        # Run conf.py in the actual Sphinx process and capture its import location.
        code = (
            "import json,runpy,intermine314; from sphinx.application import Sphinx; "
            f"conf=runpy.run_path({str(ROOT / 'docs/source/conf.py')!r}); "
            "print(json.dumps({'package_file':intermine314.__file__,'release':conf['release']})); "
            f"app=Sphinx({str(ROOT / 'docs/source')!r}, {str(ROOT / 'docs/source')!r}, "
            f"{str(output / 'docs')!r}, {str(output / 'doctrees')!r}, 'html', "
            "confoverrides={'intersphinx_mapping':{}}, warningiserror=True, freshenv=True); "
            "app.build(force_all=True); raise SystemExit(app.statuscode)"
        )
        result = run([python, "-I", "-c", code], "docs")
        report["docs"] = json.loads(result.splitlines()[0])
        require(report["docs"]["package_file"] == report["installations"][modes.index("base")]["package_file"], "Docs used a different package")
        require(report["docs"]["release"] == expected_version, "Docs version differs")
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Evidence: {output / 'verification.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Empty directory outside checkout; default: new temporary directory")
    parser.add_argument("--modes", nargs="+", choices=("base", "plots", "analytics", "sdist"), default=["base", "plots", "analytics", "sdist"])
    parser.add_argument("--skip-docs", action="store_true")
    parser.add_argument("--smoke", choices=("base", "plots", "analytics", "sdist"), help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.smoke:
        print(json.dumps(smoke(arguments.smoke, ROOT / "tests/fixtures/compatibility")))
    else:
        output = arguments.output or Path(tempfile.mkdtemp(prefix="intermine314-distribution-"))
        verify(output.resolve(), arguments.modes, not arguments.skip_docs)


if __name__ == "__main__":
    main()
