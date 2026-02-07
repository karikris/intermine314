from __future__ import annotations

import logging
import os
import sys
import time
from glob import glob
from os.path import basename, join as pjoin, splitext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestLoader, TextTestRunner

from setuptools import Command, setup

ROOT = Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"
MIN_PYTHON = (3, 14)


def _warn_if_below_python_314() -> None:
    if sys.version_info < MIN_PYTHON:
        logging.warning(
            "intermine314 targets Python %d.%d+ (running %d.%d); "
            "some workflows may not match release runtime behavior.",
            MIN_PYTHON[0],
            MIN_PYTHON[1],
            sys.version_info.major,
            sys.version_info.minor,
        )


def _project_version() -> str:
    try:
        import tomllib

        with PYPROJECT.open("rb") as fh:
            data = tomllib.load(fh)
        return data["project"]["version"]
    except Exception:
        # Fallback for unusual local states where pyproject parsing fails.
        from intermine314 import VERSION

        return VERSION


_warn_if_below_python_314()

class TestCommand(Command):
    description = "Run unit tests"
    user_options = [("verbose", "v", "produce verbose output"), ("testmodule=", "t", "test module name")]
    boolean_options = ["verbose"]

    def initialize_options(self):
        self._dir = os.getcwd()
        self.test_prefix = "test_"
        self.verbose = 0
        self.testmodule = None

    def finalize_options(self):
        pass

    def run(self):
        """
        Finds all the tests modules in tests/, and runs them,
         exiting after they are all done
        """
        from tests.server import TestServer
        from tests.test_core import WebserviceTest

        if self.verbose >= 2:
            logging.basicConfig(level=logging.DEBUG)

        testfiles = []
        if self.testmodule is None:
            for t in glob(pjoin(self._dir, "tests", self.test_prefix + "*.py")):
                if not t.endswith("__init__.py"):
                    testfiles.append(".".join(["tests", splitext(basename(t))[0]]))
        else:
            testfiles.append(self.testmodule)

        server = TestServer(daemonise=True, silent=(self.verbose < 3))
        server.start()
        WebserviceTest.TEST_PORT = server.port

        self.announce("Waiting for test server to start on port " + str(server.port), level=2)
        time.sleep(1)

        self.announce("Test files:" + str(testfiles), level=2)
        tests = TestLoader().loadTestsFromNames(testfiles)
        t = TextTestRunner(verbosity=self.verbose)
        result = t.run(tests)
        failed, errored = map(len, (result.failures, result.errors))
        raise SystemExit(failed + errored)


class PrintVersion(Command):
    user_options = []

    def initialize_options(self):
        self.version = None

    def finalize_options(self):
        self.version = _project_version()

    def run(self):
        print(self.version)


class LiveTestCommand(TestCommand):
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.test_prefix = "live"


class AnalyticsCheckCommand(Command):
    description = "Run Polars/Parquet/DuckDB compatibility smoke check"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            import duckdb
            import polars as pl
        except Exception as exc:
            raise SystemExit(
                "analyticscheck requires optional dependencies. "
                "Install with: pip install 'intermine314[dataframe]'"
            ) from exc

        from intermine314.query import Query

        expected = ("dataframe", "to_parquet", "to_duckdb")
        missing = [name for name in expected if not hasattr(Query, name)]
        if missing:
            raise SystemExit("Missing Query analytics API methods: " + ", ".join(missing))

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            parquet_file = tmp_path / "smoke.parquet"
            pl.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]}).write_parquet(parquet_file)

            con = duckdb.connect(database=":memory:")
            got = con.execute("select sum(x) from read_parquet(?)", [str(parquet_file)]).fetchone()[0]
            if got != 6:
                raise SystemExit(f"Unexpected DuckDB/Parquet result: {got}")

        print("analyticscheck: ok")


class CleanCommand(Command):
    """
    Remove all build files and all compiled files
    =============================================

    Remove everything from build, including that
    directory, and all .pyc files
    """

    user_options = [("verbose", "v", "produce verbose output")]

    def initialize_options(self):
        self._files_to_delete = []
        self._dirs_to_delete = []

        for root, dirs, files in os.walk("."):
            for f in files:
                if f.endswith(".pyc"):
                    self._files_to_delete.append(pjoin(root, f))
        for target in ("build", "dist", "intermine314.egg-info"):
            for root, dirs, files in os.walk(pjoin(target)):
                for f in files:
                    self._files_to_delete.append(pjoin(root, f))
                for d in dirs:
                    self._dirs_to_delete.append(pjoin(root, d))
            self._dirs_to_delete.append(target)
        # reverse dir list to remove children before parents
        self._dirs_to_delete = list(reversed(self._dirs_to_delete))

        self.verbose = 0

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._files_to_delete:
            if self.dry_run:
                logging.info("Would have unlinked %s", clean_me)
            else:
                try:
                    self.announce("Deleting " + clean_me, level=2)
                    os.unlink(clean_me)
                except Exception:
                    logging.warning("Failed to delete file %s", clean_me)
        for clean_me in self._dirs_to_delete:
            if self.dry_run:
                logging.info("Would have rmdir'ed %s", clean_me)
            else:
                if os.path.exists(clean_me):
                    try:
                        self.announce("Going to remove " + clean_me, level=2)
                        os.rmdir(clean_me)
                    except Exception:
                        logging.warning("Failed to delete dir %s", clean_me)
                elif clean_me != "build":
                    logging.warning("%s does not exist", clean_me)


setup(
    packages=["intermine314", "intermine314.lists"],
    cmdclass={
        "clean": CleanCommand,
        "test": TestCommand,
        "livetest": LiveTestCommand,
        "version": PrintVersion,
        "analyticscheck": AnalyticsCheckCommand,
    },
)
