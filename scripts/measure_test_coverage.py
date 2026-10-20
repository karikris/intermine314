"""Collect timings or per-case source coverage without editing the test suite."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["baseline", "coverage", "trial"])
    parser.add_argument(
        "--work", type=Path, default=Path("/tmp/intermine314-test-review")
    )
    parser.add_argument(
        "--coverage-tools",
        type=Path,
        default=Path("/tmp/intermine314-test-review-tools"),
    )
    args = parser.parse_args()
    mode, work = args.mode, args.work
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    cov = None
    if mode in {"coverage", "trial"}:
        os.environ["COVERAGE_CORE"] = "ctrace"
        sys.path.insert(0, str(args.coverage_tools))
        import coverage

        if coverage.__version__ != "7.16.2":
            raise RuntimeError("Analysis is pinned to coverage.py 7.16.2")
        cov = coverage.Coverage(
            branch=True,
            source=[str(ROOT / "src/intermine314")],
            data_file=str(
                work / (".coverage-trial" if mode == "trial" else ".coverage")
            ),
            concurrency=["thread"],
        )
        cov.start()
    import pytest

    class Probe:
        def pytest_collection_modifyitems(self, items, config):
            if mode == "trial":
                drops = set(json.loads((work / "deselect.json").read_text()))
                removed = [item for item in items if item.nodeid in drops]
                items[:] = [item for item in items if item.nodeid not in drops]
                config.hook.pytest_deselected(items=removed)
            rows = [
                dict(
                    nodeid=item.nodeid,
                    path=str(item.path.relative_to(ROOT)),
                    function=getattr(item, "originalname", None) or item.name,
                    params={
                        k: repr(v)
                        for k, v in getattr(
                            getattr(item, "callspec", None), "params", {}
                        ).items()
                    },
                    fixtures=item.fixturenames,
                    markers=[m.name for m in item.iter_markers()],
                )
                for item in items
            ]
            (work / f"{mode}-collection.json").write_text(json.dumps(rows))

        def pytest_runtest_setup(self, item):
            if cov:
                cov.switch_context(item.nodeid + "|setup")

        def pytest_runtest_call(self, item):
            if cov:
                cov.switch_context(item.nodeid + "|call")

        def pytest_runtest_teardown(self, item):
            if cov:
                cov.switch_context(item.nodeid + "|teardown")

    start = time.perf_counter()
    code = pytest.main(
        ["-q", "--junitxml=" + str(work / f"{mode}.xml"), "--durations=30"],
        plugins=[Probe()],
    )
    if cov:
        cov.stop()
        cov.save()
        cov.json_report(
            outfile=str(
                work / ("trial-coverage.json" if mode == "trial" else "coverage.json")
            )
        )
    (work / f"{mode}-result.json").write_text(
        json.dumps(
            dict(
                exit_code=int(code),
                seconds=time.perf_counter() - start,
                python=sys.version,
            )
        )
    )
    raise SystemExit(code)


if __name__ == "__main__":
    # Required: spawned transform workers import their parent's entry point.
    main()
