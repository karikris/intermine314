PYTHON ?= python3.14
SETUP := $(PYTHON) setup.py
BENCHMARK_TARGET ?= maizemine
BENCHMARK_WORKERS ?= auto
BENCHMARK_PROFILE ?= auto

.PHONY: help test live-tests analyticscheck lint docs docs-clean benchmark benchmark-oakmine benchmark-thalemine benchmark-wheatmine benchmark-maizemine build dist-check clean-build clean

help:
	@echo "Targets:"
	@echo "  test              Run unit tests"
	@echo "  live-tests        Run live mine tests"
	@echo "  analyticscheck    Run Polars/Parquet/DuckDB smoke check"
	@echo "  lint              Run minimal lint (ruff E9 via pyproject)"
	@echo "  docs              Build Sphinx HTML docs"
	@echo "  docs-clean        Clean Sphinx build output"
	@echo "  benchmark         Run benchmark target (BENCHMARK_TARGET=<name>)"
	@echo "  benchmark-<mine>  Run benchmark target for maizemine/oakmine/thalemine/wheatmine"
	@echo "  build             Build wheel and sdist"
	@echo "  dist-check        Validate built distributions"
	@echo "  clean-build       Remove build artifacts"
	@echo "  clean             Remove pyc/cache/build artifacts and docs output"

test:
	$(PYTHON) -m pytest -q

live-tests:
	INTERMINE314_RUN_LIVE_TESTS=1 $(PYTHON) -m pytest -q tests

analyticscheck:
	$(SETUP) analyticscheck

lint:
	$(PYTHON) -m ruff check .

docs:
	$(MAKE) -C docs html

docs-clean:
	$(MAKE) -C docs clean

benchmark:
	$(PYTHON) benchmarks/runners/run_live.py --benchmark-target $(BENCHMARK_TARGET) --workers $(BENCHMARK_WORKERS) --benchmark-profile $(BENCHMARK_PROFILE)

benchmark-oakmine:
	$(MAKE) benchmark BENCHMARK_TARGET=oakmine

benchmark-thalemine:
	$(MAKE) benchmark BENCHMARK_TARGET=thalemine

benchmark-wheatmine:
	$(MAKE) benchmark BENCHMARK_TARGET=wheatmine

benchmark-maizemine:
	$(MAKE) benchmark BENCHMARK_TARGET=maizemine

build:
	$(PYTHON) -m build

dist-check:
	$(PYTHON) -m twine check dist/*

clean-build:
	rm -rf build dist *.egg-info

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	$(MAKE) docs-clean
	$(MAKE) clean-build
