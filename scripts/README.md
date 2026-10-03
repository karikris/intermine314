# Scripts

- `analytics_smoke.py`: offline explicit CSV input → Parquet → DuckDB SQL → Arrow → Polars check; run `python -m scripts.analytics_smoke` from the checkout.
- `test_inventory_summary.py`: inspect test inventory.
- `verify_distribution.py`: build a fresh wheel and sdist, verify licensing and
  Python metadata, install independent base/plots/analytics/sdist environments,
  run offline installed-package checks outside the checkout, and build Sphinx
  documentation against the base wheel. Run
  `python scripts/verify_distribution.py --output /tmp/intermine314-install-check`
  with Python 3.14.5+ and an empty output directory. Installation requires
  package-index access; runtime and docs checks use offline inputs. Evidence,
  artifact SHA-256 hashes, import locations, logs, distributions, and HTML docs
  remain in the output directory. Each invocation rebuilds the current tree;
  rerun after implementation changes using another empty directory.
- `ci/`: CI service initialization helpers.

Use an installed or editable checkout on Python 3.14.5+. The root Makefile and tox
analytics environment call the repository smoke script. These development scripts
are excluded from package distributions. Live samples run with `python -m samples.alleles`
or `python -m samples.polars_parquet_duckdb` and close their Service and DuckDB resources.
