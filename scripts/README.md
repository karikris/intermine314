# Scripts

- `analytics_smoke.py`: offline explicit CSV input → Parquet → DuckDB SQL → Arrow → Polars check; run `python -m scripts.analytics_smoke` from the checkout.
- `test_inventory_summary.py`: inspect test inventory.
- `ci/`: CI service initialization helpers.

Use an installed or editable checkout on Python 3.14.5+. The root Makefile and tox
analytics environment call the repository smoke script. These development scripts
are excluded from package distributions. Live samples run with `python -m samples.alleles`
or `python -m samples.polars_parquet_duckdb` and close their Service and DuckDB resources.
