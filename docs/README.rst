Build Documentation Locally
===========================

The docs target the ``intermine314`` package line and Python 3.14+.

Build HTML docs
---------------

::

   git clone https://github.com/kriskari/intermine314
   cd intermine314
   python3.14 -m venv .venv
   . .venv/bin/activate
   pip install --upgrade pip
   pip install sphinx
   pip install -e ".[dev]"
   cd docs
   make clean
   make html

Open the generated site
-----------------------

::

   cd build/html
   python -m http.server 8000

Then browse to ``http://localhost:8000``.

Notes
-----

- The docs include analytics workflows built around ``Query.dataframe()``,
  ``Query.to_parquet()``, and ``Query.to_duckdb()``.
- API pages are generated from source modules via Sphinx autodoc.
- ``docs/build`` is generated output and is intentionally not tracked.
