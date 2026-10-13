Storage savings and job-time estimates
======================================

For the diverse one-million-row fixture, Zstandard Parquet used **61.7% less
space than CSV** and **20.7% less than gzip CSV**. A selective DuckDB aggregation
was **6.6 times faster** over Parquet than typed CSV. These are local synthetic
measurements, not a universal speedup for replacing the InterMine client.

What is compared
----------------

The original InterMine client can stream CSV, TSV and JSON and materialize a
pandas dataframe. It does not define one mandatory on-disk storage format.
GitHits verified those formats and the dataframe implementation against
`InterMine Python client 1.13.0 <https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/results.py#L320-L323>`_
and its
`dataframe method <https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query.py#L1444-L1467>`_.

The baseline here is an **InterMine-style CSV workflow** using Python's CSV
writer, followed by either a Python row aggregation or DuckDB over typed CSV.
The comparison uses intermine314's actual typed Parquet writer and
``query_parquet()``. It does not execute the original client, benchmark pandas,
measure HTTP download speed or compare the client's default iteration cost.
Both writers start with the same already available Python rows. A raw server
CSV stream could avoid this decoding/re-encoding cost. Parallel HTTP gains from
other benchmarks cannot be applied directly: the original client can use one
streaming response rather than one request per page.

Measured storage
----------------

Both fixtures contain eight columns: integer ID, identifier, length, score,
name, nullable integer, description and chromosome. The repetitive fixture has
reused annotations; the diverse fixture has deterministic random identifiers,
names, scores and 80-base sequences (seed 314). They illustrate compressibility
differences and do not establish a typical distribution for a real mine.
Sizes below are decimal MB; gzip uses level 6 and Parquet uses Zstandard with
the package's default 122,880-row target row group. CSV/TSV include headers and
UTF-8 LF records; JSON is compact arrays/objects without an HTTP status envelope.
All encoded values were checked. Text does not retain the same type/null schema
as Parquet; the benchmark supplies CSV types explicitly and retains leading zeros.

.. list-table:: Measured storage: 1,000,000 rows
   :header-rows: 1

   * - Fixture
     - Text format
     - Text MB
     - Parquet MB
     - Space saved
   * - Repetitive
     - csv
     - 120.974
     - 2.239
     - 98.1%
   * - Repetitive
     - tsv
     - 120.974
     - 2.239
     - 98.1%
   * - Repetitive
     - json_rows
     - 131.545
     - 2.239
     - 98.3%
   * - Repetitive
     - json_objects
     - 267.545
     - 2.239
     - 99.2%
   * - Repetitive
     - csv.gz
     - 13.361
     - 2.239
     - 83.2%
   * - Repetitive
     - tsv.gz
     - 13.361
     - 2.239
     - 83.2%
   * - Repetitive
     - json_rows.gz
     - 15.184
     - 2.239
     - 85.3%
   * - Repetitive
     - json_objects.gz
     - 18.931
     - 2.239
     - 88.2%
   * - Diverse
     - csv
     - 176.267
     - 67.453
     - 61.7%
   * - Diverse
     - tsv
     - 176.267
     - 67.453
     - 61.7%
   * - Diverse
     - json_rows
     - 186.838
     - 67.453
     - 63.9%
   * - Diverse
     - json_objects
     - 322.838
     - 67.453
     - 79.1%
   * - Diverse
     - csv.gz
     - 85.092
     - 67.453
     - 20.7%
   * - Diverse
     - tsv.gz
     - 85.093
     - 67.453
     - 20.7%
   * - Diverse
     - json_rows.gz
     - 86.541
     - 67.453
     - 22.1%
   * - Diverse
     - json_objects.gz
     - 93.604
     - 67.453
     - 27.9%

At 100,000 rows, diverse CSV occupied 17.527 MB, gzip CSV 8.496 MB and
Parquet 6.783 MB: savings of 61.3% and 20.2%, respectively. All 32 measured
format/size combinations are retained in ``storage/storage.parquet``.
The much higher repetitive-data savings should not be used as a general forecast.
For data resembling the diverse million-row fixture, a **conditional** 100 GB
CSV estimate is 38.3 GB of Parquet; 100 GB of gzip CSV is 79.3 GB of Parquet.
Those large sizes were not written or measured.

Measured local processing
-------------------------

Seven repetitions per case were run in randomized order after a warm-up.
The table shows median seconds and the observed minimum–maximum range for the
diverse fixture. Query work groups by chromosome and computes count/sum of
length, projecting two of eight columns. Each DuckDB operation includes a fresh
connection; Parquet includes package validation and Arrow/Polars result
conversion. Writes include conversion, staging, ordered compaction and final
publication for Parquet; CSV includes row serialization. Neither writer calls
``fsync``. Import time, fixture generation and correctness checks are untimed.

.. list-table:: Measured seconds: median (min–max)
   :header-rows: 1

   * - Stage
     - 100,000 rows
     - 1,000,000 rows
   * - Write CSV
     - 0.1466 (0.1455–0.1741)
     - 1.4921 (1.4324–2.3323)
   * - Write package Parquet
     - 0.1966 (0.1933–0.2051)
     - 1.1551 (1.1425–1.2657)
   * - CSV → Python aggregate
     - 0.1039 (0.1021–0.1054)
     - 1.0309 (1.0189–1.0606)
   * - CSV → DuckDB aggregate
     - 0.0453 (0.0436–0.0469)
     - 0.1190 (0.1107–0.1237)
   * - gzip CSV → DuckDB aggregate
     - 0.2091 (0.2074–0.2119)
     - 0.8240 (0.8196–0.8402)
   * - Parquet → package DuckDB aggregate
     - 0.0160 (0.0144–0.0174)
     - 0.0181 (0.0170–0.0215)

At 100,000 diverse rows, Parquet writing was slower. A single CSV/DuckDB
analysis plus writing remained about **0.021 seconds faster** than Parquet;
two analyses amortized that extra Parquet write cost. At one million rows,
both Parquet writing and selective analysis were faster. The one-million-row
Parquet query was 45.6 times faster than gzip CSV/DuckDB and 57.1 times faster
than Python CSV aggregation in this fixture. These large ratios describe a
narrow, warm-cache projection, not every query or an entire download job.
Repeated queries against a preloaded DuckDB table, full-column scans, joins,
real sequence distributions and cold or remote storage require other measurements.

Estimated complete jobs
-----------------------

The following totals are **estimates**, even at measured sizes. They assume
both clients deliver 100,000 rows/second including server work, transfer and
decode, then write once and run the indicated number of local analyses::

    CSV total     = rows / assumed_fetch_rate + CSV_write + K * CSV_query
    Parquet total = rows / assumed_fetch_rate + Parquet_write + K * Parquet_query
    time saved    = CSV total - Parquet total

The common fetch rate is an explicit scenario, not an observed server rate.
No parallel-download advantage is assumed. Real acquisition costs can differ
between CSV and JSON and between clients. The 100,000/1,000,000-row local stages
use measured medians. The 10/100-million-row stages scale the million-row
medians linearly and are **unmeasured extrapolations**; memory pressure, storage,
cache limits and row-group geometry can invalidate that scaling. All totals
are seconds, rounded only for display; negative savings mean Parquet is slower.

.. list-table:: CSV with DuckDB: isolates much of the format effect
   :header-rows: 1

   * - Rows
     - Local basis
     - Analyses
     - CSV total s
     - Parquet total s
     - Saved s (%)
   * - 100,000
     - measured medians
     - 1
     - 1.192
     - 1.213
     - -0.021 (-1.7%)
   * - 100,000
     - measured medians
     - 10
     - 1.599
     - 1.357
     - 0.242 (15.2%)
   * - 1,000,000
     - measured medians
     - 1
     - 11.611
     - 11.173
     - 0.438 (3.8%)
   * - 1,000,000
     - measured medians
     - 10
     - 12.682
     - 11.336
     - 1.347 (10.6%)
   * - 10,000,000
     - linear extrapolation
     - 1
     - 116.111
     - 111.731
     - 4.380 (3.8%)
   * - 10,000,000
     - linear extrapolation
     - 10
     - 126.822
     - 113.356
     - 13.466 (10.6%)
   * - 100,000,000
     - linear extrapolation
     - 1
     - 1161.107
     - 1117.311
     - 43.796 (3.8%)
   * - 100,000,000
     - linear extrapolation
     - 10
     - 1268.220
     - 1133.563
     - 134.657 (10.6%)

.. list-table:: CSV with Python: also replaces the analysis engine
   :header-rows: 1

   * - Rows
     - Local basis
     - Analyses
     - CSV total s
     - Parquet total s
     - Saved s (%)
   * - 100,000
     - measured medians
     - 1
     - 1.250
     - 1.213
     - 0.038 (3.0%)
   * - 100,000
     - measured medians
     - 10
     - 2.186
     - 1.357
     - 0.829 (37.9%)
   * - 1,000,000
     - measured medians
     - 1
     - 12.523
     - 11.173
     - 1.350 (10.8%)
   * - 1,000,000
     - measured medians
     - 10
     - 21.801
     - 11.336
     - 10.465 (48.0%)
   * - 10,000,000
     - linear extrapolation
     - 1
     - 125.230
     - 111.731
     - 13.498 (10.8%)
   * - 10,000,000
     - linear extrapolation
     - 10
     - 218.011
     - 113.356
     - 104.655 (48.0%)
   * - 100,000,000
     - linear extrapolation
     - 1
     - 1252.296
     - 1117.311
     - 134.985 (10.8%)
   * - 100,000,000
     - linear extrapolation
     - 10
     - 2180.109
     - 1133.563
     - 1046.546 (48.0%)

At one million rows and ten analyses, the assumed network-inclusive saving
is about **1.35 seconds (10.6%)** against CSV/DuckDB or **10.47 seconds (48.0%)**
against CSV/Python. Slower server delivery increases both totals and reduces
the percentage saved. For an already downloaded dataset, omit the common fetch
term. Gzip initial-write cost was not timed, so no gzip total-job estimate is
claimed. The Parquet estimates also exclude migrating an existing CSV archive.

Reproduce and inspect
---------------------

Measured on 3.14.5, Polars 2.0.0,
PyArrow 26.0.0 and DuckDB 1.5.6,
on Linux/WSL2 with an Intel Core Ultra 7 265K. The guest exposes **18 CPUs**,
not all 20 host cores. Budget: 16 active native CPU workers; Polars 12,
Arrow CPU 4/I/O 2, DuckDB SQL 16 and ordered COPY 4, admitted by stage.
Files were on ext4, with warm OS cache; this is not a bare-host disk benchmark.
The fixture is materialized in memory for equality checks, so this benchmark
does not measure the package's bounded-streaming memory use.

From the repository root, choose an empty output directory::

    .venv/bin/python -m benchmarks.runners.storage_comparison \
        --output /tmp/fresh-format-evidence \
        --scratch-root /path/on/data/disk --rows 100000 1000000 --repeats 7

Canonical observations, 168 trials, timing summaries, 24 job scenarios and
environment/source hashes are Parquet in
``docs/analysis/release-2026-10-14/storage/``. ``analysis.duckdb`` holds relative
views: open it with that directory as the working directory. For example::

    import duckdb
    with duckdb.connect("analysis.duckdb", read_only=True) as con:
        print(con.sql("SELECT * FROM storage_savings").fetchall())
        print(con.sql("SELECT * FROM timings").fetchall())
        print(con.sql("SELECT * FROM job_estimates").fetchall())

``docs/analysis/release-2026-10-14/build_storage_tables.py`` regenerates this
page and forecasts from the committed observations. Earlier performance reports
retain their original dependency versions and measurements. Their old
intermine314 baseline is not the original InterMine client.
