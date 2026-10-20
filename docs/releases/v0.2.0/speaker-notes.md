# intermine314 0.2 release series — speaker notes

Current-source preview, 21 October 2026. Publication is deferred at the user's
request. Existing PyPI 0.2.0 files are an earlier build and remain unchanged.
Allow 12–15 minutes. Numbers come from the PDF and the
[original-client comparison](../../analysis/original-comparison/README.md).

1. **Familiar calls. Modern analytics.** Explain the two choices: preserve
   historical calls through `intermine314.webservice`, or explicitly adopt
   native analytics. The baseline is original InterMine 1.13.0.
2. **Measured outcomes.** The headline example is one million annotation rows.
   Read speed compares CSV/pandas with Parquet/DuckDB; storage compares Parquet
   with plain CSV; memory compares pandas export with bounded Parquet. These
   denominators must not become a universal package speedup.
3. **Compatibility.** The 460-symbol audit has 338 passing scopes and 122 symbols
   with tested departures. This is not a percentage of unchanged programs.
   Legacy dataframe output requires optional pandas and uses Arrow dtypes.
4. **One pipeline.** Both interfaces share transport, query building and export.
   No original InterMine dependency, TypeScript runtime or custom Rust extension
   is installed. DuckDB, Arrow and Polars supply native execution.
5. **API timings.** Warm in-memory fixtures exclude HTTP, storage and native
   memory. Legacy sizing/list behavior retains extra traversals. Native removes
   them. Do not label native gains as import-only gains.
6. **Export time.** Streaming CSV is cheap. Parquet includes typing, page
   validation, compression and publication. One-response CSV and multi-page
   Parquet are complete workflow comparisons, not equal-request scaling tests.
7. **Memory.** Use absolute RSS including engines. `/proc/self/status` supplies
   each worker's peak after exec; `getrusage` inherits the parent's high-water
   mark and is avoided. Columnar engines have a larger fixed footprint. The
   100k cases use more memory than original pandas. At 1M rows, annotations use
   37.6% less but diverse data uses 7.0% more; streaming CSV stays small.
8. **Storage.** Eight columns include identifier strings, Unicode and nulls.
   Repeated annotations compress well; diverse IDs and DNA sequences less so.
   CSV+gzip is a measured control. Original InterMine can also feed an external
   Parquet writer; the format is not exclusive to this package.
9. **Repeated reads.** Every method returns identical grouped counts and sums.
   Pandas selects only two columns. DuckDB-over-CSV separates engine and format
   effects. These are warm-cache local reads with new connections, not WAN tests.
10. **Job estimates.** Add measured export time to N measured query times.
    Positive savings mean faster; negative means slower. These are modeled
    scenarios, not independently observed complete jobs.
11. **Workers.** Default native budget 16; options 8/6/4/2. Stage allocations are
    not additive. WSL2 exposes 18 CPUs of the 265K host. Four fetch workers did
    not establish an advantage over one in this low-latency test. Public mine
    concurrency must respect server permissions and throttling.
12. **Correctness.** Footer validation, compression, buffered-page accounting,
    typed values and failure-safe publication are tested. Atomic rename does
    not establish crash durability or cross-process isolation. Estimated queue
    limits are not hard RSS limits.
13. **Validation.** Source and installed-suite counts have recorded validation.
    Seven fresh modes cover base, plots, analytics, sdist, speed, pandas and
    combined extras. Two mines passed small read-only smoke checks. Authenticated
    writes, full datasets and every legacy argument combination were not tested.
14. **Adoption.** Retain query building and persist results when reuse warrants
    it. Real applications should close Service lifetimes. Prepared local 0.2.0
    artifacts differ from published 0.2.0 and cannot overwrite its files.
15. **Methods.** Both clients use Python 3.14.5. Original Service needs a
    benchmark-only MutableMapping alias; its installed source is unchanged.
    Requests and pandas are installed explicitly for the original. Candidate
    uses the speed extra. No gain is attributed independently to Python 3.14.
    Timings exclude imports, metadata, fixture generation, fsync and verification;
    RSS includes imports. Full persisted-value hashes and all aggregates match.
    Raw observations and provenance remain Parquet, queried through DuckDB.
