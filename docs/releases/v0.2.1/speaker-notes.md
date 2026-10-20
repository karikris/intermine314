# intermine314 0.2.1 — speaker notes

1. **Familiar InterMine. Modern performance.** Introduce the value proposition:
preserve established scientific workflows while adopting Python 3.14.5+ and
native analytical engines. Familiarity is a scoped compatibility promise,
not complete equivalence with every upstream behavior.
2. **A familiar way to work.** Retain the historical `webservice.Service` path
when replacing the import prefix. Change dependency declarations explicitly.
No `intermine` namespace shim is installed. Review documented departures,
dtype assumptions and resource lifetimes before migrating production code.
3. **Two releases.** Earlier package releases predate the broad compatibility
restoration. 0.2.0 restores queries, models, XML, templates, lists, summaries,
registry and identifier-resolution entry points with shared exception identity.
0.2.1 focuses on the behavior existing code expects behind those entry points.
4. **What 0.2.1 adds.** Legacy `dataframe()` defaults to optional pandas, using
Arrow extension dtypes; native defaults to Polars. Legacy length/truth checks
independently traverse results, and list construction can issue two requests.
Row/object metadata is mutable and active row/view/parser changes are honored.
Public UTF-8 encoders and typed constraint collection identity are restored.
Native streams retain unsized, immutable-layout optimizations. Shared execution
also preserves custom Query/Template endpoints and wire parameters. Fresh
installations use current configuration; copied obsolete scheduler keys are
not translated. These details are recorded in `compatibility.parquet`.
5. **Speed.** Both headline figures use one million synthetic annotation rows
and median timings from five fresh-process repetitions. Aggregation compares
CSV/pandas with Parquet/DuckDB: 0.6352 → 0.01972 seconds (32.2×). Export compares
original InterMine dataframe→CSV with four-worker Parquet: 7.3277 → 2.8642
seconds (2.6×). They are complete workflow comparisons, not import-only gains.
Original streaming CSV is cheaper to export; small jobs may favor the simpler
workflow. Timings exclude imports, metadata, fixture generation and validation.
6. **Storage.** One million rows use 112.92 → 3.74 MiB for repeated annotations
and 164.52 → 58.86 MiB for diverse identifiers/sequences. Plain-CSV savings are
96.7% and 64.2%. Gzip CSV is a measured control: 11.54 and 73.91 MiB. All values
were checked after decoding. Final file sizes exclude staging. Original
InterMine can also feed external Parquet writers; the format is not exclusive.
7. **Resources and memory.** The default native CPU budget is 16, with 8/6/4/2
alternatives. Stage allocations share admission and are not additive. The host
was an Ultra 7 265K with 18 CPUs exposed by WSL2. Four fetch workers did not
establish a general advantage over one. Public-server limits remain relevant.
For the annotation fixture, Parquet export peak RSS was 532.0 versus 852.7 MiB
for original pandas: 37.6% lower. Diverse data was 7.0% higher; streaming CSV
remained smallest. Byte estimates constrain queued work, not total process RSS.
8. **One pipeline.** Persist once, then query the dataset repeatedly. The diagram
shows the consumer flow; Arrow also supports ingest conversion before Parquet.
Footer validation, typed data and staged publication protect correctness.
Atomic replacement does not imply crash durability or cross-process ACID.
9. **Confidence.** Current evidence records 2,805 source tests, 1,003 installed
regressions and seven installation modes. Small read-only LegumeMine/MaizeMine
checks validate rows, exports and template parsing. Scoped passing assertions
rise from 331 in published 0.2.0 to 338 in the candidate, out of 460 symbols;
122 retain tested departures. Counts do not quantify complete compatibility.
Windows/macOS, free-threaded builds and all authenticated operations are not
certified by these checks.
10. **Close.** Invite review of the project, compatibility guide and measured
evidence. Keep the discussion centered on familiar calls, reusable data and
native analytics. The release-preview label remains until actual publication.

## Sources and methods

* [Release changes](../../../CHANGELOG.md) and the
[published 0.2.0 migration guide](https://github.com/karikris/intermine314/blob/6c5077ec0e61127b94fec277e25e1330e5441e87/docs/source/release_readiness.rst).
* [0.2.1 compatibility acceptance](../../analysis/v021-readiness/completion.md).
* [Full original-client measurements](../../analysis/original-comparison/README.md):
160 workflow trials, 80 API observations, full-value fingerprints, ranges and
exact dependency provenance in Parquet.
* [Validated source CI](https://github.com/karikris/intermine314/actions/runs/39432036735).

Both clients ran on CPython 3.14.5. Original InterMine 1.13.0 needs a benchmark-only
MutableMapping alias for its Service import; its installed source was unchanged.
Candidate measurements include the optional speed extra. No gain is attributed
independently to the Python version. Data are synthetic, caches warm, and network
is loopback without added latency. No WAN, cold-disk or fsync claim is made. Linux
VmHWM measures each worker's own peak RSS, including native libraries. See the
full report for all measured disadvantages and workload-specific boundaries.
