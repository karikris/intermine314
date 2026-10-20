# Original InterMine versus current intermine314

Measured 21 October 2026 on Intel Core Ultra 7 265K / WSL2 (18 exposed CPUs),
CPython 3.14.5. **Publication is deferred; existing PyPI 0.2.0 is unchanged.**
The candidate is the installed current-source wheel, not the earlier PyPI 0.2.0 wheel.

## What was measured

Original **intermine 1.13.0** runs in an isolated environment. The candidate
environment contains no original InterMine and includes its optional speed extra
(orjson). The runner checks every candidate runtime file by content digest.
A deterministic local HTTP server supplies eight typed fields: integer IDs,
leading-zero or diverse identifiers, lengths, exact quarter-valued scores,
names, nullable strings, Unicode annotation text or diverse DNA sequences, and
chromosomes. This is synthetic data, not a sampled public-mine population.

Five fresh worker interpreters run each case at 100k and 1M rows for each entropy
profile: **160 complete-workflow trials**. Order is randomized within export and
read stages. Timings exclude imports, metadata discovery, fixture creation and
post-run verification. OS file caches are warm; there is no artificial latency,
fsync, cold-disk control, CPU pinning or WAN/server-throughput certification.
Each scan includes a fresh engine connection where relevant. RSS includes imported
libraries and native buffers, but excludes the separate fixture server.

Original exports use actual `Service`, `Query.results(row="list")` or
`Query.dataframe()` calls. Streaming JSON rows are written to CSV with `csv.writer`;
the dataframe route uses pandas `to_csv`. Direct server-side CSV passthrough is
**not timed** and may be cheaper. Original InterMine can also feed external Parquet
writers; it does not require CSV. The modern route calls `Query.to_parquet()` with
ordered 10k-row pages/batches, one or four HTTP workers, one outstanding page per
worker, a 32 MiB estimated queue limit and the default cpu16 resource profile.
This is a comparison of complete storage workflows, not equal-request HTTP scaling.

Every exported value is read back with explicit CSV types or Parquet schema and
checked against the same ordered SHA-256 fingerprint. All aggregate results match.
CSV+gzip level 6 is additionally round-trip checked; compression time is not timed.
Disk sizes refer to final data files, excluding staging space and dependencies.

## Python 3.14 baseline adjustment

The original wheel cannot import `intermine.webservice.Service` unchanged on Python
3.14: it imports `collections.MutableMapping`, removed from that location.
The benchmark alone aliases it to `collections.abc.MutableMapping` before import.
No installed source is patched. Requests and pandas are installed explicitly.
GitHits verified the original [Service import](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/webservice.py#L16)
and [dataframe implementation](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query.py#L1444).
Both sides use the same Python version, so these results do not isolate a Python
3.14 interpreter speedup. Native and historical facade behavior are distinguished.

## Complete workflow timings and memory

Values are medians of five runs. Ranges are observed minimum–maximum wall times,
not confidence intervals. RSS is MiB (2^20 bytes). The baseline is current RSS
immediately before the operation; peak is Linux `/proc/self/status:VmHWM` afterwards.
`getrusage().ru_maxrss` is intentionally not used: a 400 MB parent-allocation probe
showed it inheriting ~405 MB across exec while the child VmHWM was ~14 MB.
The earlier exploratory run with inherited peaks was discarded in full.

| Dataset | Rows | Workflow | Seconds | Range (s) | Peak RSS MiB | Baseline RSS MiB | Requests |
|---|---:|---|---:|---:|---:|---:|---:|
| diverse | 100,000 | `modern_parquet_1` | 0.508 | 0.491–0.526 | 344.8 | 143.7 | 10 |
| diverse | 100,000 | `modern_parquet_4` | 0.509 | 0.500–0.554 | 367.0 | 143.8 | 10 |
| diverse | 100,000 | `original_pandas_csv` | 0.708 | 0.694–0.758 | 146.0 | 79.0 | 1 |
| diverse | 100,000 | `original_stream_csv` | 0.257 | 0.251–0.270 | 79.1 | 79.0 | 1 |
| diverse | 100,000 | `read_csv_duckdb` | 0.056 | 0.053–0.057 | 181.2 | 137.8 | 0 |
| diverse | 100,000 | `read_csv_pandas` | 0.041 | 0.040–0.046 | 91.6 | 69.8 | 0 |
| diverse | 100,000 | `read_csv_python` | 0.106 | 0.104–0.111 | 25.8 | 25.7 | 0 |
| diverse | 100,000 | `read_parquet_duckdb` | 0.018 | 0.017–0.019 | 153.2 | 137.7 | 0 |
| diverse | 1,000,000 | `modern_parquet_1` | 3.491 | 3.469–3.511 | 784.6 | 144.0 | 100 |
| diverse | 1,000,000 | `modern_parquet_4` | 3.309 | 3.258–3.429 | 778.2 | 143.8 | 100 |
| diverse | 1,000,000 | `original_pandas_csv` | 7.113 | 7.044–7.217 | 727.3 | 78.9 | 1 |
| diverse | 1,000,000 | `original_stream_csv` | 2.562 | 2.465–2.606 | 79.2 | 79.0 | 1 |
| diverse | 1,000,000 | `read_csv_duckdb` | 0.126 | 0.117–0.136 | 317.9 | 137.7 | 0 |
| diverse | 1,000,000 | `read_csv_pandas` | 0.333 | 0.321–0.344 | 109.1 | 69.9 | 0 |
| diverse | 1,000,000 | `read_csv_python` | 1.043 | 1.041–1.098 | 25.8 | 25.7 | 0 |
| diverse | 1,000,000 | `read_parquet_duckdb` | 0.019 | 0.017–0.022 | 158.6 | 137.5 | 0 |
| repetitive | 100,000 | `modern_parquet_1` | 0.405 | 0.398–0.427 | 288.7 | 143.9 | 10 |
| repetitive | 100,000 | `modern_parquet_4` | 0.418 | 0.413–0.429 | 298.5 | 143.7 | 10 |
| repetitive | 100,000 | `original_pandas_csv` | 0.713 | 0.710–0.718 | 158.6 | 78.9 | 1 |
| repetitive | 100,000 | `original_stream_csv` | 0.248 | 0.246–0.251 | 79.2 | 78.9 | 1 |
| repetitive | 100,000 | `read_csv_duckdb` | 0.054 | 0.053–0.056 | 170.0 | 137.5 | 0 |
| repetitive | 100,000 | `read_csv_pandas` | 0.072 | 0.071–0.074 | 91.4 | 69.9 | 0 |
| repetitive | 100,000 | `read_csv_python` | 0.095 | 0.094–0.096 | 25.8 | 25.7 | 0 |
| repetitive | 100,000 | `read_parquet_duckdb` | 0.018 | 0.017–0.018 | 152.7 | 137.6 | 0 |
| repetitive | 1,000,000 | `modern_parquet_1` | 2.756 | 2.748–2.865 | 528.6 | 143.7 | 100 |
| repetitive | 1,000,000 | `modern_parquet_4` | 2.864 | 2.848–2.923 | 532.0 | 143.9 | 100 |
| repetitive | 1,000,000 | `original_pandas_csv` | 7.328 | 7.278–7.366 | 852.7 | 79.0 | 1 |
| repetitive | 1,000,000 | `original_stream_csv` | 2.441 | 2.429–2.492 | 79.1 | 79.0 | 1 |
| repetitive | 1,000,000 | `read_csv_duckdb` | 0.114 | 0.107–0.116 | 294.9 | 137.7 | 0 |
| repetitive | 1,000,000 | `read_csv_pandas` | 0.635 | 0.632–0.642 | 115.9 | 69.9 | 0 |
| repetitive | 1,000,000 | `read_csv_python` | 0.968 | 0.947–0.992 | 25.8 | 25.7 | 0 |
| repetitive | 1,000,000 | `read_parquet_duckdb` | 0.020 | 0.018–0.022 | 158.4 | 137.8 | 0 |

The modern fixed footprint is higher. Streaming CSV remains substantially smaller
in memory. At 1M repetitive rows, four-worker Parquet uses 37.6% less peak RSS than
original pandas; at 1M diverse rows it uses **7.0% more**. Both 100k fixtures use more
RSS in the modern export. Memory gains depend on data shape and engine buffering,
not just row count. A page estimate is not a hard RSS cap.

## Storage

| Dataset | Rows | CSV MiB | gzip CSV MiB | Zstd Parquet MiB | Saved vs CSV | Saved vs gzip |
|---|---:|---:|---:|---:|---:|---:|
| repetitive | 100,000 | 11.20 | 1.15 | 0.37 | 96.7% | 67.8% |
| repetitive | 1,000,000 | 112.92 | 11.54 | 3.74 | 96.7% | 67.6% |
| diverse | 100,000 | 16.36 | 7.38 | 5.87 | 64.1% | 20.4% |
| diverse | 1,000,000 | 164.52 | 73.91 | 58.86 | 64.2% | 20.4% |

Compression depends on cardinality, entropy, null frequency, encoding and row groups.
Gzip CSV is a serious storage alternative, but lacks Parquet’s typed columnar reads.
Parquet row groups default to 122,880 rows; ordered compaction and Zstandard are enabled.

## Estimated complete-job time savings

`total = median(export_seconds) + analyses × median(read_seconds)`.
Savings are baseline total minus modern total; negative means the modern workflow
takes longer. This assumes fresh read operations and does not compare retaining a
previously loaded pandas dataframe in memory. These are modeled jobs, not new trials.

| Dataset | Rows | Baseline export / read | Analyses | Original total s | Modern total s | Saved s |
|---|---:|---|---:|---:|---:|---:|
| repetitive | 100,000 | pandas CSV / pandas | 1 | 0.79 | 0.44 | +0.35 |
| repetitive | 100,000 | pandas CSV / pandas | 10 | 1.44 | 0.60 | +0.84 |
| repetitive | 100,000 | pandas CSV / pandas | 100 | 7.93 | 2.19 | +5.74 |
| repetitive | 100,000 | stream CSV / Python | 1 | 0.34 | 0.44 | -0.09 |
| repetitive | 100,000 | stream CSV / Python | 10 | 1.20 | 0.60 | +0.61 |
| repetitive | 100,000 | stream CSV / Python | 100 | 9.77 | 2.19 | +7.58 |
| repetitive | 1,000,000 | pandas CSV / pandas | 1 | 7.96 | 2.88 | +5.08 |
| repetitive | 1,000,000 | pandas CSV / pandas | 10 | 13.68 | 3.06 | +10.62 |
| repetitive | 1,000,000 | pandas CSV / pandas | 100 | 70.85 | 4.84 | +66.01 |
| repetitive | 1,000,000 | stream CSV / Python | 1 | 3.41 | 2.88 | +0.53 |
| repetitive | 1,000,000 | stream CSV / Python | 10 | 12.12 | 3.06 | +9.06 |
| repetitive | 1,000,000 | stream CSV / Python | 100 | 99.27 | 4.84 | +94.44 |
| diverse | 100,000 | pandas CSV / pandas | 1 | 0.75 | 0.53 | +0.22 |
| diverse | 100,000 | pandas CSV / pandas | 10 | 1.12 | 0.69 | +0.43 |
| diverse | 100,000 | pandas CSV / pandas | 100 | 4.86 | 2.30 | +2.55 |
| diverse | 100,000 | stream CSV / Python | 1 | 0.36 | 0.53 | -0.16 |
| diverse | 100,000 | stream CSV / Python | 10 | 1.32 | 0.69 | +0.63 |
| diverse | 100,000 | stream CSV / Python | 100 | 10.86 | 2.30 | +8.55 |
| diverse | 1,000,000 | pandas CSV / pandas | 1 | 7.45 | 3.33 | +4.12 |
| diverse | 1,000,000 | pandas CSV / pandas | 10 | 10.44 | 3.50 | +6.94 |
| diverse | 1,000,000 | pandas CSV / pandas | 100 | 40.36 | 5.19 | +35.17 |
| diverse | 1,000,000 | stream CSV / Python | 1 | 3.60 | 3.33 | +0.28 |
| diverse | 1,000,000 | stream CSV / Python | 10 | 12.99 | 3.50 | +9.49 |
| diverse | 1,000,000 | stream CSV / Python | 100 | 106.86 | 5.19 | +101.66 |

## Import-level API fixtures

Separate warm local two-column fixtures, five isolated-process repetitions.
These timings exclude HTTP and persistence. Python allocation peaks in the raw
file use tracemalloc and must not be confused with complete-workflow RSS.

| Operation | Original ms | Legacy facade ms | Native ms | Legacy speedup | Native speedup |
|---|---:|---:|---:|---:|---:|
| query_xml | 5.145 | 2.106 | 2.208 | 2.44× | 2.33× |
| dict_stream | 16.669 | 13.747 | 7.570 | 1.21× | 2.20× |
| dict_list | 33.005 | 27.496 | 7.807 | 1.20× | 4.23× |
| rows_retained | 14.029 | 12.514 | 5.557 | 1.12× | 2.52× |

XML measures 200 serializations; other common cases process 10,000 rows. Legacy
list construction can make two traversals, matching the original; native makes one.
No native-mode multiplier is presented as an import-only gain.

## Provenance and reproduction

- Benchmark/source revision: `a2204932422278dc90e01da6282aa06927dcd036`.
- Runtime content digest: `5ed02d29785080b8aebb545fc88f05e62701d475fed886c3ed6f4457ee079294`.
- Benchmark script SHA-256: `252fe8100085460ac8c51126875acabd21167cb563fd1ad4ac7254d4f7d17bac`.
- Wheel built from clean revision `0a21ca7cb2c54b7e59fcfb68984382166618ba82`; all runtime files match the measured source.
- `environment.parquet`: machine, measurement time, scope and reference adjustment.
- `trials.parquet`: 160 observations, exact installed dependencies, package digest and full-value fingerprint.
- `summary.parquet`: DuckDB medians, ranges, absolute RSS and request counts.
- `storage.parquet`: 20 file-size observations and verified row counts.
- `api-trials.parquet`: 80 API observations; `api-parity.parquet`: targeted original/legacy/native contract probes.
- `job-estimates.parquet`: explicit modeled totals and denominators.
- `sources.parquet`: primary-source references checked through GitHits and PyPI.

Use Python 3.14.5 and two isolated environments. Install original InterMine,
requests and pandas only in the reference environment. Install the prepared
candidate wheel with `pandas,speed` extras in the candidate. Exact package
versions used in this run are retained in each trial’s `dependencies` field.

```bash
python -m benchmarks.runners.original_comparison \
  --reference-python /tmp/original/bin/python \
  --candidate-python /tmp/candidate/bin/python \
  --rows 100000 1000000 --repeats 5 --output /tmp/new-comparison
python -m benchmarks.runners.compatibility_acceptance \
  --fixture tests/fixtures/compatibility/model.xml \
  --reference-python /tmp/original/bin/python \
  --candidate-python /tmp/candidate/bin/python \
  --measure --repeats 5 --output /tmp/api-trials.parquet
python -m benchmarks.evidence docs/analysis/original-comparison \
  'SELECT fixture, rows, "case", seconds, peak_rss_bytes FROM summary'
```

The runner requires Linux for per-process RSS and an empty output directory.
Omit `--measure` for the API contract probe. The benchmark creates no public
server load. No runtime dependency/version/configuration was changed for this deck.

## Publication decision

The user chose **prepare artifacts and PDF; defer publication**. PyPI already
contains both 0.2.0 filenames, and [does not permit reuse](https://pypi.org/help/#file-name-reuse),
even after deletion. No release was deleted, yanked or retagged. A future code
release should use a fresh version such as 0.2.1; [post-releases are discouraged
for code fixes](https://packaging.python.org/en/latest/specifications/version-specifiers/#post-releases).
Prepared files retain 0.2.0 only for local evaluation and must be rebuilt under
the selected future version before publication.
