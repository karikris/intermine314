# Archived 0.2.0 candidate versus published 0.1.7 measurements

Measured 14 October 2026 on Linux CPython 3.14.5. Candidate source:
`a77db1dedf02b7e3c0bc992466dd6ae2016a0463`. These observations do not describe
current master performance. Both sources used identical modern dependencies;
this is not a comparison of two stock installations or the original `intermine` client.
The candidate was still declared 0.1.8 when measured.

Baseline PyPI source commit: `8d4b833b61f206cd7932707db0520a8e133879af`.
The 0.1.7 wheel SHA-256 is `31f64151cb65bc3ff9b5d3418327e06d048bbb38e5d4ffa45480086c08167ee5`;
sdist SHA-256 is `18974b4013016ac524cf6e784ee565e5cee19022c63c8349453582b29c9e2ee6`.
Raw artifact verification, source hashes, dependency versions, trial timings,
Python allocation peaks, request counts, row-group/part counts and storage bytes
remain in adjacent Parquet. Migration checksums are in
`../../cleanup/evidence-provenance.parquet`. Historical distribution/dependency
receipts retain their original sources and dates; current CI is linked in
`../../cleanup/`.

| Fixture | Source | Trials | Median ms | Total output bytes |
|---|---|---:|---:|---:|
| dict_iter | 0.1.7 | 21 | 8.645 | n/a |
| dict_iter | candidate | 21 | 11.685 | n/a |
| dict_list | 0.1.7 | 21 | 18.956 | n/a |
| dict_list | candidate | 21 | 13.022 | n/a |
| export_directory | 0.1.7 | 21 | 59.562 | 358045.0 |
| export_directory | candidate | 21 | 104.396 | 143069.0 |
| export_single_numeric | 0.1.7 | 21 | 9.487 | 35345.0 |
| export_single_numeric | candidate | 21 | 21.692 | 34807.0 |

Materializing 10,000 dictionaries took 31% less time, while streaming took 35%
more time. Custom-batch single-file and directory exports took 129% and 75%
more time. The directory fixture used 60% less total space including its
manifest. These fixtures exclude the builtin HTTP columnar path and public-mine
network latency; no universal client or complete-job speedup is established.

Reproduction uses `python -m benchmarks.runners.prepare_release_baseline`, then
`python -m benchmarks.runners.release_comparison --candidate /path/to/source
--output /tmp/fresh-comparison`. Checkout the pinned candidate to reproduce the
historical comparison; using master measures a different candidate. New runs
write outside the archived evidence. Storage measurements have a separate
`benchmarks.runners.storage_comparison` runner and
`benchmarks.runners.storage_tables` documentation builder.
