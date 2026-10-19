# intermine314 0.2.0 — speaker notes

Audience: scientific users, maintainers and technical decision makers.
Suggested duration: 12–15 minutes, with the evidence appendix available for Q&A.
The accompanying PDF uses vector graphics, embedded fonts and clickable sources.

1. **Familiar calls. Modern analytics.** Introduce 0.2.0 as a compatibility and
   reliability release with a resource-aware analytical pipeline. Polars, DuckDB
   and Parquet existed in 0.1.7; the improvement is integration and hardening.
2. **One shared runtime.** Explain the separation of orchestration, validation,
   persistent columnar storage and native analysis. CSV ingestion is explicit.
   No new TypeScript runtime or bespoke Rust extension is required.
3. **Legacy workflows.** The historical `webservice.Service` facade selects
   legacy row/object defaults; native `service.Service` keeps dictionaries.
   The illustrative `service` variable is an already opened Service. Inventory
   counts describe assertion scopes against the original InterMine client 1.13,
   rather than a compatibility percentage or the number of unchanged programs.
4. **CPU budget.** Explain stage choices rather than additive thread pools.
   Public HTTP starts at four workers; server permission and measured behavior
   determine safe increases. Affinity and discoverable quotas can reduce usable
   CPU. Already initialized native pools and caller-owned work have limitations.
5. **Memory and scheduling.** Completed pages waiting for an earlier page remain
   charged, as do emitting pages. Byte accounting is estimated; memory includes
   Python, native buffers and input materialization outside some package limits.
   One oversized page can proceed to avoid deadlock. This is not an RSS cap.
6. **Reliability.** These are direct local comparisons with the published 0.1.7
   source, both on modern dependencies: failed status footer, gzip sized reads,
   and an export producer that fails after emitting a batch. Atomic publication
   has portable fallback and cross-process/crash limits; do not imply ACID.
7. **Storage.** The fixture has 100,000 sequential integer identifiers and 100
   repeating string values, supplied as 1,000-row custom dictionary batches.
   Total bytes are 358,045 → 143,069, including the new manifest: 60.0% saved.
   Data-only bytes are 139,905. One compacted data file replaces 100 small parts.
   This is a favorable compaction case, not a savings forecast for every dataset;
   write time increased 75%. Temporary staging space is additional.
8. **Result materialization.** Median time is 18.955687 → 13.021712 ms: 31.3%
   less elapsed time. The old iterator's length hint causes another count request;
   the new unsized stream needs only one response. Fixtures exclude real network
   latency. Query counts are explicit; `len(results)` no longer works.
9. **Development uplift.** The first three timing cards compare the final uplift
   with development checkout `8e584c5`, not with PyPI 0.1.7. Keyed-row access is
   23.391205 → 7.335893 ms. Eligible numeric export improves 3.36× for its gated fixture (see the canonical
   summary for exact observations). Repeated SQL
   measures 25 small sums: borrowed parent connection creation is outside the
   timed region, and reuse is opt-in. The retained keyed-row allocation figure
   uses the separate **phase 1 → phase 2** comparison: 21,029,031 → 3,216,089
   bytes, an 84.7% reduction. It excludes fixture construction and native RSS.
10. **Workflow demos.** All use simulated 5 ms request latency and 21 paired
    trials. Prefetch batches eight identifiers with two workers, reducing 32
    requests to four. Reusing one Service reduces metadata requests from 40 to
    two. Four independent mines perform the same 12 requests with overlap.
    These are opt-in design examples, not certified public-mine speed/capacity.
11. **Performance boundaries.** All four comparisons use 0.1.7 source on the
    same modern dependencies as the candidate. DuckDB 1.0.0, pinned by the old
    package, had no compatible binary wheel in the tested CPython 3.14 Linux
    environment; source builds were not attempted. Custom writers bypass the
    built-in HTTP columnar path. New schema checks, staging and compaction add
    work, but their causal shares were not profiled. Do not multiply component
    gains or claim universally lower memory or whole-job time.
12. **Release assurance.** Source suite: 2,813 tests. Installed verification:
    904 regressions, source-package imports excluded. Six installation modes:
    base, plots, analytics alias, sdist, speed and combined. The release workflow
    revalidates the exact tag and artifacts before trusted publication. The
    publication record retains workflow URL, source SHA and matching PyPI hashes.
13. **Migration.** No original `intermine` namespace shim is installed, and
    pandas-specific code requires Polars migration. Old model submodules,
    mine-policy resolvers and Tor helpers have explicit migration requirements.
    Set parsers before stream opening; streamed built-in metadata is immutable.
    Python 3.14.5+ and core PyArrow are required.
14. **Get started.** Install exactly 0.2.0, choose the desired compatibility
    profile and migrate a representative query first. Close streams on early
    exit and close the Service after its consumers. Use Parquet for reusable
    results and DuckDB/Polars for analysis.
15. **Evidence appendix.** All timing datasets retain source digests and trial
    distributions. Warm timings exclude imports and one warmup; filesystem
    caches were not purged. WSL2 exposed 18 CPUs, rather than the unrestricted
    20-core host. Both sides used CPython 3.14.5 with JIT disabled; no speedup is
    attributed to Python 3.14 itself. Historical measurements are dated
    14 October; publication verification is dated 19 October 2026.

Primary evidence:

- `docs/analysis/release-v020-vs017/report.md` and `trials.parquet`.
- `docs/analysis/uplift-2026-10-14/report.md`, `final_summary.parquet` and
  `phase2_memory_trials.parquet`.
- `docs/analysis/uplift-2026-10-14/phase5-experiments.md`.
- The release workflow and `publication.json` alongside this deck.

The measurements use different baselines deliberately. Keep those qualifiers
when reusing any slide or numerical claim outside this presentation.
