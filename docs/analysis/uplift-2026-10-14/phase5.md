# Phase 5 handoff

Opening receipt: `f71a35bb4d4eb3cc95c269c828f215157ad301a0`, all seven jobs
passed in [CI 38471063810](https://github.com/karikris/intermine314/actions/runs/38471063810).
Tasks: 5.1 `b22f61d`, 5.2 `5f97d95`, 5.3 `98757d4`, 5.4 `d612e08`.
Full API trace **2,812 passed in 34.14s**, Ruff clean, all 460 historical symbols
accounted for. No dependencies or version changed.

Identifier upload preparation (100k inputs) improved **42.090 → 37.189 ms**
against phase 4 and **42.488 → 37.365 ms** against the original baseline in
21-pair comparisons. Bounded joined chunks replace per-identifier writes.
Complete validation still precedes the single network mutation; exact quoting,
Unicode, leading zeroes, chunk boundaries, oversized identifiers and known
Content-Length are covered. Chunk size is ~64 KiB plus one individual identifier;
Arrow input batches remain bounded separately.

Existing prefetch/service-reuse/independent-discovery APIs have measured offline
examples and explicit lifetime ownership. See `phase5-experiments.md` and the
canonical Parquet tables for request counts and simulated-latency caveats.

Two conditional candidates are completed by measured rejection:

- O16 native `read_all`: 3.52× faster locally, but removes between-batch runtime
  cancellation; keep the existing reader loop and ownership semantics.
- O17 worker setup state: 1.8% and 2.1% median improvement across two 21-pair
  comparisons; below the 5% gate. Prototype source retained in Parquet; production
  worker restored. Per-task callback lookup, native resets and teardown stay.

Pinned GitHits sources cover bytes joining, bounded Python 3.14 task submission,
spawn isolation and Arrow's native ToTable implementation. Phase 6 must verify
this exact pushed SHA before final consolidation and distribution checks.
