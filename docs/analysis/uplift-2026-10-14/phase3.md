# Phase 3 handoff

Opening receipt: clean `891a07708c61786ebff54f62a15996c26ebb72df`, seven successful
jobs in [CI 38467731555](https://github.com/karikris/intermine314/actions/runs/38467731555).
Tasks: 3.1 `2e1871f`, 3.2 `66a9ecf`, 3.3 `bceb16b`, 3.4 immediately preceding
column-conversion commit. Ruff clean; **2,788 tests pass in 33.82s** under the
full API trace; all 460 original symbols accounted for.

21 randomized paired warm runs against phase 2:

| Operation | Previous ms | Candidate ms |
|---|---:|---:|
| 100 template eager calls | 28.477 | 19.277 |
| 1,000 stage worker lookups within one operation | 6.720 | 0.091 |
| Typed 1,000 × 64 integer batch | 5.192 | 3.214 |
| Write/publish/read-count, 10k String/Int64 Parquet rows | 17.345 | 16.972 |

Original-baseline timings are separately retained in `phase3_original_trials.parquet`.
These fixtures do not measure public-mine network speed. Tests assert one deep
query clone per built-in template helper, public clone independence, mutation
fallback in request encoding, quota refresh at the next outer operation,
cancellation/admission, exact types, Decimal scale refinement and fallback
precision. Native pools initialized externally remain unchanged and charged
using their actual reported size; context-local stage decisions avoid repeated
affinity/cgroup reads without a permanent resource-policy cache.

Retired redundant execution cloning, duplicate count response reading,
per-call invariant-key sets, per-column type-map construction and homogeneous
per-column admission. Retained public clone/spec isolation, custom dispatch,
mutated wire mappings and the strict Polars fallback for distinct scalar
semantics. GitHits sources: pinned CPython copy/contextvars/dataclasses and
Polars pool configuration, recorded in `sources.parquet`. Dependencies and
version unchanged. Next phase requires this pushed SHA's green CI.
