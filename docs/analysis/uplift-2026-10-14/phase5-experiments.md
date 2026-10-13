# Workflow and conditional optimization evidence

`workflows.py` exercises existing APIs through an offline transport with an
explicit 5 ms delay per request. It checks equal results in 21 randomized paired
trials. These measurements illustrate request batching and overlap; they do not
establish public-server throughput or permissible request rates.

| Workflow | Before ms | Opt-in ms | Requests before → after |
|---|---:|---:|---:|
| Read 32 lazy object names; prefetch batches of 8, 2 workers | 180.150 | 22.901 | 32 → 4 |
| Discover model/templates on 4 independent mines, 4 workers | 67.438 | 19.104 | 12 → 12 |
| Read model metadata 20 times; reuse one Service | 222.142 | 11.115 | 40 → 2 |

Examples in `docs/source/resources.rst` retain explicit Service/iterator
lifetimes, bounded prefetch and Python 3.14 bounded task submission. They add no
automatic requests. Independent mine discovery and repeated work against one
mine have different ownership needs; mutations and publishing retain one owner.

## O16: native Arrow read_all rejected

`reader_probe.py` compared 21 randomized pairs on 100,000 rows with 128-row
batches. Remaining-batch read time, excluding query/connection setup, fell from
**13.080 → 3.714 ms** using Arrow's native `read_all`. Tables matched exactly.
Pinned Arrow 26 source confirms `ToTable()` runs under a single native call.

The lifecycle gate failed. An event-controlled source requested runtime closure
during its first batch. The current wrapper consumed **1 batch** and raised
`RuntimeError`; the native prototype consumed **all 20 batches** and returned
the entire table. Its one admission/lock scope removes the package cancellation
check between batches. `reader_cancellation.parquet` records both outcomes;
`reader_trials.parquet` stores timings. Keep the existing batch loop, closed-reader
semantics and cleanup. No runtime change is adopted for O16.
