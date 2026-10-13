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
