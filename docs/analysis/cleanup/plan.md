# Package cleanup implementation

Baseline: `5ea5b765d9edfff04fda3aff82cc9a14bc34e977`, Python 3.14.5, version 0.2.0. Source suite: **2,734 passed in 21.45 seconds**. Runtime file hashes, physical lines, non-comment tokens and declaration counts are recorded in `baseline-runtime.parquet`; package sizes in `baseline-metrics.parquet`.

| Phase | Tasks (one commit each) | Required completion |
|---|---|---|
| 1 | Baseline; keep/migrate/delete manifest; upstream/obligation map | Inventory and compatibility baseline |
| 2 | Benchmark defaults; removed-key errors and precedence; policy simplification | Single offset scheduler and resource policy |
| 3 | Dead helpers; account transaction; response ownership | Independent account state and exactly-once cleanup |
| 4 | Builtin execution; custom provider adapter; retire forwards | Identical wire behavior and override dispatch |
| 5 | Internal XML parser; Query/Template migration; DOM retirement | Empty/CDATA/descendant semantics and public DOM boundary |
| 6 | Compatibility guidance; portable evidence; superseded-file deletion | Working links and retained measurement provenance |
| 7 | Source/install acceptance; size/performance comparison; final dispositions | Full source and installed contracts, final CI |

Each phase ends with a fresh scoped API trace/render, lint, appropriate source tests and a push to master. Full CI must succeed on the exact pushed SHA before the next phase starts. The next phase records the previous phase receipt; the final response links the final run. Runtime changes require semantic obligation mappings, not comparison of shifted coverage line numbers. No version change, tag or publication is authorized by this cleanup.

Retain the 460-symbol historical inventory, 331 passing-scoped / 129 tested-departure classification across 43 groups, 40 supplemental reexports, signature/alias/exception identity and custom providers. Preserve footer validation, bounded outstanding pages and bytes, atomic Parquet publication, precision, borrowed lifetimes and host/Tor admission. Keep default CPU budget 16 and profiles 8/6/4/2. Git history provides recovery for removed implementations.
