# Replacement obligations and upstream verification

GitHits source checks (2026-10-20): InterMine 1.13.0 commit `d888b779c8050bad789e26b312f40d220bc85d0d`, `intermine/webservice.py:657–688` establishes lazy public XML-string template dictionaries; `intermine/query.py:365–395` establishes constructor/iteration and count compatibility. CPython `v3.14.5`, `Doc/library/xml.etree.elementtree.rst` establishes incremental TreeBuilder events and serialization boundaries. These upstream interfaces constrain the refactor; they do not establish blanket compatibility.

| Removed implementation | Retained behavior / acceptance owner |
|---|---|
| Runtime TOML scheduling knobs | ResourceProfile / ParallelOptions; removed keys fail with migration guidance; benchmark presets move to benchmark configuration |
| Query policy forwarding and unreachable Tor clamps | Canonical policy functions; explicit options > profile > stage; host admission, Tor transport tests |
| Two unused session-ownership wrappers | Active opener adoption and borrowed/owned lifecycle contracts |
| Query `_xml_attr`, `_append_join_xml` | Canonical QuerySpec serializer and XML roundtrip tests |
| Duplicate helper account setup | `_HelperState` transaction; independent module state, rollback, error text and interruptions |
| Duplicate response close/context | One response owner; stream/body readers retain their distinct contracts |
| Duplicate count/result request setup | Existing executor plus explicit custom provider adapter; overridden params/path/results/count remain observable |
| Internal DOM parsing | Bounded Expat events and ElementTree; descendant constraints, node paths, empty values, CDATA and Template metadata tests |
| Superseded reports, databases and ledgers | Current compatibility guide, immutable historical inventory, migrated Parquet measurements and portable DuckDB views |
| Old test administration assertions | Portable evidence/data tests, full public contracts and real scheduler invariants |

The tracked file manifest records a disposition and destination for every baseline tracked file. Deletion follows migration in the same phase. Runtime token/LOC reductions are measured separately from repository/documentation bytes; documentation deletion does not shrink installed wheels. Public compatibility facades, constructors, cloning and documented conversion helpers remain.

Coverage is compared by behavior and qualified declarations when implementations move. Raw line IDs from the prior unchanged-source test consolidation cannot certify this refactor. Every changed subsystem must retain or replace its assertion owners and pass the full source, exact-head CI, installed-wheel and scoped historical API gates.

Phase 2: GitHits CPython 3.14.5 dataclass documentation confirms initialization/replacement semantics; resolution now reads frozen option/profile fields directly without constructing a second options snapshot. Added removed-key tests for all 11 keys, override/schema/cache behavior and explicit/profile/stage precedence across all three historical scheduler labels. The real scheduler/footer/memory/admission suites remain in the full run. Source: **2,749 passed in 17.18 seconds**; lint/actionlint pass. Private worker-forwarder assertion transferred to resolved-options behavior in `test_resource_runtime`.

Phase 3: GitHits CPython `v3.14.5` contextlib documentation (ExitStack recipes, lines 678 onward) supports the existing explicit owner/rollback design. Removed four unused methods, retained opener authentication adoption, shared helper configuration without sharing module account state, and factored response close/context ownership into a private base. Sized/decompressed stream reads remain distinct from whole-response reads. The duplicate borrowed-session assertion is now an alias to the same exactly-once implementation. Ownership/XML focused gate: 81 passed; historical helper gate: 115 passed.
