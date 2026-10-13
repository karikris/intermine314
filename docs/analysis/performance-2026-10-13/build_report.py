"""Build a source-bound performance review from Parquet, using DuckDB SQL.

The .duckdb file is a view catalog. Canonical observations, recommendations,
source references, inventory, prose and metrics remain in Parquet. Re-run this
catalog.py to open the database after moving the directory; views use relative paths.
"""
from __future__ import annotations

import ast
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import sysconfig
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from catalog import open_catalog
from final_evidence import outcomes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def save(name, records):
    keys = sorted({key for record in records for key in record})
    records = [{key: record.get(key) for key in keys} for record in records]
    pq.write_table(pa.Table.from_pylist(records), HERE / f'{name}.parquet', compression='zstd')


SOURCES = [
    ('S01', 'CPython 3.14.5 executors', '5607950ef232dad16d75c0cf53101d9649d89115',
     'https://github.com/python/cpython/blob/5607950ef232dad16d75c0cf53101d9649d89115/Doc/library/concurrent.futures.rst',
     '49-83, 199-209, 246-325',
     'map(buffersize=...) bounds submitted results not yet yielded; interpreter pools isolate mutable state and pickle tasks/results; default thread counts use process-visible CPUs.', 'GitHits grep/read'),
    ('S02', 'CPython free threading', '5607950ef232dad16d75c0cf53101d9649d89115',
     'https://github.com/python/cpython/blob/5607950ef232dad16d75c0cf53101d9649d89115/Doc/howto/free-threading-python.rst',
     '43-91, 122-156',
     'Python 3.14 does not imply a disabled GIL. Unsupported extensions can re-enable it; shared iterators are unsafe; use explicit synchronization and check runtime GIL state after imports.', 'GitHits read'),
    ('S03', 'Polars multiprocessing', '1bd8ec12f42d40fcec62badf32ef2177d2377d8d',
     'https://github.com/pola-rs/polars/blob/1bd8ec12f42d40fcec62badf32ef2177d2377d8d/docs/source/user-guide/misc/multiprocessing.md',
     '3-19, 70-79',
     'Polars already uses native parallelism; extra processes can compete for CPU/memory. Use spawn when processes are actually needed with Polars.', 'GitHits search'),
    ('S04', 'Polars thread pool', '1bd8ec12f42d40fcec62badf32ef2177d2377d8d',
     'https://github.com/pola-rs/polars/blob/1bd8ec12f42d40fcec62badf32ef2177d2377d8d/py-polars/src/polars/meta/thread_pool.py',
     '21-39', 'POLARS_MAX_THREADS is a process-start configuration; the pool cannot be resized after initialization.', 'GitHits grep'),
    ('S05', 'DuckDB row groups and file sizes', '70642ff90220c13e7aa6749323bcab8ebd213f66',
     'https://github.com/duckdb/duckdb-web/blob/70642ff90220c13e7aa6749323bcab8ebd213f66/docs/current/guides/performance/file_formats.md',
     '35-66', 'Parquet scans parallelize over row groups/files. Guidance favors 100K-1M rows/group and enough groups for the desired threads. Small groups carry overhead.', 'GitHits grep'),
    ('S06', 'DuckDB memory and tuning', '70642ff90220c13e7aa6749323bcab8ebd213f66',
     'https://github.com/duckdb/duckdb-web/blob/70642ff90220c13e7aa6749323bcab8ebd213f66/docs/current/guides/performance/environment.md',
     '19-49', 'Memory depends on query shape; documentation suggests 1-4 GB/thread for ideal analytical workloads. This is guidance, not a mandatory allocation. Grouping/joins may spill.', 'GitHits read'),
    ('S07', 'DuckDB ordering and profiling', '70642ff90220c13e7aa6749323bcab8ebd213f66',
     'https://github.com/duckdb/duckdb-web/blob/70642ff90220c13e7aa6749323bcab8ebd213f66/docs/current/guides/performance/how_to_tune_workloads.md',
     '18-44, 52-100', 'Disabling insertion-order preservation can reduce memory where ordering is not required; excessive threads can hurt. EXPLAIN ANALYZE identifies expensive operators.', 'GitHits read'),
    ('S08', 'DuckDB per-thread connection', '70642ff90220c13e7aa6749323bcab8ebd213f66',
     'https://github.com/duckdb/duckdb-web/blob/70642ff90220c13e7aa6749323bcab8ebd213f66/docs/current/guides/python/multiple_threads.md',
     '37-43', 'Concurrent Python callers need their own cursor/connection. Query-internal engine threads are different from multiple Python query callers.', 'GitHits search'),
    ('S09', 'Arrow interchange', 'beccec0d0c451b7aa3e4530416ac431b3c035c69',
     'https://github.com/apache/arrow/blob/beccec0d0c451b7aa3e4530416ac431b3c035c69/docs/source/python/extending_types.rst',
     '24-70', 'Arrow C Data/PyCapsule protocols allow compatible buffers to cross language boundaries without Python row objects. Parsing JSON and decompressing Parquet still allocate.', 'GitHits read'),
    ('S10', 'Arrow scanner read-ahead', 'beccec0d0c451b7aa3e4530416ac431b3c035c69',
     'https://github.com/apache/arrow/blob/beccec0d0c451b7aa3e4530416ac431b3c035c69/python/pyarrow/_dataset.pyx',
     '299-363', 'CPU threading and batch/file read-ahead are independent controls. More read-ahead consumes more memory; configure both CPU and I/O pools.', 'GitHits grep/read'),
    ('S11', 'orjson concurrency limits', '6737895a',
     'https://github.com/ijl/orjson/blob/6737895a/README.md', '35-38, 165, 614',
     'Version 3.12.0 holds the GIL during encoding/decoding and documents no subinterpreter support. Free-threading is not certified here.', 'GitHits grep'),
    ('S14', 'Requests pool defaults and session state', '6e83187b',
     'https://github.com/psf/requests/blob/6e83187b/src/requests/adapters.py',
     '79, 206; sessions.py 797-799', 'Adapter pooling defaults to nonblocking. Sessions also carry mutable cookies/configuration; a thread-safe connection pool alone is not a guarantee for arbitrary shared session mutation.', 'GitHits grep'),
    ('S15', 'urllib3 connection capacity', 'b1d30ab6',
     'https://github.com/urllib3/urllib3/blob/b1d30ab6/src/urllib3/connectionpool.py',
     '145-156', 'pool_maxsize limits retained connections when block=False; use a separate host admission limit or carefully managed blocking pools to bound simultaneous requests.', 'GitHits grep'),
    ('S16', 'Intel 265K hardware', 'vendor page accessed 2026-10-13',
     'https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/core-ultra-desktop-processors-series-2-brief.html',
     'Processor Cores (P+E) table', '265K has 20 cores: 8 performance cores and 12 efficiency cores. The measured WSL guest exposes 18 CPUs, without a reliable P/E mapping.', 'Official Intel web search'),
    ('S17', 'Current CPython 3.14 patch release', '8e6e75d9',
     'https://github.com/python/cpython/blob/8e6e75d9/Include/patchlevel.h',
     '20-27', 'The verified v3.14.8 tag identifies Python 3.14.8; minimum 3.14.5 and current 3.14 are separate installation lanes.', 'GitHits read'),
    ('S18', 'Matplotlib rendering ownership', 'd3ca9172e392048ec242b90e8423e88d1a3000b2',
     'https://github.com/matplotlib/matplotlib/blob/d3ca9172e392048ec242b90e8423e88d1a3000b2/galleries/users_explain/figure/interactive_guide.rst',
     '380-390', 'GUI updates generally require the main thread, and Matplotlib is not thread-safe. Native aggregation can happen upstream, with rendering serialized.', 'GitHits read'),
]


def opportunities():
    # Priority is an engineering judgment, not a measured probability.
    rows = []

    def add(id, priority, section, line, stage, change, likelihood, workers, memory,
            retire, retain, risk, effort, evidence, metrics):
        rows.append(dict(id=id, priority=priority, file='src/intermine314/' + section,
                         line=line, stage=stage, proposal=change, gain_likelihood=likelihood,
                         worker_guidance=workers, memory_effect=memory,
                         potentially_redundant=retire, must_retain=retain,
                         compatibility_risk=risk, effort_days_estimate=effort,
                         evidence_ids=evidence, decision_metrics=metrics))

    add('P01',1,'export/parquet.py',139,'Parquet compaction',
        'Expose DuckDB thread/memory budgets; allow unordered compaction only when requested. Separate row-group size from producer batch size.',
        'High: directly measured native COPY scaling', 'Ordered: start 4; unordered: 8 balanced / 16 throughput',
        'Ordered 8-16 threads can retain much more data; preserve a low-memory 1-thread mode.',
        'Hard-coded threads=1/128MB as the only mode; batch_size used as output row-group size.',
        'Atomic publish, numeric-part ordering, timestamp/decimal validation, requested ordering.',
        'Medium', '2-4', ['S05','S06','S07'], ['copy throughput','peak RSS','spill bytes','ordered output equality'])
    add('P02',1,'export/model_schema.py',59,'Typed batch conversion',
        'Build typed Arrow RecordBatches directly; vectorize validation, then bridge to Polars/DuckDB. Add a supported-schema fast path.',
        'High for flat typed rows; prototype is not a semantics-equivalent replacement', '1 batch coordinator; native CPU pool 4-8 while fetching',
        'Arrow fixture buffers were 72% smaller than the Python graph, but buffers coexist during conversion.',
        'Per-column Python gathers and Python Series iteration after equivalent native validation exists.',
        'Exact Decimal scale/overflow, mixed numeric checks, dates, nested values and fallback path.',
        'High', '5-10', ['S09'], ['rows/s','allocations/row','Arrow bytes/row','edge-case parity'])
    add('P03',1,'export/parquet.py',257,'Storage layout',
        'Expose the existing partitioned mode in high-level fetch/dataframe helpers; accumulate typed batches into 122880-row groups and ~128-512 MiB files, subject to bytes budgets.',
        'High for eliminating repeated I/O; row-group gains measured locally', '1 publisher; 1-2 file writers sharing a native budget',
        'Bound the row-group accumulator in bytes, not just rows; wide rows need smaller groups.',
        'Mandatory part-to-single-file recompression and small-file churn on partitioned workflows.',
        'Single-file opt-in, all-pages-success barrier, rollback and schema unification.',
        'Medium', '3-6', ['S05','S07'], ['row groups','file count','write amplification','compression ratio','scan time'])
    add('P04',1,'query/parallel_offset.py',25,'Remote page fetching',
        'Tune workers per host and query stage; share a global host semaphore across concurrent exports; decrease admission on throttling.',
        'High when network latency dominates; low after Python decode saturation', 'Public mines start 4; test 8; 12-16 only with evidence/server capacity; Tor 2',
        'Keep pending, buffered and emitting pages charged; pipeline byte cap remains essential.',
        'Uniform default of 16 for every server/workload; disconnected benchmark-only host presets.',
        'Terminal footer validation, retry ownership, request timeouts, count/page correctness.',
        'Medium', '3-5', ['S01','S14','S15'], ['rows/s','p50/p95 page latency','429/5xx rate','retries','queue wait'])
    add('P05',1,'query/builder.py',1431,'Batch hand-off',
        'Add an internal validated-page/Arrow-batch stream with a bounded 2-batch queue; overlap fetch with native conversion/write.',
        'High potential when stage timings show fetch/write bubbles; unmeasured end-to-end', '1 scheduler + 1 converter + 1 writer; fetch pool 4-8',
        'Transfer buffer ownership and charge batches until consumed; avoid row-list duplication.',
        'Page list -> per-row yield -> iter_batches reassembly on analytical exports.',
        'Public row iterators, close/cancellation, late failures and atomic final publication.',
        'High', '5-8', ['S01','S09'], ['stage idle time','queue depth','end-to-end rows/s','peak RSS'])
    add('P06',1,'export/query.py',53,'Local analytical queries',
        'Expose DuckDB threads/memory/temp settings and an owned Arrow-reader context; retain materialized Polars convenience API.',
        'High for large SQL scans and bounded result consumption', 'DuckDB 16 balanced; 18 when isolated and enough groups; 4-8 for small inputs',
        'Streaming avoids materializing every selected row in to_arrow_table(); query state may still buffer.',
        'Mandatory full Arrow-table materialization for consumers that can stream.',
        'Connection ownership, local results view, schema guards, materialized API contract.',
        'Medium', '2-4', ['S05','S06','S08','S09'], ['time-to-first-batch','result bytes','query plan','peak RSS'])
    add('P07',1,'export/resource_profile.py',16,'Package resource policy',
        'Extend ResourceProfile into explicit network/native/memory budgets and record effective counts at job start.',
        'High for preventing oversubscription; tuning benefit is workload dependent', 'Default advisory native budget 16; profiles 8/6/4/2; clip automatic allocations to available CPUs',
        'Reserve separate queue, writer and engine memory; DuckDB memory_limit is not total process RSS.',
        'Scattered magic worker defaults and independent unconstrained engine creation.',
        'Explicit user overrides, tor_low_mem, lazy imports, existing environment configuration.',
        'Medium', '3-5', ['S01','S04','S06','S10'], ['active threads','CPU seconds','throughput/core','RSS','spill'])
    add('P08',2,'query/builder.py',1221,'Query preparation per page',
        'Freeze a deep execution snapshot once per export; pre-encode constant XML/form fields and vary start/size.',
        'Medium: repeated clone/serialization is visible in every page path; time savings unmeasured', '1 preparation task; share immutable bytes',
        'Avoid repeated deep copies and temporary XML nodes.',
        'Per-page Query.clone -> to_spec -> XML encoding for identical query state.',
        'Public clone semantics; QuerySpec contains mutable constraint objects despite frozen dataclass.',
        'Medium', '2-4', ['S01'], ['preparation ms/page','allocations/page','XML equivalence','mutation isolation'])
    add('P10',2,'query/inflight.py',86,'Memory accounting',
        'Use Arrow batch nbytes for native payloads, calibration factors for Python overhead, and explicit byte-bounded stage queues.',
        'High for predictable memory; no direct speed guarantee', 'No new pool',
        'Current sampled payload estimate was about 51% of the fixture Python graph; neither is total RSS.',
        'Recursive scalar sampling on Arrow-native batches.',
        'Reservations for completed pages, unknown/oversized-page behavior and cancellation accounting.',
        'Medium', '2-4', ['S01','S09','S10'], ['estimated/actual bytes','peak outstanding pages','RSS overshoot','backpressure'])
    add('P11',2,'export/parquet.py',345,'Schema promotion pass',
        'Resolve stable model schemas before writing; validate supported values in native kernels; rewrite only promoted parts.',
        'Medium/high when many parts change type; low for stable schemas', '2-4 independent part rewrites with byte admission',
        'Concurrent rewrites multiply frame memory; use one part per permit.',
        'Second-pass reads/rewrites for fixed-schema outputs; repeated per-file checks if a validated manifest is reusable.',
        'Dynamic decimal scale, null-first batches, mixed types, guarded schema evolution.',
        'High', '4-7', ['S05','S09'], ['parts rewritten','bytes reread','schema drift','validation failures'])
    add('P12',2,'export/parquet.py',357,'Publication copy',
        'Stage directly beside the destination when feasible; move already validated parts into publication rather than copying the entire dataset.',
        'High I/O reduction for large local datasets; physical-disk speedup unmeasured', '1 commit/publish owner; 1-2 copies only across filesystems',
        'Reduces duplicate scratch capacity; atomic publication still requires temporary ownership.',
        'Unconditional shutil.copytree(parts, staged) on the same filesystem.',
        'Cross-device fallback, unrelated entries, directory exchange and rollback tests.',
        'Medium', '2-3', ['S05'], ['bytes copied','scratch high-water','publish latency','failure recovery'])
    add('P13',2,'export/csv.py',130,'CSV ingestion',
        'Prefer path-based lazy scan/sink. Add bounded stream-to-Parquet ingestion for large nonseekable inputs; project columns early.',
        'High memory benefit for streams; native path already parallel', 'One Polars pool of 8-12; multiple files 2-4 only within total budget',
        'Current nonseekable read() materializes the entire CSV; avoid full BytesIO/StringIO snapshots.',
        'Whole-stream snapshots and intermediate Parquet re-read in directory CSV export where a streaming sink suffices.',
        'CSV dialect/identifier semantics, collisions, borrowed stream ownership and all-or-nothing publication.',
        'Medium', '3-5', ['S03','S04','S09','S10'], ['peak RSS','input bytes/s','projected columns','spill'])
    add('P14',2,'lists/_csv.py',16,'Identifier uploads',
        'Validate/quote the identifier column with columnar kernels; add a bounded upload-body API after checking server support.',
        'High memory benefit for very large lists; remote upload speed is server limited', '1 upload owner; native validation 4-8',
        'Current Arrow batches become a growing Python list plus a joined string before upload.',
        'to_pylist loop, full identifiers list and duplicate joined body in a new streaming API.',
        'Leading zeros, quote escaping, null/empty/CRLF rejection before mutation, single mutation attempt.',
        'High', '4-7', ['S09','S15'], ['body bytes','peak RSS','serialization ms','request count'])
    add('P15',2,'service/service.py',848,'Metadata and registry discovery',
        'Fetch independent mine/metadata resources concurrently after adding single-flight cache initialization and safe client lifetimes.',
        'Medium/high for cold multi-mine discovery; low for cached single-mine access', '2-4 threads across independent resources/mines',
        'Do not retain every model/template graph indefinitely; bounded caches with active leases.',
        'Repeated cold fetches and duplicate DOM/model parsing during races.',
        'Service/Registry cache semantics, no eviction/close of in-use clients, authentication boundaries.',
        'Medium', '3-5', ['S01','S02','S14'], ['cold startup','cache hit rate','duplicate requests','live clients'])
    add('P16',3,'model.py',545,'XML/model construction',
        'Investigate streaming XML parsing and memoized ancestry; prefer fewer passes over parallel mutation of the model graph.',
        'Medium for unusually large models; low for cached typical models', '1 model owner; parallelize different models only',
        'DOM plus source XML plus descriptors coexist; streaming could lower the peak.',
        'Duplicate XML trees, repeated ancestry traversal, unconditional debug f-string work.',
        'Descriptor identities, inheritance/cycle checks, reverse references and exact model errors.',
        'High', '4-8', ['S02'], ['parse wall/CPU time','model nodes','peak RSS','graph parity'])
    add('P17',2,'_result_object.py',102,'Lazy object hydration',
        'Add opt-in bulk prefetch/select of missing attributes with ID batching; execute a few independent requests concurrently.',
        'High for N+1 access patterns, otherwise no gain', '2-4 request workers; prefer one bulk query over many requests',
        'Batch and evict hydrated objects; identity maps can grow.',
        'One _fetch query per object/field when a bulk path is selected.',
        'Lazy legacy behavior, identity map, selected nulls, relation cardinality and model binding.',
        'High', '5-8', ['S01','S15'], ['requests per object','p95 hydration time','cache size','exact results'])
    add('P18',3,'idresolution.py',47,'Resolution jobs',
        'Provide a coordinator that schedules due polls across independent existing jobs instead of occupying a thread per sleep.',
        'Medium for many jobs; no benefit for one job', '2-4 I/O workers or one asynchronous timer coordinator',
        'Avoid one sleeping thread per job.',
        'Repeated external loops that block in each Job.poll sequentially.',
        'Server backoff, terminal-state behavior, job creation/deletion ownership and no mutation retry.',
        'Medium', '2-4', ['S01','S15'], ['jobs completed/min','poll requests','sleeping threads','tail latency'])
    add('P19',3,'lists/_enrichment.py',52,'Enrichment analytics',
        'Offer persisted Parquet/reader return modes and optional concurrency across independent read-only enrichments.',
        'Medium on large/multiple enrichments; small outputs rarely benefit', '2-4 network tasks; one native aggregation pool',
        'Current persist() reloads the entire completed result as a DataFrame.',
        'Mandatory query_parquet(staged) materialization for path-only consumers.',
        'Existing DataFrame return contract, schema/precision and atomic output.',
        'Medium', '2-4', ['S08','S09'], ['rows/s','materialized bytes','peak RSS','request errors'])
    add('P20',3,'service/transport.py',146,'HTTP pool coordination',
        'Coordinate pool retention with host permits and isolate mutable session configuration/cookies; reuse workers across jobs only with explicit ownership.',
        'Medium for many concurrent exports; pool reuse already exists', 'Per-host cap 4-8; global request cap 16 initially',
        'Avoid leaked sessions/adapters and unbounded simultaneous sockets.',
        'Assumption that pool_maxsize=32 bounds requests; per-export unconstrained pool creation.',
        'Caller sessions unchanged, per-request retry contexts, proxy/TLS and close-once contracts.',
        'High', '3-6', ['S14','S15'], ['open sockets','pool wait','TLS handshakes','retries','resource leaks'])
    add('P21',3,'export/query.py',22,'Dataset metadata discovery',
        'Persist a Parquet dataset manifest with schema hash, file metadata, row counts and provenance; cache only with reliable invalidation.',
        'Medium for thousands of parts and repeated queries', '2-4 metadata readers for cold remote files; 1 for small local sets',
        'Compaction reduces metadata objects and open-file pressure.',
        'Repeated glob enumeration and schema opens on every query against unchanged data.',
        'Literal path escaping, schema guards, stale-manifest detection and numeric part order.',
        'Medium', '2-4', ['S05','S10'], ['metadata open count','time to plan','manifest freshness','file count'])
    add('P22',3,'parallel/policy.py',45,'Scheduler simplification',
        'Unify profile resolution and worker telemetry. Evaluate Python 3.14 map(buffersize) only for simple count-bounded jobs.',
        'Low direct throughput gain; useful maintenance simplification', 'No extra workers',
        'buffersize is a task/result count, not a bytes envelope.',
        'Duplicate mode normalization, inert auto-to-offset branch, overlapping prefetch policy if APIs permit deprecation.',
        'Custom scheduler byte accounting, unordered completion, page diagnostics and failure cleanup.',
        'Medium', '2-3', ['S01'], ['policy branches','configuration clarity','queue invariants'])
    add('P23',3,'util/json.py',63,'Residual Python CPU work',
        'Use coarse spawn-process tasks only for proven CPU-heavy pure-Python transforms; pass Parquet paths and return small manifests.',
        'Low for current per-row JSON; high only for coarse independent pure-Python tasks', 'Start 4-8 processes; native pools 1 each; test 16 for long tasks',
        'Each process duplicates interpreter state; serialization can dominate and child RSS must be measured.',
        'Fine-grained Python executor maps over individual rows.',
        'Exact-integer fallback, no live Service/Session/DuckDB connections across processes.',
        'High', '3-6', ['S01','S03','S11'], ['compute/serialization ratio','worker RSS sum','startup time','task duration'])
    add('P24',3,'bar_chart.py',151,'Plots and helper modules',
        'Aggregate/filter large inputs in DuckDB/Polars before plotting; keep rendering and mutable helper state under one owner.',
        'Low: GO helpers already plot only five rows; large custom plots benefit from preaggregation', '1 rendering thread; native engine for preprocessing',
        'Return only chart-sized data to Python.',
        'Whole-response tab splitting in get_query for new analytics APIs, not historical helpers.',
        'Legacy helper outputs, separate account state, plot semantics and public facade bindings.',
        'Medium', '1-3', ['S03','S08'], ['rows before/after aggregate','plot time','materialization size'])
    add('P25',3,'lists/listmanager.py',99,'Mutations and control plane',
        'Keep list-name allocation, shared-list edits, account mutations and final publication serialized per owner/target.',
        'Low or negative benefit from parallelism', '1 mutation owner',
        'Concurrency adds state and failure complexity without local analytical gain.',
        'None solely for performance; consolidate common validation only when contracts match.',
        'Mutation retry safety, temporary list tracking, cleanup and legacy call semantics.',
        'High', 'Not a parallelization target', ['S02','S14'], ['duplicate mutations','state races','cleanup correctness'])
    add('P26',1,'unused',410,'Benchmark reliability',
        'Update or retire the old phase0 baseline entrypoint; unify performance evidence in Parquet and use DuckDB to compare stage-specific, correctness-checked runs.',
        'High decision-quality benefit; no direct package throughput gain', 'Run competing benchmark cases sequentially; parallelize independent CI jobs only',
        'Record total process-tree RSS and spill, not just batch memory or parent high-water.',
        'Stale joins/workflow/resource_profile/ordered_window_pages call arguments; duplicate JSON-oriented measurement pipelines after migration.',
        'Historical evidence and original-client isolation; distinguish COUNT metadata scans from actual data scans.',
        'Low', '2-4', ['S01','S05'], ['benchmark completion','reproducibility','source/data hashes','stage attribution'])
    rows[-1]['file'] = 'benchmarks/runners/phase0_baselines.py'
    return rows


def inventory():
    tracked = subprocess.check_output(['git','ls-files','-z'], cwd=ROOT).decode().split('\0')
    files, functions = [], []
    for name in filter(None, tracked):
        path = ROOT / name
        if not path.is_file():
            continue
        content = path.read_bytes()
        runtime = name.startswith('src/intermine314/')
        category = name.split('/')[0]
        stage = ('transport/results' if '/service/' in name else 'analytics/export' if '/export/' in name
                 else 'query construction/execution' if '/query/' in name else 'lists/enrichment' if '/lists/' in name
                 else 'policy/utilities' if '/config/' in name or '/util/' in name or '/parallel/' in name
                 else 'legacy facade/model/helpers' if runtime else category)
        record = dict(file=name, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
                      runtime=runtime, category=category, stage=stage, python=name.endswith('.py'),
                      review_scope='AST inventory plus targeted call-path inspection' if runtime else 'supporting code/data inventory',
                      lines=len(content.splitlines()), functions=0, loops=0, calls=0)
        if name.endswith('.py'):
            tree = ast.parse(content, filename=name)
            nodes = list(ast.walk(tree))
            record.update(functions=sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in nodes),
                          loops=sum(isinstance(n,(ast.For,ast.While,ast.comprehension)) for n in nodes),
                          calls=sum(isinstance(n,ast.Call) for n in nodes))

            def walk(node, prefix=''):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                        label = prefix + child.name
                        if not isinstance(child, ast.ClassDef):
                            calls = [ast.unparse(n.func) for n in ast.walk(child) if isinstance(n, ast.Call)]
                            functions.append(dict(file=name, symbol=label, line=child.lineno,
                                                  end_line=child.end_lineno, span_lines=child.end_lineno-child.lineno+1,
                                                  runtime=runtime, stage=stage, lexical_calls=calls,
                                                  loops=sum(isinstance(n,(ast.For,ast.While,ast.comprehension)) for n in ast.walk(child))))
                        walk(child, label + '.')
                    else:
                        walk(child,prefix)
            walk(tree)
        files.append(record)
    save('source_inventory', files)
    save('function_inventory', functions)
    return files


WORKER_PLANS = [
    dict(stage='Public mine HTTP', current='Profile default 4; shared per-host/global admission', start_workers=4, sweep='1,2,4,8; then 12,16 only if allowed and useful', native_threads=4, arrow_io_threads=2, inflight='workers pages + byte cap; avoid blindly doubling prefetch', rationale='Public-server optimum remains unmeasured. Default 4 is conservative; increase only with permission and server evidence.'),
    dict(stage='Low-latency fetch/decode', current='Low-latency profile 2; bounded validated pages', start_workers=2, sweep='2,4,8', native_threads=4, arrow_io_threads=2, inflight='2-4 pages', rationale='Final 5 ms HTTP-to-Parquet sweep selected 2 within 5% of the best median; extra fetch workers give little benefit.'),
    dict(stage='High-latency capable server', current='High-latency profile 8; explicit host override required', start_workers=8, sweep='8,12,16,18', native_threads=4, arrow_io_threads=2, inflight='one page/worker initially; byte capped', rationale='Final 30 ms pipeline selected 16; default remains 8. Loopback is not a certified public-mine cap.'),
    dict(stage='Ordered Parquet compaction', current='Ordered COPY 4; multi-file ordered COPY 1; memory 60% envelope', start_workers=4, sweep='1,2,4,8 with memory measurements', native_threads=4, arrow_io_threads=2, inflight='one COPY, no outer query pool', rationale='Final single-file writer selected 8 at 330 MiB RSS versus 245 MiB at 1; retain 4 as balanced default and 1 for low memory.'),
    dict(stage='Unordered Parquet compaction', current='Explicit ParquetOptions(preserve_order=False), COPY 8', start_workers=8, sweep='4,8,12,16,18', native_threads=8, arrow_io_threads=2, inflight='one COPY; preserve_insertion_order=false only when valid', rationale='Final writer selected 12 within 5% of best median; 8 remains a conservative memory/CPU default.'),
    dict(stage='DuckDB SQL over Parquet', current='Owned connections use shared budget 16; borrowed settings retained', start_workers=16, sweep='4,8,12,16,18,20 on bare host', native_threads=16, arrow_io_threads=2, inflight='one main query; enough total row groups', rationale='Final integer aggregation selected 12; earlier different SQL favored 18. Default 16 is a budget, not a universal optimum.'),
    dict(stage='Polars scan/aggregate', current='Lazy engine configuration requests 12; existing pool reported', start_workers=12, sweep='8,12,16,18,20 on bare host', native_threads=12, arrow_io_threads=2, inflight='one streaming plan; set POLARS_MAX_THREADS before import', rationale='Final integer aggregation selected 8; earlier floating aggregation favored 12-16. Keep one streaming plan and configure before import.'),
    dict(stage='Pure-Python coarse transforms', current='transform_parquet: spawn, paths, bounded queue, default 4', start_workers=4, sweep='4,8,12,16 after measuring IPC and startup', native_threads=1, arrow_io_threads=1, inflight='1-2 coarse path-based tasks/worker', rationale='Final coarse callback selected 16 but summed RSS reached 2.5 GiB; default 4 uses much less memory. No subinterpreter certification.'),
    dict(stage='Metadata / independent mines', current='Per-key lazy initialization and Registry leases; bounded opt-in prefetch', start_workers=4, sweep='2,4; global and per-host permits', native_threads=1, arrow_io_threads=1, inflight='one initialization per cache key', rationale='I/O overlap helps cold discovery; protect cache and Service lifetimes first.'),
    dict(stage='Mutation / publish / plots', current='Explicit resource owners and serialized publication/rendering', start_workers=1, sweep='1', native_threads=1, arrow_io_threads=1, inflight='one owner per mutable resource', rationale='Correctness and tiny work units dominate; parallelize independent reads upstream instead.'),
]


LANGUAGES = [
    dict(technology='Arrow + DuckDB/Polars', priority=1, recommendation='Use existing native engines first', likely_gain='High for conversion, grouping, projection, compression and scans', boundary='RecordBatch/Arrow C Stream in memory; Parquet on disk; DuckDB executes SQL', cost='No new language/toolchain; add APIs and retain validation', evidence=['S03','S05','S09']),
    dict(technology='Python process pools', priority=2, recommendation='For coarse independent pure-Python CPU tasks only', likely_gain='Toy CPU benchmark scales; row-level JSON threading does not', boundary='Pass Parquet paths; separate DuckDB connection per worker; native threads=1', cost='Interpreter memory and serialization; use spawn with Polars', evidence=['S01','S03']),
]


def build_catalog():
    con = open_catalog(HERE, read_only=False)
    for path in sorted(HERE.glob('*.parquet')):
        con.read_parquet(path.name).create_view(path.stem, replace=True)
    con.execute("""CREATE OR REPLACE VIEW all_measurements AS
        SELECT *, 'pilot' AS series, false AS decision_eligible FROM benchmark_pilot
        UNION ALL BY NAME
        SELECT *, 'initial' AS series, stage IN ('python_cpu','json_decode','export_writer') AS decision_eligible FROM benchmark_runs
        UNION ALL BY NAME
        SELECT *, 'validation' AS series, true AS decision_eligible FROM benchmark_validation""")
    con.execute('CREATE OR REPLACE VIEW selected_measurements AS SELECT * FROM all_measurements WHERE decision_eligible')
    con.execute("""CREATE OR REPLACE VIEW measured_summary AS
        SELECT stage,variant,workers,native_threads,delay_ms,row_group_size,ordered,
               output_row_group_size,batch_size,single_file, count(*) AS trials,
               median(wall_s) AS median_s,min(wall_s) AS minimum_s,max(wall_s) AS maximum_s,
               stddev_samp(wall_s)/avg(wall_s)*100 AS cv_pct,
               median(rows_per_s) AS units_per_second,
               CASE WHEN stage='python_cpu' THEN 'iterations/s' ELSE 'rows/s' END AS throughput_unit,
               median(parent_cpu_cores) AS parent_cpu_cores,
               max(sampled_peak_rss_bytes) AS peak_rss_bytes,
               max(sampled_rss_delta_bytes) AS peak_rss_delta_bytes,
               median(output_bytes) AS output_bytes,median(write_syscall_bytes) AS write_syscall_bytes,
               median(input_bytes) AS input_bytes
        FROM selected_measurements GROUP BY ALL""")
    con.execute("COPY (SELECT * FROM measured_summary ORDER BY stage,variant,workers) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(HERE / 'benchmark_summary.parquet')])
    con.read_parquet('benchmark_summary.parquet').create_view('benchmark_summary',replace=True)
    con.execute("""CREATE OR REPLACE VIEW scaling AS
        SELECT candidate.stage,candidate.variant,candidate.workers,candidate.delay_ms,
               candidate.row_group_size,candidate.ordered,candidate.trials,candidate.median_s,
               baseline.median_s/candidate.median_s AS speedup_vs_one_worker,
               (baseline.median_s/candidate.median_s)/candidate.workers AS worker_efficiency,
               candidate.peak_rss_bytes, candidate.cv_pct
        FROM measured_summary candidate JOIN measured_summary baseline
          ON candidate.stage=baseline.stage AND candidate.variant=baseline.variant
          AND candidate.delay_ms IS NOT DISTINCT FROM baseline.delay_ms
          AND candidate.row_group_size IS NOT DISTINCT FROM baseline.row_group_size
          AND candidate.ordered IS NOT DISTINCT FROM baseline.ordered
          AND candidate.output_row_group_size IS NOT DISTINCT FROM baseline.output_row_group_size
          AND candidate.batch_size IS NOT DISTINCT FROM baseline.batch_size
          AND candidate.single_file IS NOT DISTINCT FROM baseline.single_file
          AND baseline.workers=1
        WHERE candidate.stage IN ('http_fetch','duckdb_query','duckdb_compact','polars_query','json_decode')""")
    con.execute("COPY scaling TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(HERE / 'worker_scaling.parquet')])
    con.read_parquet('worker_scaling.parquet').create_view('worker_scaling',replace=True)
    return con


def records(con, query):
    cursor = con.execute(query)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names,row)) for row in cursor.fetchall()]


def table(rows, fields):
    out = '<div class="table-scroll"><table><thead><tr>'
    out += ''.join('<th>' + html.escape(label) + '</th>' for key,label in fields) + '</tr></thead><tbody>'
    for row in rows:
        out += '<tr>'
        for key,label in fields:
            value = row.get(key)
            if value is None:
                value = '—'
            elif isinstance(value,float):
                value = f'{value:,.3f}'
            elif isinstance(value,list):
                value = ', '.join(map(str,value))
            out += '<td>' + html.escape(str(value)) + '</td>'
        out += '</tr>'
    return out + '</tbody></table></div>'


def main():
    implementation = outcomes()
    files = inventory()
    save('opportunities', opportunities())
    save('worker_plans', WORKER_PLANS)
    save('language_options', LANGUAGES)
    save('sources', [dict(id=i,title=t,revision=r,url=u,lines=lines,finding=f,method=m) for i,t,r,u,lines,f,m in SOURCES])
    versions = {p:importlib.metadata.version(p) for p in ['intermine314','polars','duckdb','pyarrow','orjson','requests','urllib3']}
    meminfo = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    metadata = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    generated_at=datetime.now(UTC).isoformat(),python=sys.version,
                    platform=platform.platform(),cpu_model='Intel Core Ultra 7 265K',physical_cores_spec=20,
                    physical_p_cores_spec=8,physical_e_cores_spec=12,visible_cpu_count=os.cpu_count(),
                    process_cpu_count=os.process_cpu_count(),affinity_count=len(os.sched_getaffinity(0)),
                    gil_enabled=sys._is_gil_enabled(),gil_disabled_build=bool(sysconfig.get_config_var('Py_GIL_DISABLED')),
                    mem_total_bytes=int(meminfo['MemTotal'].split()[0])*1024,
                    mem_available_at_report_bytes=int(meminfo['MemAvailable'].split()[0])*1024,
                    benchmark_scratch_filesystem='WSL /tmp tmpfs; warm cache; not physical NVMe throughput',
                    runtime_python_files=sum(r['runtime'] and r['python'] for r in files),
                    runtime_python_lines=sum(r['lines'] for r in files if r['runtime'] and r['python']),
                    tracked_files=len(files),versions=json.dumps(versions,sort_keys=True))
    save('environment',[metadata])
    con=build_catalog()
    runtime_coverage=records(con,"SELECT stage,count(*) AS modules,sum(lines) AS lines,sum(functions) AS functions FROM source_inventory WHERE runtime AND python GROUP BY stage ORDER BY stage")
    metrics=records(con,"""SELECT stage,variant,workers,delay_ms,row_group_size,ordered,batch_size,single_file,trials,
        round(median_s*1000,2) AS median_ms,round(minimum_s*1000,2) AS min_ms,round(maximum_s*1000,2) AS max_ms,
        round(cv_pct,1) AS cv_pct,round(units_per_second) AS units_per_second,throughput_unit,
        round(peak_rss_bytes/1048576,1) AS peak_rss_mib FROM measured_summary ORDER BY stage,variant,delay_ms,row_group_size,ordered,workers""")
    save('review_coverage',runtime_coverage)
    n=con.execute('SELECT count(*) FROM selected_measurements').fetchone()[0]
    all_n=con.execute('SELECT count(*) FROM all_measurements').fetchone()[0]
    # Comparison values are computed by DuckDB from the selected measurements.
    comparisons=records(con,"""WITH pairs AS (
      SELECT 'HTTP 5ms: 1 to 4 workers' AS label,
        (SELECT median_s FROM measured_summary WHERE stage='http_fetch' AND delay_ms=5 AND workers=1) baseline_s,
        (SELECT median_s FROM measured_summary WHERE stage='http_fetch' AND delay_ms=5 AND workers=4) candidate_s
      UNION ALL SELECT 'HTTP 30ms: 1 to 16 workers',
        (SELECT median_s FROM measured_summary WHERE stage='http_fetch' AND delay_ms=30 AND workers=1),
        (SELECT median_s FROM measured_summary WHERE stage='http_fetch' AND delay_ms=30 AND workers=16)
      UNION ALL SELECT 'DuckDB 122880-row groups: 1 to 18 threads',
        (SELECT median_s FROM measured_summary WHERE stage='duckdb_query' AND row_group_size=122880 AND workers=1),
        (SELECT median_s FROM measured_summary WHERE stage='duckdb_query' AND row_group_size=122880 AND workers=18)
      UNION ALL SELECT 'Polars streaming: 1 to 12 threads',
        (SELECT median_s FROM measured_summary WHERE stage='polars_query' AND workers=1),
        (SELECT median_s FROM measured_summary WHERE stage='polars_query' AND workers=12)
      UNION ALL SELECT 'Unordered COPY: 1 to 16 threads',
        (SELECT median_s FROM measured_summary WHERE stage='duckdb_compact' AND variant='copy_zstd' AND NOT ordered AND workers=1),
        (SELECT median_s FROM measured_summary WHERE stage='duckdb_compact' AND variant='copy_zstd' AND NOT ordered AND workers=16)
      UNION ALL SELECT 'Model conversion to typed Arrow prototype',
        (SELECT median_s FROM measured_summary WHERE variant='package_model'),
        (SELECT median_s FROM measured_summary WHERE variant='arrow_from_pylist_prototype')
      UNION ALL SELECT 'Inferred conversion to typed Arrow prototype',
        (SELECT median_s FROM measured_summary WHERE variant='package_inferred'),
        (SELECT median_s FROM measured_summary WHERE variant='arrow_from_pylist_prototype')
      UNION ALL SELECT 'JSON serial to 16 Python threads',
        (SELECT median_s FROM measured_summary WHERE stage='json_decode' AND variant='serial'),
        (SELECT median_s FROM measured_summary WHERE stage='json_decode' AND variant='thread' AND workers=16)
      UNION ALL SELECT 'Toy pure Python: serial to 16 interpreters (warm)',
        (SELECT median_s FROM measured_summary WHERE stage='python_cpu' AND variant='serial'),
        (SELECT median_s FROM measured_summary WHERE stage='python_cpu' AND variant='interpreter' AND workers=16)
      ) SELECT *,baseline_s/candidate_s AS local_speedup FROM pairs""")
    save('comparisons',comparisons)
    enriched = opportunities()
    measured_links = {'P01':4,'P02':5,'P04':1,'P06':2,'P23':8}
    for opportunity in enriched:
        comparison = comparisons[measured_links[opportunity['id']]] if opportunity['id'] in measured_links else None
        opportunity['measured_comparison'] = comparison['label'] if comparison else None
        opportunity['measured_stage_speedup'] = comparison['local_speedup'] if comparison else None
        opportunity['measurement_limit'] = ('Isolated stage only; prototype validation is incomplete for conversion; toy pure-Python scaling is not package scaling.'
                                            if comparison else 'No direct benchmark of the proposed change; mechanism/inspection only.')
    save('opportunities',enriched)
    con.read_parquet('opportunities.parquet').create_view('opportunities',replace=True)
    save('integration_findings',[
        dict(file='src/intermine314/export/resource_profile.py',line=101,finding='ResourceProfile is integrated through ResourceRuntime and public analytics entry points.',impact='Named cpu16/8/6/4/2 budgets configure native engines and default fetch limits; explicit overrides remain authoritative.',verification='tests/test_resource_profiles.py, test_resource_runtime.py and test_resource_integration.py'),
        dict(file='benchmarks/runners/phase0_baselines.py',line=413,finding='Implemented: obsolete fetch_from_mine call replaced with Service/Query/ParallelOptions APIs.',impact='Joins and profile budgets reach supported APIs; COUNT validates output only and is outside measured export time.',verification='tests/test_phase0_export_runner.py: gzip/zstd, joins, pagination and late failure preserve existing output'),
    ])
    con.read_parquet('integration_findings.parquet').create_view('integration_findings',replace=True)
    settings = [
        dict(scope='Process launch', setting='POLARS_MAX_THREADS', value='12', available_now=True, condition='Set before importing Polars. Use a smaller pool if overlapping with another CPU-heavy engine.'),
        dict(scope='Arrow', setting='set_cpu_count / set_io_thread_count', value='4 / 2', available_now=True, condition='Starting values for Arrow work while ingesting; sweep 4/8 when Arrow is the active engine.'),
        dict(scope='Query', setting='ParallelOptions', value='max_workers=4, page_size=2000, prefetch=4, inflight_limit=4, max_inflight_bytes_estimate=134217728', available_now=True, condition='Public-mine starting policy; preserve requested order and adjust only from server/latency evidence.'),
        dict(scope='Export', setting='Query.to_parquet', value='single_file=False, batch_size=65536, compression="zstd"', available_now=True, condition='Existing partitioned output avoids final single-file merge; batch-size tuning alone showed little speed gain.'),
        dict(scope='DuckDB query', setting='threads / memory_limit / temp_directory', value='16 / 60% of resource memory envelope / owned spill directory', available_now=True, condition='Owned connections use the profile. Borrowed connection settings are unchanged. Memory_limit is not a strict RSS cap.'),
        dict(scope='Parquet geometry', setting='independent row_group_size and file size', value='122880 rows / 256 MiB target', available_now=True, condition='ParquetOptions separates geometry from producer batch size and caps estimated row-group bytes.'),
        dict(scope='Global CPU policy', setting='effective CPU budget', value='Advisory profiles 16 (default), 8, 6, 4, 2; external oversized pools are reported', available_now=True, condition='ResourceRuntime coordinates package native stages. External pools remain advisory and are reported.'),
    ]
    save('suggested_settings',settings)
    con.read_parquet('suggested_settings.parquet').create_view('suggested_settings',replace=True)
    sections=[
        dict(id=1,title='Implemented performance uplift',body=f"All 25 included opportunities have implementation, replacement and regression-test evidence below; P09 remains excluded by the user. The code inventory covers {metadata['runtime_python_files']} runtime modules and {metadata['runtime_python_lines']:,} lines. Ten phases and 32 tasks are tracked with commit and CI evidence. Version remains 0.1.8; this work does not publish a release. Accepted status comes from the phase ledger, and the final evidence commit may follow the CI-verified implementation head."),
        dict(id=2,title='Hardware and resource policy',body='Intel Core Ultra 7 265K has 20 cores (8 P + 12 E); this WSL guest exposes 18 CPU/affinity slots. Default advisory native CPU budget is 16, with cpu8/cpu6/cpu4/cpu2 options and affinity/quota clipping. Native stages share admission instead of summing independent pools. Existing external pools are reported and may exceed the advisory budget. There is no verified P/E affinity map, bare-host 20-core sweep, or free-threaded/native-wheel certification. Python 3.14.5 local tests use the GIL; current Python 3.14 is verified separately in CI.'),
        dict(id=3,title='Current processing path',body='Immutable execution snapshot → shared host fetch permits → terminal-status-validated pages → typed Arrow conversion → bounded native write → validated Parquet parts and relative manifest → atomic publication → DuckDB SQL → owned Arrow reader or deliberate DataFrame materialization. Pending, buffered and emitting pages remain accounted until consumption. CSV compatibility APIs remain available, while durable analytics, manifests and evidence use Parquet. DuckDB holds the SQL catalog. XML/HTTP scratch, native spill and known-length upload staging are temporary, not alternative analytical stores.'),
        dict(id=4,title='Final measurements and tradeoffs',body='259 new trials cover 37 configurations, seven repetitions each, with full result-value equality on every run. Configurations run in fresh processes with one warmup and randomized case order. HTTP compares identical 32,000-row gzip fixtures against baseline 6a85ed7; final writer uses one million rows, SQL/Polars four million integer rows, and spawned transforms sixteen coarse path tasks. The smallest worker count within 5% of the best median is selected subject to summed process RSS <=4 GiB. These choices describe these fixtures, not revised universal defaults. At four HTTP workers the 5 ms pipeline median changes from 0.2201s to 0.2318s (5.3% slower); at 30 ms it changes from 0.4168s to 0.4137s. The small fast-fixture regression is retained alongside stronger terminal-status, schema, memory and publication checks; those checks add work, but this experiment does not isolate their causal contribution. At the recommended low-latency count of two, the current median is 0.2295s. New measurements never overwrite the 689 historical trials (482 selected).'),
        dict(id=5,title='Memory and uncertainty',body='Final RSS is sampled every 10ms across the process tree; summing resident sets counts shared pages once per process and is not unique PSS. CPU includes exited children. Peak includes retained setup/warmup/validation allocations; validation itself is outside timed actions. Seven repetitions support descriptive min/median/max/CV, not production p95 or a confidence interval. Spawn 16 reached 2588 MiB versus 775 MiB at four, while median time fell from 0.767s to 0.512s. Byte accounting remains an estimate, a first oversized page can exceed it, and DuckDB memory_limit is not a hard RSS limit. Relationship fanout and external pools require workload-specific monitoring.'),
        dict(id=6,title='Implemented conversion and metadata evidence',body='Seven paired production conversion trials preserve full-frame parity: inferred conversion median 0.09820s → 0.02710s (3.62x), model conversion 0.03724s → 0.02432s (1.53x). Seven paired 1000-class model trials preserve descriptor fingerprints: 0.03580s → 0.01137s (3.15x). Separate Python allocation peaks fall from 18,723,917 to 3,119,573 bytes; this is tracemalloc, not RSS/native memory. Phase 4 container accounting estimates 6,555,176 bytes versus a 4,643,400-byte unique Python graph on its flat fixture. These isolated improvements must not be multiplied into an end-to-end claim.'),
        dict(id=7,title='Compatibility and ownership coverage',body='Legacy imports under intermine314.webservice, Query/Template results and list/helper contracts remain covered by the compatibility corpus and installed-wheel binding checks. Users replace imports of intermine with intermine314; an intermine namespace is not installed. Existing DataFrame returns, lazy object fields, explicit DOM responses, raw chart cardinality and custom execution hooks retain distinct semantics. New reader/path/prefetch/spawn APIs are opt-in. Tests cover late server-failure rollback, compressed sized XML, ordered-buffer bounds, exact numeric/nested schema cases, reader cancellation, borrowed resource lifetimes and mutation ownership. No authenticated production writes were exercised.'),
        dict(id=8,title='Immediate replacements and retained contracts',body='Every included opportunity has a completed implementation and an explicit removal/retention decision below; there is no deferred retirement cycle. Normal discovery no longer builds a full DOM, export no longer reassembles individual page rows, CSV paths no longer snapshot entire streams, identifier uploads no longer accumulate all IDs, and publication shares one owner-aware transaction. Precision guards and semantically distinct legacy return/input contracts remain. Git history is the recovery mechanism for replaced code.'),
        dict(id=9,title='Limits of certification',body='Public-mine optima, WAN retries/tails, remote snapshot stability, server offset-query plans, physical disk saturation, energy use, Windows execution and bare 20-core affinity remain unmeasured. Remote pagination still needs stable server ordering/data. Manifest fields are relative paths, size/mtime, row counts/groups, serialized schema and producer order; server releases, query hashes and content checksums are not promised. External files are validated; caches do not establish snapshot isolation. GUI plotting belongs on the main thread. Transform callback side effects outside assigned outputs are not transactional. Cross-process publication needs application coordination.'),
        dict(id=10,title='Reproduction and phase handshakes',body='Use catalog.open_catalog(directory) for portable relative Parquet views in analysis.duckdb. phase10_benchmark.py reproduces the final sweeps; --baseline-src must point to src extracted from baseline Git revision 6a85ed7, and --resume continues completed configurations. phase8_xml_benchmark.py reproduces model comparisons. Each phase handoff records behavior, removals, verification, source evidence, integration and limits. Successful push/CI acceptance is recorded in phase_status.parquet. scripts/verify_distribution.py supports minimum constraints and fresh outside-checkout base/extras/sdist installs; CI uses both minimum and current Python 3.14. Durable evidence is Parquet and report.html is derived.'),
    ]
    save('report_sections',sections)
    for name in ['review_coverage','comparisons','report_sections']:
        con.read_parquet(f'{name}.parquet').create_view(name,replace=True)
    phase_status = records(con, 'SELECT * FROM phase_status ORDER BY phase')
    phase_tasks = records(con, 'SELECT * FROM implementation_tasks ORDER BY phase, task_id')
    planned_profiles = records(con, 'SELECT * FROM planned_worker_profiles ORDER BY stage, cpu_budget DESC')
    final_trials = records(con, 'SELECT * FROM phase10_summary ORDER BY stage,delay_ms,implementation,workers')
    choices = records(con, 'SELECT * FROM phase10_worker_choices ORDER BY stage,delay_ms')
    distributions = records(con, 'SELECT * FROM distribution_matrix ORDER BY lane,mode')
    con.execute('CHECKPOINT')
    con.close()
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>InterMine314 performance review — 265K</title><style>
    body{font:16px/1.55 system-ui,sans-serif;margin:0;color:#19283a;background:#f5f7fa}main{max-width:1400px;margin:auto;padding:32px}h1{font-size:32px}h2{margin-top:36px}p{max-width:1050px}a{color:#155ba4}.table-scroll{overflow:auto;margin:20px 0;background:white;border:1px solid #cad4e0;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid #dde4ec;min-width:90px}th{background:#14304d;color:white;position:sticky;top:0}tr:nth-child(even){background:#f4f7fb}input{padding:12px;width:90%;max-width:600px;font:inherit;border:1px solid #a9bbcf;border-radius:6px}.note{padding:14px;background:#e5effa;border-left:4px solid #2e73b4}code{background:#e9edf3;padding:2px 5px}details{margin:18px 0}summary{cursor:pointer;font-weight:bold}</style><main>
    <h1>InterMine314 performance review</h1><p class="note">Baseline 6a85ed7 and implemented uplift · Python 3.14.5 / CI current 3.14 · Intel Core Ultra 7 265K · 18 CPUs exposed to WSL · 20-core host recommendations</p>'''
    for section in sections[:4]:
        doc+=f'<h2>{html.escape(section["title"])}</h2><p>{html.escape(section["body"])}</p>'
    doc+='<h2>Final fixture worker choices</h2><p>Measured choices complement conservative stage defaults. No public mine was load tested.</p>'+table(choices,[(k,k.replace("_"," ").title()) for k in ['stage','delay_ms','workers','median_s','cv','peak_tree_rss_bytes','rule']])
    doc+='<details><summary>All 259 final trials summarized</summary>'+table(final_trials,[(k,k.replace("_"," ").title()) for k in ['stage','delay_ms','implementation','workers','repetitions','median_s','min_s','max_s','cv','cpu_cores','peak_tree_rss_bytes','stable_values']])+'</details>'
    doc+='<h2>Historical baseline measurements</h2><p>Original stage/prototype evidence, retained unchanged. These are not post-uplift speedups.</p>'
    doc+='<p><img src="worker_scaling.svg" alt="Worker throughput scaling and compaction memory" style="width:100%;max-width:1400px"></p><h2>Measured stage comparisons</h2><p>Ratios apply only to the named fixture and stage. Prototype conversion does not include all production validation; toy CPU scaling is not a package speedup.</p>'
    doc+=table(comparisons,[('label','Comparison'),('baseline_s','Baseline seconds'),('candidate_s','Candidate seconds'),('local_speedup','Local speedup')])
    doc+='<h2>Worker counts by stage</h2>'+table(WORKER_PLANS,[(k,k.replace('_',' ').title()) for k in ['stage','current','start_workers','sweep','native_threads','arrow_io_threads','inflight','rationale']])
    doc+='<h2>Current controls and starting settings</h2>'+table(settings,[(k,k.replace('_',' ').title()) for k in ['scope','setting','value','available_now','condition']])
    doc+='<h2>Completed implementation and replacement register</h2>'+table(implementation,[(k,k.replace("_"," ").title()) for k in ['opportunity_id','status','code','behavior','removed','retained_reason','tests']])
    doc+='<h2>Baseline opportunity register (original source locations)</h2><p>Likelihood is qualitative engineering judgment; priority 1 items have the strongest near-term case. Effort is an estimate in engineering days. Filter searches every visible table.</p><input id="filter" aria-label="Filter tables" placeholder="Filter: Arrow, compaction, memory, P01…">'
    overview=[dict(row,section=f'{row["file"]}:{row["line"]}') for row in enriched]
    doc+=table(overview,[(k,k.replace('_',' ').title()) for k in ['id','priority','section','proposal','gain_likelihood','worker_guidance','potentially_redundant']])
    for row in enriched:
        doc+=f'<details><summary>{row["id"]}: {html.escape(row["stage"])} — evidence, memory and implementation risks</summary>'
        for key in ['measured_comparison','measured_stage_speedup','measurement_limit','memory_effect','must_retain','compatibility_risk','effort_days_estimate','evidence_ids','decision_metrics']:
            if row.get(key) is not None:
                doc+=f'<p><strong>{html.escape(key.replace("_"," ").title())}:</strong> {html.escape(str(row[key]))}</p>'
        doc+='</details>'
    for section in sections[4:]:
        doc+=f'<h2>{html.escape(section["title"])}</h2><p>{html.escape(section["body"])}</p>'
    doc+='<h2>Implementation status and handoff</h2><p class="note">Git metadata is writable. Task and phase status below comes from the implementation ledger. Version remains 0.1.8.</p>'
    doc+=table(phase_status, [(k,k.replace('_',' ').title()) for k in ['phase','status','push_status','ci_status','head_commit','ci_url']])
    doc+=table(phase_tasks, [(k,k.replace('_',' ').title()) for k in ['task_id','opportunities','behavior','acceptance','status','commit_status']])
    doc+='<h2>Implemented advisory resource profiles</h2><p>These allocations are approved starting policies and are implemented. Existing external pools may exceed the budget and are reported. Stage pools are not allocated simultaneously.</p>'
    doc+=table(planned_profiles, [(k,k.replace('_',' ').title()) for k in ['stage','profile','cpu_budget','workers','advisory','implemented']])
    doc+='<h2>Clean distribution matrix</h2>'+table(distributions,[(k,k.replace("_"," ").title()) for k in ['lane','mode','python','version','dependencies','verified_bindings','status','source_revision','source_dirty','ci_url']])
    doc+='<h2>Workflow technologies</h2>'+table(LANGUAGES,[(k,k.replace('_',' ').title()) for k in ['technology','priority','recommendation','likely_gain','boundary','cost','evidence']])
    doc+='<h2>Runtime coverage</h2>'+table(runtime_coverage,[(k,k.title()) for k in ['stage','modules','lines','functions']])
    doc+='<details><summary>All measured configurations, spread and memory</summary><p>Parent RSS only for process pools. Native pools, setup, warmups and allocator retention affect RSS. Min/median/max and CV are descriptive statistics, not confidence intervals.</p>'
    doc+=table(metrics,[(k,k.replace('_',' ').title()) for k in metrics[0]])+'</details>'
    doc+='<h2>Verified primary sources</h2><ol>'
    for i,title,revision,url,lines,finding,method in SOURCES:
        doc+=f'<li id="{i}"><a href="{html.escape(url)}">{i}: {html.escape(title)}</a> — {html.escape(revision)}; lines {html.escape(lines)}. {html.escape(finding)} ({html.escape(method)})</li>'
    doc+='</ol><h2>Files</h2><ul>'
    for path in sorted(HERE.iterdir()):
        if path.suffix in {'.parquet','.duckdb','.py'}:
            doc+=f'<li><a href="{path.name}">{path.name}</a></li>'
    doc+='</ul></main><script>document.querySelector("#filter").addEventListener("input",e=>{let q=e.target.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>'
    (HERE/'report.html').write_text(doc)
    print(json.dumps(dict(selected_trials=n,raw_trials=all_n,configurations=len(metrics),runtime_files=metadata['runtime_python_files'],tracked_files=len(files),opportunities=len(opportunities()),report=str(HERE/'report.html'))))


if __name__ == '__main__':
    main()
