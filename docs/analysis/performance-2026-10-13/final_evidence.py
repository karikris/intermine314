"""Final implementation/retirement register and installed-distribution evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# Distinct public contracts stay supported; superseded private paths are removed.
# Test references are executable evidence, not claims about unmeasured WAN speed.
OUTCOMES = [
    ('P01', 'export/parquet.py', 'Stage-budgeted COPY; ordered/unordered controls', 'Hardcoded one-thread/128MB merge replaced', 'Ordered multi-file COPY stays single-threaded: DuckDB ordering limitation', 'test_parquet_layout.py'),
    ('P02', 'export/arrow.py', 'Typed Arrow primitive conversion with exactness guards', 'Repeated generic/model conversion loops replaced by one dispatcher', 'Mixed/nested/custom rows retain fallback; float-to-integer guards remain', 'test_typed_arrow.py'),
    ('P03', 'export/layout.py', 'Independent row-group, byte and target-file geometry', 'Producer batch size no longer dictates final row groups', 'Tiny outputs and type-sensitive parts need not be compacted', 'test_parquet_layout.py'),
    ('P04', 'service/admission.py', 'Shared per-origin/global permits through response close, throttle feedback', 'Independent uncoordinated request fanout replaced', 'Explicit overrides and borrowed/custom session serialization retain ownership', 'test_parallel_http_integration.py'),
    ('P05', 'query/parallel_offset.py', 'ValidatedPage/PageBatch transfers to analytics without reassembling rows', 'Export page-to-row-to-batch cycle removed', 'Public row iteration retains its existing format', 'test_parallel_page_validation.py'),
    ('P06', 'export/query.py', 'Managed bounded Arrow reader over DuckDB SQL', 'Reader path avoids full Arrow table and Polars materialization', 'query_parquet keeps deliberate materialized DataFrame return', 'test_query_parquet_reader.py'),
    ('P07', 'export/runtime.py', 'Shared native admission, lazy engine configuration and owned cleanup', 'Uncoordinated per-operation native defaults replaced', 'Existing oversized pools are reported, not forcibly resized', 'test_resource_runtime.py'),
    ('P08', 'query/executor.py', 'Immutable prepared execution snapshot reused by page workers', 'Repeated built-in query serialization and preparation removed', 'Custom execution hooks retain their extension contract', 'test_parallel_page_validation.py'),
    ('P10', 'query/inflight.py', 'Pending, buffered, emitting and native-buffer accounting', 'Payload-only estimates and pending-future-only bounds replaced', 'One oversized probe page can exceed the estimate; not a hard RSS cap', 'test_parallel_memory_bounds.py'),
    ('P11', 'export/parquet.py', 'Fixed model schema avoids rereads; promotions touch only affected parts', 'Unconditional casts/schema rewrites removed', 'Late promotion preserves lossless precision/type validation', 'test_schema_evolution.py'),
    ('P12', 'export/parquet.py', 'Same-filesystem moves, shared transactional dataset publication', 'Always-copy final publication and duplicate publisher removed', 'EXDEV copy fallback and rollback remain necessary', 'test_parquet_layout.py'),
    ('P13', 'export/csv.py', 'Sized borrowed-stream spool, lazy projection, Arrow-reader directory export', 'Whole-stream snapshots and duplicate DuckDB loop removed', 'CSV input/output APIs remain supported; internal durable data are Parquet', 'test_import_csv.py'),
    ('P14', 'lists/_csv.py', 'Validate identifiers to Parquet and stream a known-length upload', 'Whole identifier list and joined request string removed from CSV path', 'Legacy text/list inputs keep their wire format; writes are one attempt', 'test_identifier_stream_upload.py'),
    ('P15', 'service/metadata.py', 'Per-key metadata initialization barriers and Registry leases', 'Racy duplicate cold initialization and unprotected cache mutation replaced', 'Unleased legacy Registry lookup retains its lifetime behavior', 'test_metadata_concurrency.py'),
    ('P16', 'service/xmlstream.py', 'Sized Expat events, selected subtrees, validation-pass ancestry cache', 'Normal discovery full DOM and repeated ancestry traversal removed', 'Explicit DOM API and full model-error source diagnostics remain', 'test_service_metadata.py'),
    ('P17', '_prefetch.py', 'Bounded object prefetch grouped by service/model/class/ID', 'Opt-in path avoids lazy one-request-per-object traffic', 'Lazy access stays default; relationship fanout may exceed nominal input-row memory', 'test_object_prefetch.py'),
    ('P18', 'idresolution.py', 'Monotonic due queue and bounded status-fetch workers', 'Batch path no longer sleeps separately in every worker', 'Single-job poll remains; remote mutation is never retried by batch polling', 'test_job_polling.py'),
    ('P19', 'lists/_enrichment.py', 'Path/reader result modes validate and publish before consumption', 'Nonmaterialized modes avoid unconditional collect/reload', 'Default DataFrame return remains; caller owns returned reader', 'test_lists_enrichment.py'),
    ('P20', 'export/runtime.py', 'Runtime-owned reusable executors and private session snapshots', 'Per-call pool churn removed for reusable runtime', 'Borrowed mutable/custom sessions retain serialized use', 'test_worker_ownership.py'),
    ('P21', 'export/manifest.py', 'Relative Parquet manifest and bounded fingerprinted schema cache', 'Repeated unchanged schema decoding bypassed', 'External data still get validation; no remote snapshot/checksum provenance claim', 'test_parquet_manifests.py'),
    ('P22', 'query/parallel_runtime.py', 'Shared scheduler/policy, strict options and one lifecycle path', 'Duplicated scheduling and inconsistent policy normalization removed', 'Public compatibility adapters retain their distinct signatures', 'test_parallel_cleanup.py'),
    ('P23', 'export/transforms.py', 'Bounded spawn workers, path-only IPC, native child pools of one', 'No redundant prior CPU-transform executor existed', 'Callbacks outside assigned outputs are not transactional; startup/IPC require coarse work', 'test_parquet_transforms.py'),
    ('P24', 'bar_chart.py', 'DuckDB top-category aggregation persisted as Parquet before serial plotting', 'New analytics plots bypass full raw TSV/row processing', 'Legacy plot cardinality/return behavior remains; GUI main-thread responsibility', 'test_parquet_plots.py'),
    ('P25', 'util/ownership.py', 'Explicit owners for mutation, publication and plotting', 'Unprotected shared mutation paths replaced', 'Independent reads may overlap; separate processes need publication coordination', 'test_mutation_ownership.py'),
    ('P26', '../../benchmarks/runners/phase0_baselines.py', 'Supported Query API benchmarks, portable Parquet evidence and DuckDB catalog', 'Obsolete fetch helper call and absolute evidence-view paths removed', 'Historical observations preserved as baseline evidence', 'test_phase0_export_runner.py'),
]


def outcomes():
    dispositions = {r['opportunity_id']: r for r in pq.read_table(HERE / 'opportunity_dispositions.parquet').to_pylist()}
    rows = []
    for key, code, behavior, removed, retained, proof in OUTCOMES:
        path = (ROOT / 'src/intermine314' / code).resolve()
        assert path.is_file() and (ROOT / 'tests' / proof).is_file()
        rows.append(dict(opportunity_id=key, status='accepted' if dispositions[key]['completed'] else 'awaiting_phase_acceptance',
                         code=str(path.relative_to(ROOT)), behavior=behavior, removed=removed,
                         retained_reason=retained, retirement_backlog=False, tests='tests/' + proof,
                         tasks=dispositions[key]['tasks'], performance_scope='See measured phase tables; no unmeasured end-to-end gain claim'))
    pq.write_table(pa.Table.from_pylist(rows), HERE / 'implementation_outcomes.parquet', compression='zstd')
    return rows


def distribution(report_file, lane, ci_url):
    report = json.loads(Path(report_file).read_text())
    assert report['status'] == 'ok' and report['version'] == '0.1.8'
    path = HERE / 'distribution_matrix.parquet'
    rows = pq.read_table(path).to_pylist() if path.exists() else []
    rows = [row for row in rows if row['lane'] != lane]
    for entry in report['installations']:
        rows.append(dict(lane=lane, mode=entry['mode'], python=entry['python'].split()[0],
                         dependencies=json.dumps(entry['dependencies'], sort_keys=True),
                         optional_dependencies=json.dumps(entry['optional_dependencies'], sort_keys=True),
                         version=entry['package_version'], verified_bindings=entry['verified_bindings'],
                         source_revision=report['source_revision'], source_dirty=report['source_dirty'],
                         artifact_sha256=[item['sha256'] for item in report['artifacts']],
                         ci_url=ci_url, status='passed'))
    pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distribution-report')
    parser.add_argument('--lane', default='local-minimum')
    parser.add_argument('--ci-url', default='')
    args = parser.parse_args()
    if args.distribution_report:
        distribution(args.distribution_report, args.lane, args.ci_url)
    outcomes()
