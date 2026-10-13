"""Reproduce the approved task ledger; never infer completion from file presence.

Task commits are discovered from Git history. A phase is accepted only after
its pushed head has a successful GitHub CI run, recorded with --accept-phase.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent

# task, opportunities, behavior, acceptance
TASKS = [
    ('1.1', ['P26'], 'Capture immutable baseline observations and source/dependency fingerprints', 'Original selected trials=482; raw trials=689; baseline validation and fixture hashes retained'),
    ('1.2', ['P26'], 'Replace broken phase0 invocation with supported Query APIs', 'Real compressed loopback export with joins passes; late status failure preserves prior output'),
    ('1.3', ['P26', 'P09'], 'Create phase ledger and relocatable catalog; remove excluded accelerator recommendations', '26 opportunity dispositions; P09 excluded; every included task has checks; relocated views queryable'),
    ('2.1', ['P07'], 'Extend ResourceProfile with 16/8/6/4/2 budgets, memory and precedence', 'Reject invalid counts; preserve tor_low_mem and overrides; clip automatic allocations to usable CPUs'),
    ('2.2', ['P07'], 'Add shared ResourceRuntime, engine ownership, admission and diagnostics', 'Concurrent jobs share reservations; oversized preinitialized pools are advisory and reported; close is deterministic'),
    ('2.3', ['P07', 'P22'], 'Wire production APIs and remove duplicate policy/default resolution', 'Export/SQL/fetch/Template forwarding uses profiles; lazy imports preserved; public aliases still work'),
    ('3.1', ['P04'], 'Add per-origin HTTP permits and throttling recovery', 'Concurrent exports obey total host ceiling; Retry-After honored; cancellation releases permits'),
    ('3.2', ['P20'], 'Own reusable executors and immutable worker session configuration', 'Borrowed sessions remain unchanged; cookie-sensitive flows serialize; owned resources close once'),
    ('3.3', ['P25'], 'Serialize account/list mutations and publication by owner/target', 'List names remain unique; mutation retry prohibition retained; concurrent target publication is coordinated'),
    ('4.1', ['P08'], 'Freeze execution snapshot and pre-encode invariant query fields once', 'Page XML equivalent; mutation isolation; public clone behavior unchanged'),
    ('4.2', ['P05'], 'Deliver validated pages and remove analytical row reassembly', 'Status footer checked before accepting every page; public row iterator adapter remains compatible'),
    ('4.3', ['P10'], 'Bound pending/completed/emitting pages and native batches together', 'Stalled early page cannot exceed outstanding limits; cancellation/spill cleanup; calibrated object estimates'),
    ('5.1', ['P02'], 'Build typed Arrow batches with precision-preserving validation', 'Decimal, arbitrary-integer, mixed numeric, timestamp and nested-value differential tests pass'),
    ('5.2', ['P11'], 'Resolve stable schemas once and rewrite only promoted parts', 'Null-first and changing-scale batches preserve values; fixed schemas avoid rereads'),
    ('5.3', ['P02', 'P11'], 'Delete superseded supported-schema conversion loops', 'One implementation per supported case; every remaining fallback documents a distinct semantic requirement'),
    ('6.1', ['P03'], 'Separate producer batches, row groups and output files', '122880-row/256MiB starting geometry remains byte bounded; high-level partitioned output is available'),
    ('6.2', ['P01'], 'Apply profile-specific COPY threads/memory/order/spill settings', 'One COPY; requested ordering preserved; low-memory one-thread mode retained'),
    ('6.3', ['P12'], 'Move validated same-filesystem staging instead of copying', 'Cross-device fallback, rollback, unrelated entries and existing-output preservation pass'),
    ('6.4', ['P21'], 'Publish Parquet manifests and validate cache freshness', 'Relative parts/schema/order/row counts valid; stale or untrusted files force validation; metadata excluded from scans'),
    ('7.1', ['P06'], 'Expose context-owned query_parquet_reader and shared connection policy', 'Exhaustion/early close/error/cancellation close owned resources; borrowed settings remain intact'),
    ('7.2', ['P13'], 'Bound nonseekable CSV ingestion and project path scans early', 'No whole-input snapshot; explicit types; dialect, stream ownership and publication parity'),
    ('7.3', ['P14'], 'Validate identifiers to Parquet and stream a known-length upload body', 'Invalid input causes zero mutation requests; leading zeros/quoting exact; one upload attempt'),
    ('7.4', ['P19'], 'Add enrichment path/reader modes and independent read admission', 'Path mode avoids materialization; legacy DataFrame results and precision remain equivalent'),
    ('8.1', ['P15'], 'Add single-flight metadata caches and leased Registry clients', 'One cold initializer per key; failed initializer retry; no close of an in-use client'),
    ('8.2', ['P16'], 'Replace duplicate XML trees and memoize validated ancestry', 'Compressed sized reads, descriptor identity, reverse references, cycles and model errors match'),
    ('8.3', ['P17'], 'Add opt-in batched object prefetch by service/model/class/ID', 'Fewer requests; selected nulls, identity map, cardinality and legacy lazy access preserved'),
    ('9.1', ['P18'], 'Coordinate due polls without a sleeping worker per job', 'Monotonic deadlines, backoff, terminal states and cancellation pass'),
    ('9.2', ['P23'], 'Add coarse Parquet-path transforms using spawn processes', 'One native thread per child; bounded submissions; crash cleanup; no live service/connection IPC'),
    ('9.3', ['P24'], 'Preaggregate analytics before single-owner plotting', 'Bounded upstream reads; expected plots and historical helper output contracts preserved'),
    ('10.1', [], 'Sweep stage workers and compare correctness-checked resource measurements', 'Seven measured repetitions; select smallest count within 5% of best median subject to memory; report uncertainty'),
    ('10.2', [], 'Run full compatibility, packaging and dependency matrix', 'Python 3.14 minimum/current dependency wheels; optional extras; clean installs; all required tests pass'),
    ('10.3', [], 'Regenerate final evidence and close every included opportunity', 'All 25 included opportunities accepted; P09 excluded; no retirement backlog or unexplained regressions; pushed CI verified'),
]

HANDSHAKE = [
    ('identity', 'Phase/task IDs, source tree, task commits, predecessor, dependency versions, pinned GitHits sources'),
    ('behavior', 'API/default/schema/ownership changes and completed opportunity IDs'),
    ('removals', 'Deleted implementation and replacement; semantic reasons for retained fallbacks'),
    ('verification', 'Tests, failure injection, benchmark comparisons, resource peaks and artifact hashes'),
    ('integration', 'Inputs for successor, resource/lifecycle obligations and migrations'),
    ('disposition', 'Accepted checks, unresolved failures and excluded items; failed acceptance blocks successor'),
    ('vcs', 'One task per master commit; phase push; required CI green; verify predecessor before dependent work'),
]


def write(name, rows):
    pq.write_table(pa.Table.from_pylist(rows), HERE / (name + '.parquet'), compression='zstd')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-phase', type=int, choices=range(1, 11))
    parser.add_argument('--ci-run')
    args = parser.parse_args()
    root = HERE.parents[2]
    log = subprocess.check_output(['git', 'log', '--format=%H %s'], cwd=root, text=True)
    commits = {}
    for line in log.splitlines():
        match = re.search(r'\((\d+\.\d+)\)$', line)
        if match:
            commits.setdefault(match.group(1), line.split()[0])
    previous = {}
    if (HERE / 'phase_status.parquet').exists():
        previous = {row['phase']: row for row in pq.read_table(HERE / 'phase_status.parquet').to_pylist()}
    if args.accept_phase is not None:
        if not args.ci_run:
            parser.error('--accept-phase requires --ci-run')
        run = json.loads(subprocess.check_output([
            'gh', 'run', 'view', args.ci_run, '--repo', 'karikris/intermine314',
            '--json', 'headSha,conclusion,status,url,workflowName,headBranch',
        ], text=True))
        if not (run['status'] == 'completed' and run['conclusion'] == 'success'
                and run['workflowName'] == 'CI' and run['headBranch'] == 'master'):
            raise ValueError('The phase requires successful completed CI on master')
        if args.accept_phase > 1 and previous.get(args.accept_phase-1, {}).get('status') != 'accepted':
            raise ValueError('Predecessor phase is not accepted')
        for task, _, _, _ in TASKS:
            if int(task.split('.')[0]) == args.accept_phase:
                if task not in commits:
                    raise ValueError(f'Missing task commit: {task}')
                subprocess.run(['git', 'merge-base', '--is-ancestor', commits[task], run['headSha']], cwd=root, check=True)
        previous[args.accept_phase] = dict(
            phase=args.accept_phase, predecessor_phase=args.accept_phase-1, status='accepted',
            push_status='verified', ci_status='success', ci_url=run['url'],
            head_commit=run['headSha'], version='0.1.8', branch='master', blocker='',
        )
    write('implementation_tasks', [dict(
        task_id=task, phase=int(task.split('.')[0]), opportunities=ids,
        behavior=behavior, acceptance=acceptance,
        predecessor_phase=max(0, int(task.split('.')[0]) - 1),
        status='committed' if task in commits else ('implemented_uncommitted' if task.startswith('1.') else 'not_started'),
        commit_status='committed' if task in commits else 'pending',
        commit_sha=commits.get(task, ''),
        commit_subject=f'perf: {behavior} ({task})',
        retirement_policy='Delete superseded internals in this task; retain distinct public semantics',
    ) for task, ids, behavior, acceptance in TASKS])
    write('opportunity_dispositions', [dict(
        opportunity_id=f'P{i:02}',
        disposition='excluded_by_user' if i == 9 else 'included',
        tasks=[task for task, ids, _, _ in TASKS if f'P{i:02}' in ids],
        completed=i != 9 and all(previous.get(int(task.split('.')[0]), {}).get('status') == 'accepted' for task, ids, _, _ in TASKS if f'P{i:02}' in ids),
    ) for i in range(1, 27)])
    write('phase_handshake_requirements', [dict(field=key, requirement=value) for key, value in HANDSHAKE])
    write('phase_status', [previous[phase] if previous.get(phase, {}).get('status') == 'accepted' else dict(
        phase=phase, predecessor_phase=phase-1,
        status='awaiting_push_ci' if any(task.startswith(str(phase)+'.') for task in commits) else 'not_started',
        push_status='pending', ci_status='not_run', ci_url='', head_commit='',
        version='0.1.8', branch='master', blocker='',
    ) for phase in range(1, 11)])
    allocations = {
        'public_http': [4, 4, 4, 4, 2], 'low_latency_http': [2, 2, 2, 2, 2],
        'high_latency_http': [8, 8, 6, 4, 2], 'arrow_cpu': [4, 4, 3, 2, 1],
        'arrow_io': [2, 2, 2, 2, 1], 'ordered_copy': [4, 4, 3, 2, 1],
        'unordered_copy': [8, 8, 6, 4, 2], 'duckdb_sql': [16, 8, 6, 4, 2],
        'polars': [12, 8, 6, 4, 2], 'python_processes': [4, 4, 3, 2, 1],
        'metadata': [4, 4, 4, 4, 2], 'mutable_owner': [1, 1, 1, 1, 1],
    }
    write('planned_worker_profiles', [dict(
        stage=stage, profile=f'cpu{budget}', cpu_budget=budget, workers=workers,
        advisory=True, implemented=previous.get(2, {}).get('status') == 'accepted',
    ) for stage, counts in allocations.items() for budget, workers in zip([16, 8, 6, 4, 2], counts)])


if __name__ == '__main__':
    main()
