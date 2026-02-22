# Benchmarks

This directory is the modern top-level home for benchmark runners, benchmark cases,
and benchmark profile inputs.

Current implementation status:
- The main benchmark implementation lives directly in `benchmarks/`.
- `benchmarks/runners/run_live.py` is the canonical entrypoint and delegates to
  `benchmarks/benchmarks.py`.

Run:

```bash
python benchmarks/runners/run_live.py --benchmark-target legumemine --workers auto --benchmark-profile auto
```

Phase-0 baseline capture:

```bash
python benchmarks/runners/phase0_baselines.py \
  --benchmark-target maizemine \
  --mode both \
  --workflow elt \
  --rows-target 100000 \
  --json-out /tmp/intermine314_phase0.json
```

This records import latency, peak RSS, rows/sec, and intermine314 log-volume
for direct and Tor modes. If a mode is not reachable in the current
environment, it is marked as `skipped` with a reason.

Phase-0 parallel policy baseline capture:

```bash
python benchmarks/runners/phase0_parallel_baselines.py \
  --modes ordered,unordered,window \
  --rows-target 50000 \
  --page-size 1000 \
  --max-workers 4 \
  --import-repetitions 5 \
  --json-out /tmp/intermine314_phase0_parallel.json
```

This captures:
- cold import latency for `intermine314.parallel.policy` and `intermine314.query.builder`
- synthetic rows/sec and peak RSS by order mode
- structured log volume plus observability probes:
  - `parallel_export_start` and `parallel_export_done` paired once per run
  - `parallel_ordered_scheduler_stats` emitted in ordered mode and at `DEBUG` level

Phase-0 model layer baseline capture:

```bash
python benchmarks/runners/phase0_model_baselines.py \
  --kinds both \
  --object-count 50000 \
  --import-repetitions 5 \
  --json-out /tmp/intermine314_phase0_model.json
```

This captures:
- cold import latency for `intermine314.model`
- construction throughput for `Path` and `Column` objects
- tracemalloc peak bytes plus best-effort peak RSS snapshots

## Live Runner Behavior

`benchmarks/runners/run_live.py` now treats live connectivity checks as optional
and self-diagnosing:

- Exit code `0`: benchmark completed successfully.
- Exit code `2`: live preflight skipped due to environment constraints
  (for example DNS/network/proxy unavailable).
- Exit code `1`: real benchmark failure after preflight passed.

Preflight diagnostics log compact fields only:

- `mode` (`direct` / `tor`)
- `host`
- `reason` (`ok` / `dns_failed` / `connect_failed` / `proxy_failed`)
- `elapsed_s`
- `err_type`

No query payloads, list IDs, or full responses are logged.

### CI and Sandbox Defaults

- In CI, live runs are skipped by default unless `RUN_LIVE=1` is set.
- In sandboxed runners, DNS and outbound networking may be blocked by policy.
  In that case preflight exits with code `2` rather than signaling a regression.
- Tor-mode checks require a working SOCKS proxy and use `socks5h://` to keep DNS
  resolution inside Tor.

### Tor Overhead Expectations

Tor routing usually adds latency and retry variance versus direct mode. Compare
success/failure and bounded retries, not strict timing thresholds.
