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
