# LegumeMine Dev Benchmark (Workers 2,3,4,6)

- Run timestamp (UTC): `2026-03-05T12:25:11.325530+00:00`
- Target endpoint: `https://mines.dev.lis.ncgr.org/legumemine/service`
- Matrix rows: `5000,10000,25000,50000,100000`
- Repetitions: `3`
- Transport modes: `direct`
- Storage compare: `disabled`
- Worker set: `2,3,4,6`

## Command

```bash
.venv314/bin/python -m benchmarks.benchmarks \
  --mine-url https://mines.dev.lis.ncgr.org/legumemine/service \
  --workers 2,3,4,6 \
  --transport-modes direct \
  --no-randomize-mode-order \
  --no-storage-compare \
  --json-out benchmark-results/legumemine_dev_full_w2346_direct.json
```

## Artifact References

- Full run JSON (local, gitignored): `benchmark-results/legumemine_dev_full_w2346_direct.json`
  - SHA256: `a109a08d38c7dc616e9f9d36a1df53645e2c07a6927b2915fd60050f29e578a3`
- Smoke JSON (local, gitignored): `benchmark-results/legumemine_dev_smoke_w2346.json`
  - SHA256: `cffb68a405973c2e4a81c8ee16886980bba2a9ef4b4e852af7cd6562e43765e9`

## Speedup vs Legacy (mean seconds)

- `rows=5000`: `w2=2.39x`, `w3=3.57x`, `w4=4.30x`, `w6=6.93x`
- `rows=10000`: `w2=3.06x`, `w3=3.84x`, `w4=5.10x`, `w6=7.66x`
- `rows=25000`: `w2=2.92x`, `w3=4.21x`, `w4=5.38x`, `w6=7.06x`
- `rows=50000`: `w2=3.01x`, `w3=3.93x`, `w4=5.69x`, `w6=8.04x`
- `rows=100000`: `w2=2.97x`, `w3=4.10x`, `w4=5.63x`, `w6=8.02x`

## Production LegumeMine Availability Note

On `2026-03-05` UTC, production endpoint probes against
`https://mines.legumeinfo.org/legumemine/service/version/ws`
returned repeated `503 Service Unavailable` responses (~60.6s each),
so the comparison run was executed against the reachable LegumeMine dev endpoint.
