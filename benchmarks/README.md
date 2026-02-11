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
