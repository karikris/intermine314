Benchmark: ThaleMine (live)
Python: 3.14
Rows requested: 50,000
Runs per mode: 3

======================================================================
FETCH PERFORMANCE
======================================================================

Mode                         Mean Time (s)   Rows/s     Speedup
----------------------------------------------------------------------
intermine (sequential)        45.36–48.01     ~1,050     1.00x (baseline)
intermine314 (4 workers)       6.97           ~7,176     6.51x
intermine314 (16 workers)      2.84           ~17,578    16.88x

Relative improvement:
- intermine314 (4 workers):   ~84.6% faster than intermine
- intermine314 (16 workers):  ~94.1% faster than intermine
- Scaling from 4 → 16 workers: ~2.45x additional speedup

All modes returned exactly 50,000 rows (min/max = 50,000 / 50,000).

======================================================================
STORAGE COMPARISON (same 50k result set)
======================================================================

Format / Tool              Size (bytes)    Size (MiB)
----------------------------------------------------------------------
CSV (intermine)             17,511,102      ~16.70
Parquet (intermine314)         414,593       ~0.40

Storage savings:
- Absolute saved:            17,096,509 bytes
- Reduction:                 97.63%
- CSV is:                    42.24× larger than Parquet

Artifacts:
- CSV (workers=4 run):      /tmp/thalemine_intermine_50000.csv
- Parquet (workers=4 run):  /tmp/thalemine_intermine314_50000.parquet
- CSV (workers=16 run):     /tmp/thalemine_intermine_50000_w16.csv
- Parquet (workers=16 run): /tmp/thalemine_intermine314_50000_w16.parquet

======================================================================
NOTES
======================================================================

- intermine used batched sequential fetch (batch_size=5000).
- intermine314 used parallel paging with:
  - workers=4 run: page_size=5000, max_workers=4, prefetch=4
  - workers=16 run: page_size=5000, max_workers=16, prefetch=16
- Outer joins were enabled for transcript/CDS and protein domain paths.
- Legacy intermine required Python 3.14 compatibility shims.
