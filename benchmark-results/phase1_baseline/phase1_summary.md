# Phase 1 Baseline Summary

## Lean tests (py314)
- passed: 21
- elapsed_s: 0.47
- max_rss_kb: 45744

## Import-time hotspots (mean cumulative ms)
- intermine314.query.builder: 54.544 ms
- intermine314.service.service: 42.384 ms
- intermine314.registry.mines: 35.012 ms
- intermine314.service.iterators: 33.507 ms

## Representative query memory baseline (synthetic parallel)
- status: ok
- rows_target: 50000
- rows_exported: 50000
- rows_per_s: 3742603.7774200207
- peak_rss_bytes: 37507072
- elapsed_s: 0.013359682983718812
- workers: 2
- max_inflight_bytes_estimate: None

## Representative query memory baseline (network-inclusive fixed fetch)
- maizemine status: ok
- maizemine rows_per_s: 1444.2265201207053
- maizemine peak_rss_bytes: 41213952
- maizemine elapsed_s: 1.3848243140091654
- legumemine status: ok
- thalemine status: skipped
- thalemine reason: connect_failed
