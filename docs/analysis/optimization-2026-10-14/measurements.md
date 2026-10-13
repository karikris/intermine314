# Measured operations

Seven randomized warm repetitions per variant on CPython 3.14.5; JIT disabled. Numbers describe each isolated operation, not end-to-end mine jobs. Prototypes are in `probe.py`; they have not been installed into the package.

| Operation | Units | Current median, ms | Prototype median, ms | Operation ratio | Time reduction |
|---|---:|---:|---:|---:|---:|
| arrow_reservation | 64,000 | 6.664 | 5.209 | 1.28× | 21.8% |
| json_bytes | 20,000 | 10.067 | 8.854 | 1.14× | 12.1% |
| list_parser | 20,000 | 1.844 | 0.919 | 2.01× | 50.2% |
| object_binding | 20,000 | 18.095 | 9.128 | 1.98× | 49.6% |
| result_length_hint | 10,000 | 12.628 | 6.415 | 1.97× | 49.2% |
| row_key_map | 20,000 | 33.213 | 2.279 | 14.57× | 93.1% |
| single_part_file | 10,000 | 14.895 | 1.935 | 7.70× | 87.0% |
| stage_lookup | 1,000 | 7.024 | 0.024 | 292.06× | 99.7% |

`stage_lookup` compares 1,000 live CPU/affinity/quota resolutions with a precomputed integer. Its large ratio is bookkeeping only; it is not a pipeline forecast.

`single_part_file` is a 10,000-row, two-column String/Int64 fixture; both variants stage and publish through the actual writer. The prototype replaces only the final DuckDB rewrite with a move/copy. Equality, order, one row group, and SQL readback were checked. General eligibility and fault-injection coverage remain to be implemented.

`row_key_map` and `object_binding` process 20,000 synthetic rows/objects. `json_bytes` decodes 20,000 UTF-8 arrays with the optional orjson backend. `arrow_reservation` converts 1,000 rows × 64 columns of inferred integers. `result_length_hint` consumes 10,000 rows from in-memory response streams; no HTTP latency is included.

## Retained Python allocations

The 10,000-row, 20-column map experiment shares its input data/views and measures new row/map allocations with tracemalloc. It excludes native memory and input payloads. One prebuilt shared map is outside tracing; its small one-time cost is excluded.

| Variant | Retained bytes | Peak bytes |
|---|---:|---:|
| current | 18,065,632 | 18,065,753 |
| shared | 1,045,360 | 1,045,360 |

## JIT screening

Separate process runs, seven repetitions each; process order was not counterbalanced. This is a screening experiment, not proof of a JIT effect. Ratios above 1 favor JIT.

| Current operation | JIT off, ms | JIT on, ms | Off/on ratio |
|---|---:|---:|---:|
| arrow_reservation | 6.664 | 6.888 | 0.97× |
| json_bytes | 10.067 | 9.930 | 1.01× |
| list_parser | 1.844 | 1.857 | 0.99× |
| object_binding | 18.095 | 17.718 | 1.02× |
| result_length_hint | 12.628 | 12.216 | 1.03× |
| row_key_map | 33.213 | 31.326 | 1.06× |
| single_part_file | 14.895 | 13.670 | 1.09× |
| stage_lookup | 7.024 | 7.280 | 0.96× |
