"""Build/query the relocatable DuckDB catalog; canonical evidence is Parquet.

python catalog.py --build
python catalog.py 'SELECT * FROM comparison ORDER BY case_name'
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb

DIRECTORY = Path(__file__).resolve().parent


def open_catalog(*, read_only=True):
    connection = duckdb.connect(str(DIRECTORY / "analysis.duckdb"), read_only=read_only,
                                config={"threads": 4, "memory_limit": "512MB"})
    connection.execute("SET file_search_path = ?", [str(DIRECTORY)])
    return connection


def build():
    with open_catalog(read_only=False) as connection:
        tables = {
            "trials": "trials-*.parquet", "environment": "environment-*.parquet",
            "memory": "memory-*.parquet", "requests": "requests-*.parquet",
            **{name: f"{name}.parquet" for name in (
                "inventory", "duplicate_bodies", "declarations", "research", "tests", "structural", "verification")},
        }
        for name, path in tables.items():
            connection.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet(\'{path}\')')
        connection.execute('''CREATE OR REPLACE VIEW medians AS
            SELECT "case" AS case_name, variant, jit, units, count(*) trials,
                median(seconds) AS seconds, min(seconds) AS minimum, max(seconds) AS maximum
            FROM trials GROUP BY ALL''')
        connection.execute('''CREATE OR REPLACE VIEW comparison AS
            SELECT a.case_name, a.jit, a.units, a.trials,
                a.seconds current_seconds, b.seconds prototype_seconds,
                a.seconds / b.seconds operation_speedup,
                100 * (1 - b.seconds/a.seconds) time_reduction_percent
            FROM medians a JOIN medians b USING(case_name, jit, units)
            WHERE a.variant='current' AND b.variant='prototype' ''')
        rows = connection.execute("SELECT * FROM comparison WHERE jit='jit-off' ORDER BY case_name").fetchall()
        lines = [
            "# Measured operations", "",
            "Seven randomized warm repetitions per variant on CPython 3.14.5; JIT disabled. "
            "Numbers describe each isolated operation, not end-to-end mine jobs. "
            "Prototypes are in `probe.py`; they have not been installed into the package.", "",
            "| Operation | Units | Current median, ms | Prototype median, ms | Operation ratio | Time reduction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, _, units, _, current, prototype, ratio, reduction in rows:
            lines.append(f"| {name} | {units:,} | {current * 1000:.3f} | {prototype * 1000:.3f} | {ratio:.2f}× | {reduction:.1f}% |")
        lines += ["", "`stage_lookup` compares 1,000 live CPU/affinity/quota resolutions with a precomputed "
                  "integer. Its large ratio is bookkeeping only; it is not a pipeline forecast.", "",
                  "`single_part_file` is a 10,000-row, two-column String/Int64 fixture; both variants "
                  "stage and publish through the actual writer. The prototype replaces only the final "
                  "DuckDB rewrite with a move/copy. Equality, order, one row group, and SQL readback "
                  "were checked. General eligibility and fault-injection coverage remain to be implemented.", "",
                  "`row_key_map` and `object_binding` process 20,000 synthetic rows/objects. "
                  "`json_bytes` decodes 20,000 UTF-8 arrays with the optional orjson backend. "
                  "`arrow_reservation` converts 1,000 rows × 64 columns of inferred integers. "
                  "`result_length_hint` consumes 10,000 rows from in-memory response streams; no HTTP latency is included.", "",
                  "## Retained Python allocations", "",
                  "The 10,000-row, 20-column map experiment shares its input data/views and measures "
                  "new row/map allocations with tracemalloc. It excludes native memory and input payloads. "
                  "One prebuilt shared map is outside tracing; its small one-time cost is excluded.", "",
                  "| Variant | Retained bytes | Peak bytes |", "|---|---:|---:|"]
        for variant, retained, peak in connection.execute("SELECT variant,retained_bytes,peak_bytes FROM memory WHERE jit='jit-off' ORDER BY variant").fetchall():
            lines.append(f"| {variant} | {retained:,} | {peak:,} |")
        lines += ["", "## JIT screening", "",
                  "Separate process runs, seven repetitions each; process order was not counterbalanced. "
                  "This is a screening experiment, not proof of a JIT effect. Ratios above 1 favor JIT.", "",
                  "| Current operation | JIT off, ms | JIT on, ms | Off/on ratio |", "|---|---:|---:|---:|"]
        for name, off, on in connection.execute('''SELECT case_name,
                max(seconds) FILTER(WHERE jit='jit-off'), max(seconds) FILTER(WHERE jit='jit-on')
                FROM medians WHERE variant='current' GROUP BY case_name ORDER BY case_name''').fetchall():
            lines.append(f"| {name} | {off * 1000:.3f} | {on * 1000:.3f} | {off/on:.2f}× |")
        (DIRECTORY / "measurements.md").write_text("\n".join(lines) + "\n")


def validate():
    with open_catalog() as connection:
        root = DIRECTORY.parents[2]
        for path, digest in connection.execute("SELECT path,sha256 FROM inventory").fetchall():
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
        assert connection.execute("SELECT count(*) FROM trials").fetchone()[0] == 224
        assert connection.execute("SELECT count(*) FROM medians WHERE trials != 7").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM tests WHERE passed").fetchone()[0] == 2769
        assert connection.execute("SELECT distinct requests FROM requests WHERE variant='facade'").fetchall() == [(2,)]
        assert connection.execute("SELECT distinct requests FROM requests WHERE variant='concrete'").fetchall() == [(1,)]
        assert connection.execute("SELECT count(*) FROM duckdb_tables() WHERE NOT internal").fetchone()[0] == 0
    print("Recorded source/test hashes, trials, request counts, and view-only catalog verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sql", nargs="?")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.build:
        build()
    if args.validate:
        validate()
    if args.sql:
        with open_catalog() as connection:
            result = connection.execute(args.sql)
            fields = [column[0] for column in result.description]
            print(json.dumps([dict(zip(fields, row)) for row in result.fetchall()], default=str))


if __name__ == "__main__":
    main()
