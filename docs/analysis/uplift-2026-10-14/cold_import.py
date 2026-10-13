"""Paired cold import only; no warmup, fixture or memory-tracing run."""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODE = '''
import json,sys,time
sys.path.insert(0, sys.argv[1] + '/src')
start=time.perf_counter_ns()
from intermine314.webservice import Service
elapsed=time.perf_counter_ns()-start
assert not {'polars','duckdb','pyarrow','numpy','matplotlib'}.intersection(sys.modules)
print(json.dumps({'ns':elapsed,'python':sys.version}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', default='.')
    args = parser.parse_args()
    rows, rng = [], random.Random(314)
    for repetition in range(21):
        variants = [('baseline', args.baseline), ('candidate', args.candidate)]
        rng.shuffle(variants)
        for variant, root in variants:
            root = str(Path(root).resolve())
            output = subprocess.check_output([sys.executable, '-I', '-c', CODE, root], text=True,
                                             env=dict(os.environ, PYTHON_JIT='0'))
            rows.append(dict(repetition=repetition, variant=variant, root=root, **json.loads(output)))
    pq.write_table(pa.Table.from_pylist(rows), HERE / 'cold_import.parquet', compression='zstd')
    print({v: statistics.median(r['ns'] for r in rows if r['variant'] == v) / 1e6 for v in ('baseline', 'candidate')})


if __name__ == '__main__':
    main()
