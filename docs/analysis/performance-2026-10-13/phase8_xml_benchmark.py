"""Seven paired model parsing/ancestry trials against the pre-uplift parser."""
from __future__ import annotations

import gc
import hashlib
import json
import random
import subprocess
import time
import tracemalloc
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from intermine314.model import Model

HERE = Path(__file__).resolve().parent
BASELINE = '6dccbeb'


def fingerprint(model):
    rows = [(name, [p.name for p in cl.parent_classes],
             [(f.name, f.type_name, f.fieldtype, f.declared_in.name) for f in cl.fields])
            for name, cl in model.classes.items()]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def main():
    module = types.ModuleType('intermine314_baseline_model')
    source = subprocess.check_output(['git', 'show', BASELINE + ':src/intermine314/model.py'])
    exec(compile(source, '<baseline_model>', 'exec'), module.__dict__)
    classes = []
    for i in range(1000):
        parent = f' extends="C{i-i%10}"' if i % 10 else ''
        attributes = ''.join(f'<attribute name="a{j}" type="java.lang.String"/>' for j in range(8))
        classes.append(f'<class name="C{i}"{parent}>{attributes}</class>')
    payload = ('<model name="fixture" package="fixture">' + ''.join(classes) + '</model>').encode()
    expected = fingerprint(module.Model(payload))
    assert fingerprint(Model(payload)) == expected
    rows = []
    randomizer = random.Random(314)
    for repetition in range(7):
        cases = [('baseline', module.Model), ('streaming', Model)]
        randomizer.shuffle(cases)
        for case, constructor in cases:
            gc.collect()
            started = time.perf_counter()
            model = constructor(payload)
            elapsed = time.perf_counter() - started
            assert fingerprint(model) == expected
            del model
            gc.collect()
            tracemalloc.start()
            model = constructor(payload)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert fingerprint(model) == expected
            del model
            rows.append(dict(case=case, repetition=repetition, wall_seconds=elapsed,
                             python_allocation_peak_bytes=peak, classes=1000, fields_per_class=9,
                             input_bytes=len(payload), input_sha256=hashlib.sha256(payload).hexdigest(),
                             result_sha256=expected, baseline_commit=BASELINE,
                             memory_scope='Separate tracemalloc run; Python allocations, not RSS/native allocator'))
    pq.write_table(pa.Table.from_pylist(rows), HERE / 'phase8_xml_trials.parquet', compression='zstd')
    import statistics
    print({case: {'median_seconds': statistics.median(r['wall_seconds'] for r in rows if r['case'] == case),
                  'median_python_peak_bytes': statistics.median(r['python_allocation_peak_bytes'] for r in rows if r['case'] == case)}
           for case in ('baseline', 'streaming')})


if __name__ == '__main__':
    main()
