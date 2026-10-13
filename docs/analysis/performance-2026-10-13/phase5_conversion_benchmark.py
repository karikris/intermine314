"""Seven paired conversion trials against the pre-Arrow implementation in Git.

Original baseline artifacts are immutable. This benchmark uses the same
dependency environment and compares complete frames after each timed action.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

from run_benchmarks import COLUMNS, measure, rows_fixture

from intermine314.export.model_schema import ModelSchema, model_batch_frame
from intermine314.export.parquet import _batch_frame
from intermine314.export.runtime import get_runtime

HERE = Path(__file__).resolve().parent
BASELINE = 'fbe84a604486e395f4109d6a3d40f9d6bdfed6c7'


def historical_module(path):
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], text=True)
    namespace = {'__name__': 'conversion_baseline'}
    exec(compile(source, f'{BASELINE}:{path}', 'exec'), namespace)
    return namespace


def main():
    runtime = get_runtime('cpu16')
    pl = runtime.engine('polars')
    pa = runtime.engine('pyarrow')
    import pyarrow.parquet as pq
    from polars.testing import assert_frame_equal

    old_infer = historical_module('src/intermine314/export/parquet.py')['_batch_frame']
    old_model = historical_module('src/intermine314/export/model_schema.py')['model_batch_frame']
    rows = rows_fixture()
    schema = ModelSchema(dict(zip(COLUMNS, [pl.Int64, pl.String, pl.Int32, pl.Float64,
                                           pl.String, pl.Int32, pl.String, pl.String])), ())
    actions = {
        ('inferred', 'baseline'): lambda: old_infer(pl, rows, COLUMNS),
        ('inferred', 'typed_arrow'): lambda: _batch_frame(pl, rows, COLUMNS),
        ('model', 'baseline'): lambda: old_model(pl, rows, COLUMNS, schema),
        ('model', 'typed_arrow'): lambda: model_batch_frame(pl, rows, COLUMNS, schema),
    }
    references = {stage: actions[stage, 'baseline']() for stage in ('inferred', 'model')}
    for (stage, _), action in actions.items():
        assert_frame_equal(action(), references[stage])
    randomizer = random.Random(314)
    results = []
    for repeat in range(7):
        cases = list(actions)
        randomizer.shuffle(cases)
        for stage, implementation in cases:
            def check(value, stage=stage):
                assert_frame_equal(value, references[stage])
                return value['Record.id'].sum(), value.estimated_size()

            results.append(measure(actions[stage, implementation], check,
                                   spec=dict(stage=stage, implementation=implementation,
                                             baseline_ref=BASELINE, fixture='flat-50000x8',
                                             cpu_budget=runtime.cpu_budget), repeat=repeat, rows=len(rows)))
    pq.write_table(pa.Table.from_pylist(results), HERE / 'phase5_conversion_trials.parquet', compression='zstd')


if __name__ == '__main__':
    main()
