import multiprocessing
import os
import threading
from concurrent.futures import CancelledError

import polars as pl
import pytest

from intermine314.export import (
    ResourceProfile,
    ResourceRuntime,
    query_parquet,
    transform_parquet,
)

CALLBACKS = 'tests.fixtures.parquet_transforms:'


def inputs(tmp_path, count=4):
    paths = []
    for index in range(count):
        path = tmp_path / f'input-{index}.parquet'
        pl.DataFrame({'value': [index * 2, index * 2 + 1]}).write_parquet(path)
        paths.append(path)
    return paths


def test_spawn_transforms_native_limits_order_manifest_and_environment(tmp_path, monkeypatch):
    paths = inputs(tmp_path)
    target = tmp_path / 'output'
    target.mkdir()
    (target / 'notes.txt').write_text('keep me')
    monkeypatch.setenv('POLARS_MAX_THREADS', '7')
    children = {child.pid for child in multiprocessing.active_children()}
    with ResourceRuntime('cpu2') as runtime:
        result = transform_parquet(paths, target, CALLBACKS + 'add_value', max_workers=8,
                                   inflight_limit=4, runtime=runtime, timeout=20)
        assert runtime.diagnostics()['active_native_reservations'] == 0
    assert result == str(target)
    assert query_parquet(target)['value'].to_list() == list(range(1, 9))
    assert (target / 'notes.txt').read_text() == 'keep me'
    assert (target / '_intermine314/manifest.parquet').exists()
    assert len(list(target.glob('part-*.parquet'))) == 4
    assert os.environ['POLARS_MAX_THREADS'] == '7'
    assert {child.pid for child in multiprocessing.active_children()} == children
    assert sorted(path.name for path in tmp_path.iterdir()) == [*[path.name for path in paths], 'output']


@pytest.mark.parametrize('callback,match,timeout', [('fail', 'injected transform failure', 20),
                                                    ('crash', 'exited unexpectedly', 20),
                                                    ('stall', 'timed out', .5),
                                                    ('inconsistent', 'schemas must match', 20)])
def test_worker_failures_preserve_existing_output_and_reap_children(tmp_path, callback, match, timeout):
    paths = inputs(tmp_path, 2)
    target = tmp_path / 'output'
    target.mkdir()
    (target / 'part-00000.parquet').write_bytes(b'old output')
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises((RuntimeError, TimeoutError, ValueError), match=match):
        transform_parquet(paths, target, CALLBACKS + callback, max_workers=2, timeout=timeout)
    assert (target / 'part-00000.parquet').read_bytes() == b'old output'
    assert {child.pid for child in multiprocessing.active_children()} == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [*[path.name for path in paths], 'output']


def test_transform_cancellation_and_late_invalid_input_preserve_target(tmp_path):
    paths = inputs(tmp_path, 1)
    target = tmp_path / 'output'
    event = threading.Event()
    event.set()
    with pytest.raises(CancelledError):
        transform_parquet(paths, target, CALLBACKS + 'add_value', cancel_event=event)
    assert not target.exists()
    with pytest.raises(TypeError, match='Transform input'):
        transform_parquet([*paths, None], target, CALLBACKS + 'add_value', max_workers=1, timeout=20)
    assert not target.exists()
    assert sorted(tmp_path.iterdir()) == paths


@pytest.mark.parametrize('options', [{'max_workers': 0}, {'inflight_limit': 9}, {'timeout': -1},
                                    {'timeout': True}, {'cancel_event': object()}])
def test_invalid_transform_options_precede_process_creation(tmp_path, options):
    with pytest.raises((TypeError, ValueError)):
        transform_parquet([], tmp_path / 'output', CALLBACKS + 'add_value', **options)
    assert list(tmp_path.iterdir()) == []


def test_transform_requires_importable_callable_name_and_preserves_inputs(tmp_path):
    with pytest.raises(ValueError, match='module:function'):
        transform_parquet([], tmp_path / 'output', lambda a, b: None)
    source = inputs(tmp_path, 1)[0]
    with pytest.raises(ValueError, match='replace an input'):
        transform_parquet([source], tmp_path, CALLBACKS + 'add_value', max_workers=1)
    assert pl.read_parquet(source)['value'].to_list() == [0, 1]


def test_input_submission_is_bounded_even_when_transform_is_slow(tmp_path):
    paths = inputs(tmp_path, 1)
    pulled = []
    def source():
        for index in range(100):
            pulled.append(index)
            yield paths[0]
    with pytest.raises(TimeoutError):
        transform_parquet(source(), tmp_path / 'output', CALLBACKS + 'stall',
                          max_workers=2, inflight_limit=2, timeout=.5,
                          resource_profile=ResourceProfile(cpu_budget=2))
    assert pulled == [0, 1]
