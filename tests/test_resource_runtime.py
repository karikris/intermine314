import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from intermine314.export.runtime import ResourceRuntime, get_runtime


def test_operation_worker_plan_reuses_quota_and_refreshes_next_operation(monkeypatch):
    from intermine314.export import resource_profile

    cpus, calls = [16], []

    def available():
        calls.append(cpus[0])
        return cpus[0]

    monkeypatch.setattr(resource_profile, 'available_cpus', available)
    with ResourceRuntime('cpu16') as runtime:
        with runtime.activate():
            assert runtime._workers_for('duckdb_sql') == 16
            cpus[0] = 2
            with runtime.activate(), runtime.native('ordered_copy'):
                assert runtime._workers_for('ordered_copy') == 4
                assert runtime.cpu_budget == 16
            assert len(calls) == 1
        with runtime.activate():
            assert runtime._workers_for('duckdb_sql') == 2
            assert runtime._workers_for('ordered_copy') == 1
            assert runtime.cpu_budget == 2
            with pytest.raises(ValueError, match='Unknown'):
                runtime._workers_for('missing')
        assert calls == [16, 2]


def test_native_admission_is_shared_across_runtimes_and_releases_on_error():
    first = ResourceRuntime('cpu4')
    second = ResourceRuntime('cpu4')
    started, release, entered = threading.Event(), threading.Event(), threading.Event()

    def holding():
        with first.native('ordered_copy', threads=3):
            started.set()
            assert release.wait(5)

    def waiting():
        with second.native('ordered_copy', threads=3):
            entered.set()
            raise ValueError('kernel failed')

    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(holding)
        try:
            assert started.wait(5)
            b = pool.submit(waiting)
            assert not entered.wait(0.05)
        finally:
            release.set()
        a.result(5)
        with pytest.raises(ValueError, match='kernel failed'):
            b.result(5)
    assert entered.is_set()
    assert first.diagnostics()['active_native_reservations'] == 0
    first.close()
    second.close()


def test_closed_runtime_and_conflicting_profiles():
    with ResourceRuntime('cpu4') as runtime:
        assert get_runtime('cpu4', runtime) is runtime
        with pytest.raises(ValueError, match='conflicts'):
            get_runtime('cpu8', runtime)
        with runtime.activate():
            assert get_runtime() is runtime
            with pytest.raises(RuntimeError, match='active operations'):
                runtime.close()
    with pytest.raises(RuntimeError, match='closed'):
        with runtime.native('duckdb_sql'):
            pass
    runtime.close()


@pytest.mark.parametrize('profile,budget,polars,arrow', [
    ('cpu16', 16, 12, 4), ('cpu8', 8, 8, 4), ('cpu6', 6, 6, 3),
    ('cpu4', 4, 4, 2), ('cpu2', 2, 2, 1),
])
def test_fresh_process_native_pools(profile, budget, polars, arrow):
    code = '''
import json, os
os.environ.pop('POLARS_MAX_THREADS', None)
from intermine314.export import resource_profile
resource_profile.available_cpus = lambda: 20
from intermine314.export.runtime import ResourceRuntime
with ResourceRuntime(PROFILE) as runtime:
    runtime.engine('polars')
    runtime.engine('pyarrow')
    print(json.dumps(runtime.diagnostics()))
'''.replace('PROFILE', repr(profile))
    data = json.loads(subprocess.check_output([sys.executable, '-c', code], text=True))
    assert data['effective_cpu_budget'] == budget
    assert data['engines']['polars']['effective'] == polars
    assert data['engines']['pyarrow']['effective'] == arrow


def test_preinitialized_pool_is_advisory_and_not_resized():
    code = '''
import os
os.environ['POLARS_MAX_THREADS'] = '8'
import polars
assert polars.thread_pool_size() == 8
from intermine314.export.runtime import ResourceRuntime
with ResourceRuntime('cpu2') as runtime:
    with runtime.native('polars'):
        assert runtime.diagnostics()['engines']['polars']['effective'] == 8
    assert polars.thread_pool_size() == 8
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'advisory CPU budget=2' in result.stderr


def test_close_releases_all_owned_resources_even_after_error():
    closed = []

    class Owned:
        def __init__(self, value):
            self.value = value

        def close(self):
            closed.append(self.value)
            if self.value == 'bad':
                raise ValueError('close failed')

    runtime = ResourceRuntime('cpu2')
    runtime.own(Owned('first'))
    runtime.own(Owned('bad'))
    with pytest.raises(ValueError, match='close failed'):
        runtime.close()
    assert closed == ['bad', 'first']
    runtime.close()
