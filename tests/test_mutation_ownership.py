import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import polars as pl
import pytest

from intermine314.export import parquet
from intermine314.util.ownership import publication_owner
from tests.support.parquet import query_with_rows
from tests.test_lists_crud import client


def test_multiple_managers_allocate_unique_names_and_keep_separate_cleanup():
    service, _ = client()
    managers = [service.list_manager() for _ in range(4)]
    with ThreadPoolExecutor(8) as pool:
        names = list(pool.map(lambda i: managers[i % 4].get_unused_list_name(), range(40)))
    assert len(set(names)) == 40
    assert all(len(manager._temp_lists) == 10 for manager in managers)
    assert len({id(manager._mutation_state) for manager in managers}) == 1
    managers[0].delete_temporary_lists()
    assert not managers[0]._temp_lists
    assert all(len(manager._temp_lists) == 10 for manager in managers[1:])


def test_lazy_manager_initialization_has_one_owner():
    service, _ = client()
    with ThreadPoolExecutor(8) as pool:
        managers = list(pool.map(lambda _: service._get_list_manager(), range(40)))
    assert all(manager is managers[0] for manager in managers)


def test_target_aliases_share_lock_and_distinct_targets_do_not(tmp_path):
    entered = threading.Event()
    target = tmp_path / 'output'

    def acquire_alias():
        with publication_owner(tmp_path / 'unused' / '..' / 'output'):
            entered.set()

    with ThreadPoolExecutor(1) as pool:
        with publication_owner(target):
            future = pool.submit(acquire_alias)
            assert not entered.wait(0.05)
            with publication_owner(tmp_path / 'other'):
                pass
        future.result(5)
    assert entered.is_set()


@pytest.mark.parametrize('single_file', [True, False])
def test_concurrent_publication_is_complete_and_preserves_unrelated_entries(tmp_path, monkeypatch, single_file):
    target = tmp_path / 'result'
    if not single_file:
        target.mkdir()
        (target / 'notes.txt').write_text('keep')
    gate = threading.Barrier(2)

    @contextmanager
    def simultaneous_publication(path):
        gate.wait(timeout=5)
        with publication_owner(path):
            yield

    monkeypatch.setattr(parquet, 'publication_owner', simultaneous_publication)

    def write(value):
        query_with_rows([{'Employee.id': value}] * 6, ['Employee.id']).to_parquet(
            target, batch_size=2, single_file=single_file,
        )

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(write, value) for value in [1, 2]]
        for future in futures:
            future.result(10)
    paths = [target] if single_file else sorted(target.glob('part-*.parquet'))
    result = pl.concat([pl.read_parquet(path) for path in paths])
    assert result.height == 6 and result['Employee.id'].n_unique() == 1
    if not single_file:
        assert (target / 'notes.txt').read_text() == 'keep'
    assert list(tmp_path.iterdir()) == [target]
