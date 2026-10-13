from dataclasses import replace

import pytest

from intermine314.export import resource_profile as policy


@pytest.mark.parametrize('budget,polars,arrow,copy,processes', [
    (16, 12, 4, 4, 4), (8, 8, 4, 4, 4), (6, 6, 3, 3, 3),
    (4, 4, 2, 2, 2), (2, 2, 1, 1, 1),
])
def test_stage_profiles(monkeypatch, budget, polars, arrow, copy, processes):
    monkeypatch.setattr(policy, 'available_cpus', lambda: 20)
    profile = policy.resolve_resource_profile(f'cpu{budget}')
    assert profile.effective_cpu_budget == budget
    assert profile.workers_for('polars') == polars
    assert profile.workers_for('arrow_cpu') == arrow
    assert profile.workers_for('ordered_copy') == copy
    assert profile.workers_for('python_processes') == processes
    assert profile.workers_for('duckdb_sql') == budget
    assert profile.workers_for('unordered_copy') == min(8, budget)


@pytest.mark.parametrize('field', ['cpu_budget', 'max_workers', 'memory_budget_bytes', 'arrow_cpu_threads', 'per_host_limit'])
@pytest.mark.parametrize('invalid', [True, 2.5, '4', 0, -1])
def test_resource_counts_are_exact_positive_integers(field, invalid):
    with pytest.raises((TypeError, ValueError)):
        policy.resolve_resource_profile(replace(policy.ResourceProfile(), **{field: invalid}))


def test_environment_explicit_overrides_affinity_and_memory(monkeypatch):
    monkeypatch.setenv(policy.RESOURCE_PROFILE_ENV_VAR, 'cpu6')
    monkeypatch.setenv(policy.TEMP_DIR_ENV_VAR, '/tmp/resource-profile-test')
    monkeypatch.setattr(policy, 'available_cpus', lambda: 3)
    assert policy.resolve_resource_profile(None).cpu_budget == 6
    profile = policy.resolve_resource_profile(policy.ResourceProfile(cpu_budget=8, max_workers=7, memory_budget_bytes=1024))
    assert profile.cpu_budget == 8 and profile.effective_cpu_budget == 3
    assert profile.max_workers == 7
    assert profile.queue_bytes == 128
    assert profile.temp_dir == '/tmp/resource-profile-test'
    assert policy.resolve_resource_profile('tor_low_mem').workers_for('ordered_copy') == 1


def test_memory_budget_has_portable_fallback(monkeypatch):
    def absent(*args, **kwargs):
        raise OSError('unavailable')
    monkeypatch.setattr(policy.Path, 'read_text', absent)
    assert policy.default_memory_budget() == 512 * 1024**2


def test_default_profile_is_cpu16(monkeypatch):
    monkeypatch.delenv(policy.RESOURCE_PROFILE_ENV_VAR, raising=False)
    assert policy.resolve_resource_profile(None).name == 'cpu16'
