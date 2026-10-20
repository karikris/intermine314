"""Fresh-install configuration, with one canonical scheduling policy."""
import tomllib
from importlib.resources import files

import pytest

from intermine314.config.runtime_defaults import (
    clear_runtime_defaults_cache,
    get_runtime_defaults,
    parse_runtime_defaults,
)


@pytest.mark.parametrize(('section', 'key'), [
    ('query_defaults', key) for key in (
        'default_parallel_workers', 'default_parallel_pagination', 'default_parallel_profile',
        'default_parallel_ordered_mode', 'default_large_query_mode', 'default_parallel_prefetch',
        'default_parallel_inflight_limit')
] + [('registry_defaults', key) for key in (
    'default_workers_tier', 'server_limited_workers_tier', 'full_workers_tier',
    'default_production_profile_switch_rows')])
def test_removed_scheduling_keys_require_migration(section, key):
    with pytest.raises(ValueError, match=rf'{section}\.{key}.*ResourceProfile.*ParallelOptions'):
        parse_runtime_defaults({section: {key: 4}})


def test_override_migration_errors_are_not_hidden_by_schema_fallback(tmp_path, monkeypatch):
    config = tmp_path / 'runtime.toml'
    config.write_text('[meta]\nschema_version = 999\n[query_defaults]\ndefault_parallel_workers = 4\n')
    monkeypatch.setenv('INTERMINE314_RUNTIME_DEFAULTS_PATH', str(config))
    clear_runtime_defaults_cache()
    try:
        with pytest.raises(ValueError, match='Removed scheduling keys'):
            get_runtime_defaults()
        config.write_text('[query_defaults]\ndefault_parallel_page_size = 234\n')
        clear_runtime_defaults_cache()
        assert get_runtime_defaults().query_defaults.default_parallel_page_size == 234
        assert get_runtime_defaults().registry_defaults.default_registry_service_cache_size == 32
    finally:
        clear_runtime_defaults_cache()


def test_packaged_configuration_and_fresh_override_match_supported_schema(tmp_path, monkeypatch):
    from dataclasses import fields

    from intermine314.config.runtime_defaults import QueryDefaults, RegistryDefaults

    payload = tomllib.loads(files("intermine314.config").joinpath("runtime-defaults.toml").read_text())
    for section, schema in (("query_defaults", QueryDefaults), ("registry_defaults", RegistryDefaults)):
        assert set(payload[section]) <= {field.name for field in fields(schema)}
    packaged = parse_runtime_defaults(payload)
    assert packaged.query_defaults.default_parallel_page_size == 1000
    assert packaged.registry_defaults.default_registry_service_cache_size == 32
    override = tmp_path / "runtime.toml"
    override.write_text("[query_defaults]\ndefault_parallel_page_size = 234\ndefault_batch_size = 321\n")
    monkeypatch.setenv("INTERMINE314_RUNTIME_DEFAULTS_PATH", str(override))
    clear_runtime_defaults_cache()
    try:
        configured = get_runtime_defaults()
        assert configured.query_defaults.default_parallel_page_size == 234
        assert configured.query_defaults.default_batch_size == 321
        assert configured.service_defaults == packaged.service_defaults
        assert configured.registry_defaults == packaged.registry_defaults
    finally:
        clear_runtime_defaults_cache()


@pytest.mark.parametrize('legacy_profile', ['default', 'large_query', 'unordered'])
def test_explicit_options_then_resource_fields_then_stage_defaults(legacy_profile):
    from intermine314.export import ResourceProfile, ResourceRuntime
    from intermine314.query import ParallelOptions, Query

    query = Query(root='Gene')
    resource = ResourceProfile(cpu_budget=8, max_workers=3, prefetch=5, inflight_limit=6, ordered=True)
    with ResourceRuntime(resource) as runtime, runtime.activate():
        resolved = query._resolve_parallel_options(start=0, size=10, options=ParallelOptions(profile=legacy_profile))
        assert (resolved.max_workers, resolved.prefetch, resolved.inflight_limit, resolved.order_mode) == (3, 5, 6, 'ordered')
        resolved = query._resolve_parallel_options(start=0, size=10, options=ParallelOptions(
            profile=legacy_profile, max_workers=2, prefetch=4, inflight_limit=7, ordered=False))
        assert (resolved.max_workers, resolved.prefetch, resolved.inflight_limit, resolved.order_mode) == (2, 4, 7, 'unordered')
        assert resolved.strategy == 'offset' and not resolved.tor_aware_defaults_applied
    with ResourceRuntime('cpu2') as runtime, runtime.activate():
        resolved = query._resolve_parallel_options(start=0, size=10, options=ParallelOptions(prefetch=4))
        assert (resolved.max_workers, resolved.prefetch, resolved.inflight_limit) == (2, 4, 4)
        with pytest.raises(ValueError, match='positive integer'):
            query._resolve_parallel_options(start=0, size=10, options=ParallelOptions(max_workers=0))


@pytest.mark.parametrize(('explicit', 'configured', 'expected'), [(7, 9, 7), (None, 9, 9), (None, None, 3)])
def test_published_window_utilities_retain_explicit_configured_and_worker_fallbacks(explicit, configured, expected):
    from intermine314.parallel.policy import resolve_inflight_limit, resolve_prefetch

    assert resolve_prefetch(explicit, max_workers=3, large_query_mode=True,
                            default_parallel_prefetch=configured) == expected
    assert resolve_inflight_limit(explicit, prefetch=3,
                                  default_parallel_inflight_limit=configured) == expected


@pytest.mark.parametrize('invalid', [0, True, '4'])
def test_published_window_utilities_reject_invalid_explicit_counts(invalid):
    from intermine314.parallel.policy import resolve_inflight_limit, resolve_prefetch

    for resolve, options in ((resolve_prefetch, dict(max_workers=3, large_query_mode=False, default_parallel_prefetch=None)),
                             (resolve_inflight_limit, dict(prefetch=3, default_parallel_inflight_limit=None))):
        with pytest.raises((TypeError, ValueError)):
            resolve(invalid, **options)
