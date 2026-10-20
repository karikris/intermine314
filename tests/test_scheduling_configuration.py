import pytest

from intermine314.config.runtime_defaults import (
    _REMOVED_SCHEDULING_KEYS,
    clear_runtime_defaults_cache,
    get_runtime_defaults,
    parse_runtime_defaults,
)


@pytest.mark.parametrize(('section', 'key'), [(section, key) for section, keys in _REMOVED_SCHEDULING_KEYS.items() for key in keys])
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
