import importlib
import sys


def _legacy_alias_finders():
    return [
        finder
        for finder in sys.meta_path
        if finder.__class__.__name__ == "_AliasFinder" and finder.__class__.__module__ == "intermine314"
    ]


def test_import_and_reload_do_not_mutate_meta_path():
    baseline = len(sys.meta_path)
    import intermine314

    assert len(sys.meta_path) == baseline
    for _ in range(5):
        importlib.reload(intermine314)
    assert len(sys.meta_path) == baseline
    assert _legacy_alias_finders() == []
    assert "intermine314.query_export" in sys.modules


def test_legacy_alias_module_maps_to_target_module():
    import intermine314  # noqa: F401

    legacy = importlib.import_module("intermine314.query_export")
    target = importlib.import_module("intermine314.export.parquet")
    assert legacy.write_single_parquet_from_parts is target.write_single_parquet_from_parts
