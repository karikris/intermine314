from intermine314.model.operators import Column
from intermine314.service.service import Registry, Service


def test_removed_service_aliases_are_not_present():
    assert not hasattr(Service, "new_query")
    assert not hasattr(Service, "tor")
    assert not hasattr(Registry, "tor")


def test_removed_column_filter_alias_is_not_present():
    assert not hasattr(Column, "filter")
