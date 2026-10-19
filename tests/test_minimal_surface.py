"""Behavioral structure not expressible as signatures or export bindings."""

from importlib import import_module

from intermine314.query import Query, Template
from intermine314.query.constraints import ConstraintFactory
from intermine314.service import Service


def test_service_list_manager_surface():
    assert hasattr(Service, "list_manager")
    assert "create_list" in Service.LIST_MANAGER_METHODS


def test_constraint_families_and_template_inheritance():
    assert len(ConstraintFactory.CONSTRAINT_CLASSES) == 9
    assert issubclass(Template, Query)


def test_result_class_names():
    results = import_module("intermine314.results")
    for name in ("ResultObject", "EnrichmentLine"):
        assert getattr(results, name).__name__ == name


def test_list_manager_operations():
    manager = import_module("intermine314.lists").ListManager
    for name in ("union", "intersect", "xor", "subtract"):
        assert hasattr(manager, name)


def test_template_metadata_properties_and_service_paths():
    for name in ("templates", "all_templates", "all_templates_names"):
        assert isinstance(getattr(Service, name), property)
    assert Service.TEMPLATEQUERY_PATH == "/template/results"
    assert Service.TEMPLATES_PATH == "/templates"
    assert Service.ALL_TEMPLATES_PATH == "/alltemplates"
    assert Service.IDS_PATH == "/ids"
