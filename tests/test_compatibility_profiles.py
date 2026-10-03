"""Compatibility profiles share the native builder, executor and transport."""

import inspect
import sys
from types import ModuleType, SimpleNamespace
from urllib.parse import parse_qs

import pytest

from intermine314.query import ParallelOptions, Query
from intermine314.query.spec import QuerySpec
from intermine314.service.service import Registry, Service
from intermine314.webservice import Registry as LegacyRegistry
from intermine314.webservice import Service as LegacyService
from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession, fixture_bytes


@pytest.fixture
def restored_model_type(monkeypatch):
    # The actual Model is restored in phase 3; isolate its type inference here.
    module = ModuleType("intermine314.model")
    module.Model = type("Model", (), {"name": "testmodel"})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.Model


@pytest.mark.parametrize("invalid", ["", "Legacy", "other", False, 1, [], {}])
def test_invalid_profiles_fail_before_transport(invalid, offline_session_factory):
    session = offline_session_factory()
    for cls, args in (
        (Service, (SERVICE_ROOT,)),
        (LegacyService, (SERVICE_ROOT,)),
        (Registry, ()),
        (LegacyRegistry, ()),
    ):
        with pytest.raises(ValueError, match="compatibility"):
            cls(*args, session=session, compatibility=invalid)
    with pytest.raises(ValueError, match="compatibility"):
        Query(compatibility=invalid)
    with pytest.raises(ValueError, match="compatibility"):
        QuerySpec(compatibility=invalid)
    assert session.requests == []


@pytest.mark.parametrize(
    "cls,default", [(Service, "native"), (LegacyService, "legacy")]
)
@pytest.mark.parametrize("profile", [None, "native", "legacy"])
def test_service_factory_aliases_propagate_profile(
    cls, default, profile, offline_session_factory
):
    with cls(
        SERVICE_ROOT, session=offline_session_factory(), compatibility=profile
    ) as service:
        expected = default if profile is None else profile
        assert service.compatibility == expected
        for factory in (service.select, service.new_query, service.query):
            query = factory("Employee.name", "Employee.age", "Employee.fullTime")
            assert type(query) is Query
            assert query.compatibility == expected
            assert query.service is service
            if expected == "legacy":
                assert query.root is service.model.get_class("Employee")
                assert query.rootClass is query.root
            else:
                assert query.root == "Employee"
            assert all(isinstance(row, dict) for row in query.results())
        assert service._owns_session is False


def test_profile_keyword_preserves_original_positional_constructor(
    offline_session_factory,
):
    session = offline_session_factory()
    with LegacyService(
        SERVICE_ROOT,
        None,
        None,
        None,
        3,
        True,
        9,
        None,
        session,
        False,
        False,
        True,
        False,
        False,
        "profile-test",
    ) as service:
        assert service.compatibility == "legacy"
        assert service.prefetch_depth == 3
        assert service.prefetch_id_only is True
        assert service.request_timeout == 9
        assert service.verify_tls is False
        assert service.user_agent == "profile-test"
    for cls in (Service, Registry, Query, LegacyService, LegacyRegistry):
        assert (
            inspect.signature(cls).parameters["compatibility"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
    assert inspect.signature(Service) == inspect.signature(LegacyService)
    assert inspect.signature(Registry) == inspect.signature(LegacyRegistry)


def test_query_profile_inference_priority(
    restored_model_type, native_service_factory, legacy_service_factory
):
    model = restored_model_type()
    native = native_service_factory()
    legacy = legacy_service_factory()
    assert Query().compatibility == "native"
    assert Query(SimpleNamespace(name="testmodel")).compatibility == "native"
    assert Query(service=legacy).compatibility == "legacy"
    assert Query(model, service=native).compatibility == "legacy"
    assert (
        Query(model, service=legacy, compatibility="native").compatibility == "native"
    )
    assert Query(service=native, compatibility="legacy").compatibility == "legacy"
    native._query_model = model
    legacy._query_model = model
    assert native.select("Employee.name").compatibility == "native"
    assert legacy.select("Employee.name").compatibility == "legacy"


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_clone_has_independent_codes_constraints_views_joins_and_sort_orders(profile):
    query = Query(root="Employee", compatibility=profile).add_view("name", "age")
    query.name, query.description = "original", "description"
    query.add_constraint("name", "=", "Alice")
    query.add_sort_order("name", "DESC")
    query.add_join("department", "OUTER")
    left, right = query.clone(), query.clone()
    assert left.compatibility == right.compatibility == profile
    assert left.constraint_factory is not query.constraint_factory
    assert left.constraint_factory is not right.constraint_factory
    left.add_constraint("age", ">", 20)
    right.add_constraint("age", "<", 50)
    query.add_constraint("age", "=", 35)
    assert [c.code for c in left.constraints] == ["A", "B"]
    assert [c.code for c in right.constraints] == ["A", "B"]
    assert [c.code for c in query.constraints] == ["A", "B"]
    assert left.get_constraint("B").value == 20
    assert right.get_constraint("B").value == 50
    assert query.get_constraint("B").value == 35
    left.get_constraint("A").value = "Bob"
    left.views.append("Employee.id")
    left.joins[0].style = "INNER"
    left._sort_order_list.sort_orders[0].order = "asc"
    assert query.get_constraint("A").value == right.get_constraint("A").value == "Alice"
    assert query.views == right.views == ["Employee.name", "Employee.age"]
    assert query.joins[0].style == right.joins[0].style == "OUTER"
    assert (
        str(query.get_sort_order())
        == str(right.get_sort_order())
        == "Employee.name desc"
    )
    assert left.name == "original" and left.description == "description"


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_where_triple_tuple_and_keyword_forms_clone_without_mutation(profile):
    query = Query(root="Employee", compatibility=profile).add_view("name", "age")
    query.add_constraint("name", "!=", "excluded")
    triple = query.where("age", ">", 20)
    tuples = query.where(("age", ">", 20), ("name", "LIKE", "A%"))
    keywords = query.where(age=20, name=["Alice", "Bob"])
    assert list(query.constraint_dict) == ["A"]
    assert triple.get_constraint("B").path == "Employee.age"
    assert triple.get_constraint("B").value == 20
    assert list(tuples.constraint_dict) == ["A", "B", "C"]
    assert keywords.get_constraint("C").values == ["Alice", "Bob"]
    for cloned in (triple, tuples, keywords):
        assert cloned.compatibility == profile
        assert cloned.get_constraint("A").value == "excluded"
    assert query.constraint_factory._code_index == 1


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_query_aliases_match_target_mutation_and_count(
    profile, native_service_factory, offline_session_factory
):
    session = offline_session_factory(rows=b"3\n")
    service = native_service_factory(session=session, compatibility=profile)
    query = service.new_query("Employee.name")
    for alias, path in (
        ("add_column", "age"),
        ("add_columns", ["fullTime"]),
        ("add_views", ("id",)),
        ("add_to_select", "department.name"),
    ):
        assert getattr(query, alias)(path) is query
    assert query.views == [
        "Employee.name",
        "Employee.age",
        "Employee.fullTime",
        "Employee.id",
        "Employee.department.name",
    ]
    assert query.order_by("age", "DESC") is query
    assert str(query.get_sort_order()) == "Employee.age desc"
    assert query.size() == query.count() == 3
    for request in session.requests:
        if request.path.endswith("/query/results"):
            assert parse_qs(request.data.decode())["format"] == ["count"]


def test_profiles_preserve_spec_executor_and_wire_xml(
    native_service_factory, legacy_service_factory
):
    queries = [
        factory()
        .select("Employee.name", "Employee.age", "Employee.fullTime")
        .where("age", ">", 20)
        for factory in (native_service_factory, legacy_service_factory)
    ]
    native, legacy = queries
    assert native.to_xml() == legacy.to_xml()
    for query, profile in zip(queries, ("native", "legacy"), strict=True):
        spec = query.to_spec()
        assert spec.compatibility == profile
        assert spec.root_class == "Employee" and isinstance(spec.root_class, str)
        executor = query.service.execute(spec)
        assert executor.compatibility == profile
        assert executor.spec is spec
        assert executor.to_query_params() == query.to_query_params()
        assert list(iter(executor.results())) == list(iter(query.results()))
        payloads = [
            parse_qs(request.data.decode())["query"][0]
            for request in query.service.opener._session.requests
            if request.path.endswith("/query/results")
        ]
        assert payloads == [query.to_xml(), query.to_xml()]
    # Profile is carried by the spec, including execution on another service.
    assert native.service.execute(legacy.to_spec()).compatibility == "legacy"


def test_spec_serialization_normalizes_class_name_at_boundary():
    query = Query(validate=False, compatibility="legacy")
    query.root = SimpleNamespace(name="Employee")
    assert query.to_spec().root_class == "Employee"
    assert QuerySpec().compatibility == "native"


@pytest.mark.parametrize("tor", [False, True])
@pytest.mark.parametrize(
    "cls,requested,expected_cls,profile",
    [
        (Registry, None, Service, "native"),
        (LegacyRegistry, None, LegacyService, "legacy"),
        (Registry, "legacy", LegacyService, "legacy"),
        (LegacyRegistry, "native", Service, "native"),
    ],
)
def test_registry_chooses_profile_service_and_shares_transport(
    cls, requested, expected_cls, profile, tor
):
    session = FixtureSession.service()
    session.routes[("GET", "/service/instances")] = fixture_bytes("registry.json")
    with cls(
        "https://registry.example",
        session=session,
        compatibility=requested,
        verify_tls=False,
        request_timeout=19,
        user_agent="profile-test",
        tor=tor,
        proxy_url="socks5h://localhost:9150" if tor else None,
    ) as registry:
        service = registry["offlinemine"]
        assert type(service) is expected_cls
        assert service.compatibility == registry.compatibility == profile
        assert service.opener._session is registry._session is session
        assert not service._owns_session and not registry._owns_session
        assert service.verify_tls is False
        assert service.tor is registry.tor is tor
        assert service.proxy_url == registry.proxy_url
        assert service.strict_tor_proxy_scheme == registry.strict_tor_proxy_scheme
        assert service.allow_http_over_tor == registry.allow_http_over_tor
        assert service.request_timeout == 19 and service.user_agent == "profile-test"
        assert registry["OFFLINEMINE"] is service
        assert registry.service_cache_metrics()["cache_hits"] == 1
    assert service._closed and all(response.closed for response in session.responses)
    assert session.close_calls == 0


def test_native_registry_factory_retains_service_monkeypatch(monkeypatch):
    from intermine314.service import service as implementation

    created = []

    def replacement(root, **kwargs):
        created.append((root, kwargs))
        return SimpleNamespace(compatibility=kwargs["compatibility"])

    monkeypatch.setattr(implementation, "Service", replacement)
    session = FixtureSession.service()
    session.routes[("GET", "/service/instances")] = fixture_bytes("registry.json")
    with Registry("https://registry.example", session=session) as registry:
        assert registry["offlinemine"].compatibility == "native"
    assert len(created) == 1 and created[0][1]["session"] is session


@pytest.mark.parametrize("profile", ["native", "legacy"])
def test_parallel_pages_keep_profile_in_shared_executor(
    profile, native_service_factory, monkeypatch
):
    service = native_service_factory(compatibility=profile)
    query = service.select("Employee.name", "Employee.age", "Employee.fullTime")
    executed = []
    original_execute = service.execute

    def execute(spec):
        executor = original_execute(spec)
        executed.append((spec.compatibility, executor.compatibility))
        return executor

    monkeypatch.setattr(service, "execute", execute)
    options = ParallelOptions(page_size=1, max_workers=2, ordered=True)
    rows = list(query.run_parallel(size=2, parallel_options=options))
    assert rows == [
        {"Employee.name": "foo", "Employee.age": "bar", "Employee.fullTime": "baz"},
        {"Employee.name": "foo", "Employee.age": "bar", "Employee.fullTime": "baz"},
    ]
    assert executed == [(profile, profile), (profile, profile)]
    params = [
        parse_qs(request.data.decode())
        for request in service.opener._session.requests
        if request.path.endswith("/query/results")
    ]
    assert sorted(int(param["start"][0]) for param in params) == [0, 1]
    assert all(
        param["query"] == [query.to_xml()] and param["size"] == ["1"]
        for param in params
    )
    assert all(response.closed for response in service.opener._session.responses)


@pytest.mark.parametrize("cls", [Registry, LegacyRegistry])
def test_registry_owned_session_closes_once_after_cached_services(cls, monkeypatch):
    from intermine314.service import session as transport

    session = FixtureSession.service()
    session.routes[("GET", "/service/instances")] = fixture_bytes("registry.json")
    monkeypatch.setattr(transport, "build_session", lambda **kwargs: session)
    registry = cls("https://registry.example")
    service = registry["offlinemine"]
    assert registry._owns_session and not service._owns_session
    assert registry._session is service.opener._session is session
    registry.clear_cache()
    assert service._closed and session.close_calls == 0
    replacement = registry["offlinemine"]
    assert replacement is not service and not replacement._owns_session
    registry.close()
    registry.close()
    assert replacement._closed and session.close_calls == 1
