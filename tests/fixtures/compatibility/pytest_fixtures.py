"""Factories defer restored API and analytics imports until a test requests them."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from importlib import import_module
from io import StringIO

import pytest

from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession, fixture_bytes


@pytest.fixture
def model_xml():
    return fixture_bytes("model.xml").decode("utf-8")


@pytest.fixture
def protocol_payloads():
    return {name: fixture_bytes(name) for name in (
        "rows-modern.json", "rows-legacy.json", "objects-nested.json", "templates.xml",
        "lists.json", "widgets.json", "job-created.json", "job-pending.json",
        "job-success.json", "job-results.json", "registry.json",
        "version-modern.txt", "version-legacy.txt", "version-release.txt",
    )}


@pytest.fixture
def offline_session_factory():
    """Create fresh transports; route overrides support errors and method tests."""
    def create(*, version=8, rows=None, routes=None):
        session = FixtureSession.service(version=version, rows=rows)
        session.routes.update(routes or {})
        return session
    return create


@pytest.fixture
def native_service_factory(offline_session_factory):
    services = []

    def create(*, session=None, **kwargs):
        from intermine314.service.service import Service

        service = Service(SERVICE_ROOT, session=session if session is not None else offline_session_factory(), **kwargs)
        services.append(service)
        return service

    yield create
    for service in services:
        service.close()


@pytest.fixture
def legacy_service_factory(offline_session_factory):
    """Not invoked until the legacy facade/profile is implemented in later tasks."""
    services = []

    def create(*, session=None, **kwargs):
        service_class = import_module("intermine314.webservice").Service
        kwargs.setdefault("compatibility", "legacy")
        service = service_class(SERVICE_ROOT, session=session if session is not None else offline_session_factory(), **kwargs)
        services.append(service)
        return service

    yield create
    for service in services:
        service.close()


@pytest.fixture
def csv_input_factory():
    """Only in-memory CSV input; String IDs must be read with explicit schema."""
    def create():
        return StringIO(
            'identifier,name,large_integer,active,amount,observed_at\n'
            '0007,"Müller, Ada",9007199254740993,true,123456789.1234,2026-01-02T03:04:05\n'
            '0002,,9007199254740995,false,,2026-01-01T00:00:00\n'
            '0010,"line\nbreak",-9007199254740993,,,\n'
        )
    return create


@pytest.fixture
def typed_parquet(tmp_path):
    """Generated on demand; importing/collecting fixtures needs no analytics extra."""
    polars = import_module("polars")
    frame = polars.DataFrame({
        "identifier": ["0007", "0002", "0010"],
        "name": ["Müller, Ada", None, "line\nbreak"],
        "large_integer": [9007199254740993, 9007199254740995, -9007199254740993],
        "active": [True, False, None],
        "amount": [Decimal("123456789.1234"), None, Decimal("-0.0001")],
        "observed_at": [datetime(2026, 1, 2, 3, 4, 5), datetime(2026, 1, 1), None],
    }, schema={
        "identifier": polars.String, "name": polars.String,
        "large_integer": polars.Int64, "active": polars.Boolean,
        "amount": polars.Decimal(precision=20, scale=4), "observed_at": polars.Datetime("us"),
    })
    path = tmp_path / "typed-rows.parquet"
    frame.write_parquet(path)
    return path
