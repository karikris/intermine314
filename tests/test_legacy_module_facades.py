"""Restored import paths share implementation and preserve exception identities."""

import importlib
import inspect
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REEXPORTS = json.loads((Path(__file__).resolve().parents[1] / "docs/analysis/legacy-reexport-contract.json").read_text())["bindings"]


@pytest.mark.parametrize("binding", REEXPORTS, ids=lambda b: b["module"] + "." + b["name"])
def test_historical_owned_reexports(binding):
    module = importlib.import_module(binding["module"])
    target = importlib.import_module(binding["target"])
    expected = target if binding["kind"] == "module" else getattr(target, binding["name"])
    assert getattr(module, binding["name"]) is expected
    assert binding["name"] in module.__all__
    assert binding["name"] in dir(module)


@pytest.mark.parametrize(
    "facade,target,names",
    [
        ("webservice", "service.service", ("ensure_str",)),
        (
            "constraints",
            "query.constraints",
            (
                "Constraint",
                "CodedConstraint",
                "UnaryConstraint",
                "BinaryConstraint",
                "MultiConstraint",
                "ListConstraint",
                "LoopConstraint",
                "TernaryConstraint",
                "RangeConstraint",
                "IsaConstraint",
                "SubClassConstraint",
                "ConstraintFactory",
            ),
        ),
        (
            "pathfeatures",
            "query.pathfeatures",
            ("PathFeature", "Join", "SortOrder", "SortOrderList"),
        ),
        (
            "results",
            "service.session",
            (
                "JSONIterator",
                "ResultRow",
                "TableResultRow",
                "FlatFileIterator",
                "ResultIterator",
                "InterMineURLOpener",
                "encode_str",
                "decode_binary",
                "encode_dict",
            ),
        ),
        ("errors", "service.errors", ("ServiceError", "WebserviceError")),
        ("lists", "lists.list", ("List",)),
        ("lists", "lists.listmanager", ("ListManager", "ListServiceError")),
        (
            "query",
            "query.builder",
            ("QueryError", "ConstraintError", "QueryParseError", "ResultError"),
        ),
    ],
)
def test_facades_preserve_implementation_identity(facade, target, names):
    public = importlib.import_module("intermine314." + facade)
    native = importlib.import_module("intermine314." + target)
    for name in names:
        assert getattr(public, name) is getattr(native, name)
        assert name in public.__all__
        assert name in dir(public)
    with pytest.raises(AttributeError):
        getattr(public, "not_a_public_symbol")


def test_service_facades_subclass_native_clients_and_preserve_constructor_signatures():
    from intermine314 import webservice
    from intermine314.service import service

    for name in ("Service", "Registry"):
        public = getattr(webservice, name)
        native = getattr(service, name)
        assert public is not native
        assert issubclass(public, native)
        assert inspect.signature(public) == inspect.signature(native)
        assert public._DEFAULT_COMPATIBILITY == "legacy"
        assert native._DEFAULT_COMPATIBILITY == "native"
        assert getattr(webservice, name) is public
        assert name in webservice.__all__ and name in dir(webservice)


def test_facade_imports_are_lazy_and_never_require_analytics():
    script = """
import importlib
import importlib.abc
import logging
import sys
class BlockExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'polars', 'duckdb', 'pyarrow', 'pandas', 'matplotlib', 'plotly', 'intermine'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockExtras())
original_handlers = list(logging.getLogger().handlers)
modules = [importlib.import_module('intermine314.' + name) for name in
           ('webservice', 'constraints', 'pathfeatures', 'results', 'errors', 'query', 'decorators', 'util', 'model', 'lists')]
assert logging.getLogger().handlers == original_handlers
assert 'intermine314.service.service' not in sys.modules
assert 'intermine314.query.builder' not in sys.modules
assert 'intermine314.service.session' not in sys.modules
for module in modules:
    for name in module.__all__:
        getattr(module, name)
"""
    environment = dict(
        os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")
    )
    result = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_exception_facades_catch_native_errors_and_preserve_readable_exception():
    from intermine314.errors import ServiceError, UnimplementedError, WebserviceError
    from intermine314.query import (
        ConstraintError,
        QueryError,
        QueryParseError,
        ResultError,
        builder,
    )
    from intermine314.service import errors
    from intermine314.util import ReadableException

    for native, public in (
        (errors.ServiceError, ServiceError),
        (errors.WebserviceError, WebserviceError),
        (builder.ConstraintError, ConstraintError),
        (builder.QueryError, QueryError),
        (builder.QueryParseError, QueryParseError),
        (builder.ResultError, ResultError),
    ):
        with pytest.raises(public):
            raise native("example")
    assert issubclass(ConstraintError, QueryError)
    assert issubclass(QueryParseError, QueryError)
    assert issubclass(ServiceError, ReadableException)
    assert issubclass(UnimplementedError, Exception)
    assert str(ServiceError("problem", "cause")) == "'problem''cause'"


@pytest.mark.parametrize("version", [7, 8])
def test_webservice_and_result_facades_execute_native_protocols(
    offline_session_factory, version
):
    from intermine314.errors import WebserviceError
    from intermine314.query import ConstraintError
    from intermine314.results import ResultIterator
    from intermine314.webservice import Service
    from tests.fixtures.compatibility import SERVICE_ROOT

    session = offline_session_factory(version=version)
    with Service(SERVICE_ROOT, session=session) as service:
        query = service.select("Employee.name", "Employee.age", "Employee.fullTime")
        results = query.results("dict")
        assert isinstance(results, ResultIterator)
        assert list(iter(results)) == [
            {"Employee.name": "foo", "Employee.age": "bar", "Employee.fullTime": "baz"},
            {"Employee.name": 123, "Employee.age": 1.23, "Employee.fullTime": -1.23},
            {"Employee.name": True, "Employee.age": False, "Employee.fullTime": None},
        ]
        with pytest.raises(ConstraintError):
            query.get_constraint("missing")
        with pytest.raises(WebserviceError):
            service.opener.http_error_400(
                SERVICE_ROOT,
                io.BytesIO(b'{"error":"invalid query"}'),
                400,
                "Bad Request",
                {},
            )
    assert all(response.closed for response in session.responses)
    assert session.close_calls == 0


def test_requires_version_preserves_wrapped_function_and_checks_before_call():
    from intermine314.decorators import requires_version
    from intermine314.errors import ServiceError

    calls = []

    @requires_version(8)
    def feature(self, value, *, suffix="!"):
        """A versioned feature."""
        calls.append(self.version)
        return value + suffix

    assert feature.__name__ == "feature"
    assert feature.__doc__ == "A versioned feature."
    assert inspect.signature(feature) == inspect.signature(feature.__wrapped__)
    with pytest.raises(ServiceError) as raised:
        feature(SimpleNamespace(version=7), "denied")
    assert raised.value.message == "Service must be at version 8, but is at 7"
    assert calls == []
    assert feature(SimpleNamespace(version=8), "accepted", suffix="?") == "accepted?"
    assert feature(SimpleNamespace(version=9), "newer") == "newer!"
    assert calls == [8, 9]


def test_open_anything_paths_xml_bytes_and_borrowed_streams(tmp_path):
    from intermine314.util import openAnything

    path = tmp_path / "model.xml"
    path.write_text("<model>text</model>", encoding="utf-8")
    for source in (path, str(path)):
        with openAnything(source) as opened:
            assert opened.read() == "<model>text</model>"
    with openAnything("<model>é</model>") as opened:
        assert opened.read() == "<model>é</model>"
    with openAnything(b"<model>bytes</model>") as opened:
        assert opened.read() == b"<model>bytes</model>"
    for borrowed in (io.BytesIO(b"model"), io.StringIO("model")):
        assert openAnything(borrowed) is borrowed
        assert not borrowed.closed
        borrowed.close()


def test_open_anything_uses_urlopen_and_preserves_response_ownership(monkeypatch):
    from intermine314.util import openAnything

    response = io.BytesIO(b"<model/>")
    calls = []

    def urlopen(source):
        calls.append(source)
        return response

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert openAnything("https://offline.example/model") is response
    assert calls == ["https://offline.example/model"]
    assert not response.closed
    response.close()


def test_open_anything_file_url(tmp_path):
    from intermine314.util import openAnything

    path = tmp_path / "model.xml"
    path.write_bytes(b"<model/>")
    with openAnything(path.as_uri()) as opened:
        assert opened.read() == b"<model/>"


def test_encode_headers_converts_ascii_text_and_preserves_bytes():
    from intermine314.results import encode_headers

    key, value = b"X-Binary", b"\xff"
    headers = {"Accept": "application/json", key: value}
    encoded = encode_headers(headers)
    assert encoded == {b"Accept": b"application/json", key: value}
    assert headers["Accept"] == "application/json"
    assert next(k for k in encoded if k == key) is key
    assert encoded[key] is value
    with pytest.raises(UnicodeEncodeError):
        encode_headers({"X-Name": "é"})
    with pytest.raises(UnicodeEncodeError):
        encode_headers({"é": "value"})
