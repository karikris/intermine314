"""Installing speed must not change integer values or Python types."""

import json
from decimal import Decimal

import pytest

from intermine314.util import json as codec
from tests.fixtures.compatibility import SERVICE_ROOT, FixtureSession, fixture_bytes


@pytest.mark.parametrize("number", [
    -(2**80 + 1), -(2**63 + 1), -(2**63), -(2**63) + 1,
    2**53 + 1, 2**63, 2**64 - 1, 2**64, 2**64 + 1, 2**80 + 1, 10**100 + 1,
])
@pytest.mark.parametrize("encode", [str, str.encode, lambda s: bytearray(s.encode())])
def test_integer_lexemes_remain_exact(number, encode):
    payload = json.dumps({"rows": [number, {"nested": number}], "label": str(number)})
    actual = codec.json_loads(encode(payload))
    assert actual == json.loads(payload)
    assert type(actual["rows"][0]) is int
    assert type(actual["rows"][1]["nested"]) is int


@pytest.mark.parametrize("payload", [
    b'"18446744073709551617000"',
    b'{"escaped":"\\\"18446744073709551617000\\\""}',
    b'[18446744073709551617000.0, -92233720368547758090e-2]',
    b'[0.18446744073709551617000, 1e18446744073709551617000]',
    b'[-9223372036854775808,18446744073709551615]',
])
def test_guard_ignores_strings_floats_and_exact_boundaries(payload):
    assert not codec._requires_exact_integer_decoder(payload)


def test_guard_uses_standard_decoder_only_for_overflow_tokens(monkeypatch):
    pytest.importorskip("orjson")
    accelerated, exact = [], []
    fast = codec._loads
    standard = codec._stdlib_json.loads
    monkeypatch.setattr(codec, "_loads", lambda p: (accelerated.append(p), fast(p))[1])
    monkeypatch.setattr(codec._stdlib_json, "loads", lambda p: (exact.append(p), standard(p))[1])
    assert codec.json_loads(b'[1,2,"3"]') == [1, 2, "3"]
    assert codec.json_loads(b'[18446744073709551617]') == [2**64 + 1]
    assert len(accelerated) == len(exact) == 1


@pytest.mark.parametrize("profile", ["native", "legacy"])
@pytest.mark.parametrize("number", [2**64 + 1, -(2**80 + 1)])
def test_wire_integer_survives_query_rows_objects_and_parquet(profile, number, tmp_path):
    import polars as pl

    from intermine314.service.service import Service

    rows = b'{"results":[\n[' + str(number).encode() + b']\n],"wasSuccessful":true}\n'
    session = FixtureSession.service(version=35, rows=rows)
    session.routes[("GET", "/service/model")] = fixture_bytes("model.xml").replace(
        b'<attribute name="age" type="int"/>', b'<attribute name="age" type="java.math.BigDecimal"/>',
    )
    with Service(SERVICE_ROOT, session=session, compatibility=profile) as service:
        query = service.select("Employee.age")
        for results in (query.results("dict"), query.rows()):
            row = next(results)
            assert row["Employee.age"] == number
            assert type(row["Employee.age"]) is int
            results.close()
        session.routes[("POST", "/service/query/results")] = (
            b'{"results":[\n{"class":"Employee","objectId":1,"age":' + str(number).encode()
            + b'}\n],"wasSuccessful":true}\n'
        )
        objects = query.results("jsonobjects")
        obj = next(objects)
        assert obj.age == number and type(obj.age) is int
        objects.close()
        session.routes[("POST", "/service/query/results")] = rows
        path = tmp_path / "precise.parquet"
        query.export(path, size=1)
        assert pl.read_parquet(path).item() == Decimal(number)
    assert all(response.closed and response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 0
