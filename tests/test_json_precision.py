"""Installing speed must not change integer values or Python types."""

import json

import pytest

from intermine314.util import json as codec


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
