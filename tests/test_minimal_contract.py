from __future__ import annotations

import inspect
import json
from importlib import import_module
from pathlib import Path


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "contracts" / "minimal_public_contract.json"


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _resolve_dotted(name: str):
    parts = name.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            module = import_module(module_name)
        except Exception:
            continue
        obj = module
        for attr in parts[index:]:
            obj = getattr(obj, attr)
        return obj
    raise AssertionError(f"Could not resolve symbol: {name}")


def _has_dotted(name: str) -> bool:
    parts = name.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            module = import_module(module_name)
        except Exception:
            continue
        obj = module
        for attr in parts[index:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


def _assert_subsequence(sequence: list[str], required: list[str], label: str) -> None:
    pos = 0
    for item in required:
        try:
            found = sequence.index(item, pos)
        except ValueError as exc:
            raise AssertionError(f"{label}: missing required parameter {item!r}") from exc
        pos = found + 1


def test_public_entrypoints_match_contract_exactly():
    contract = _load_contract()
    public_entrypoints = contract["public_entrypoints"]
    for module_name, expected_symbols in public_entrypoints.items():
        module = import_module(module_name)
        expected = set(expected_symbols)
        exported = set(getattr(module, "__all__", []))
        assert exported == expected
        for symbol in expected_symbols:
            assert hasattr(module, symbol), f"{module_name} missing symbol {symbol}"


def test_required_signatures_contain_minimal_parameters_in_order():
    contract = _load_contract()
    for symbol, required_params in contract["required_signatures"].items():
        target = _resolve_dotted(symbol)
        param_names = list(inspect.signature(target).parameters.keys())
        _assert_subsequence(param_names, list(required_params), symbol)


def test_required_dataclass_fields_match_contract():
    contract = _load_contract()
    for symbol, expected_fields in contract["required_dataclass_fields"].items():
        target = _resolve_dotted(symbol)
        assert hasattr(target, "__dataclass_fields__")
        field_names = list(target.__dataclass_fields__.keys())
        for field in expected_fields:
            assert field in field_names


def test_forbidden_symbols_are_absent():
    contract = _load_contract()
    for symbol in contract["forbidden_symbols"]:
        assert not _has_dotted(symbol), f"forbidden symbol still present: {symbol}"
