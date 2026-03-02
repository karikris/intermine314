from __future__ import annotations

import hashlib
import json

from intermine314.config.loader import load_packaged_mine_parallel_preferences_detailed
import intermine314.registry.mines as mine_registry


def _normalize_registry_from_loaded_payload(payload):
    return {
        "benchmark_profiles": mine_registry._normalize_benchmark_profiles(
            mine_registry._merge_benchmark_profiles(mine_registry.DEFAULT_BENCHMARK_PROFILES, payload)
        ),
        "production_profiles": mine_registry._normalize_production_profiles(
            mine_registry._merge_production_profiles(mine_registry.DEFAULT_PRODUCTION_PROFILES, payload)
        ),
        "mines": mine_registry._normalize_mines_for_matching(
            mine_registry._merge_mines(mine_registry.DEFAULT_REGISTRY, payload)
        ),
    }


def _normalize_registry_from_fallback():
    return {
        "benchmark_profiles": mine_registry._normalize_benchmark_profiles(mine_registry.DEFAULT_BENCHMARK_PROFILES),
        "production_profiles": mine_registry._normalize_production_profiles(mine_registry.DEFAULT_PRODUCTION_PROFILES),
        "mines": mine_registry._normalize_mines_for_matching(mine_registry.DEFAULT_REGISTRY),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def test_packaged_preferences_are_in_strict_parity_with_python_fallback_defaults():
    packaged = load_packaged_mine_parallel_preferences_detailed()

    assert packaged["ok"] is True
    payload = packaged.get("payload")
    assert isinstance(payload, dict)

    normalized_packaged = _normalize_registry_from_loaded_payload(payload)
    normalized_fallback = _normalize_registry_from_fallback()

    assert normalized_packaged == normalized_fallback


def test_packaged_preferences_normalized_snapshot_hash_is_stable():
    packaged = load_packaged_mine_parallel_preferences_detailed()

    assert packaged["ok"] is True
    payload = packaged.get("payload")
    assert isinstance(payload, dict)

    normalized = _normalize_registry_from_loaded_payload(payload)
    snapshot_json = json.dumps(_json_safe(normalized), sort_keys=True, separators=(",", ":"))
    snapshot_hash = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()

    assert snapshot_hash == "34bf1467460211646cf1b75bb0879c64b97ee974821584bcf812d11167ed2608"
