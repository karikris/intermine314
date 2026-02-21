import pytest

from intermine314.parallel import policy


def test_require_int_contracts():
    assert policy.require_int("value", 3) == 3
    with pytest.raises(TypeError):
        policy.require_int("value", True)
    with pytest.raises(TypeError):
        policy.require_int("value", "3")


def test_positive_and_non_negative_contracts():
    assert policy.require_positive_int("value", 1) == 1
    assert policy.require_non_negative_int("value", 0) == 0
    with pytest.raises(ValueError):
        policy.require_positive_int("value", 0)
    with pytest.raises(ValueError):
        policy.require_non_negative_int("value", -1)


def test_resolve_parallel_strategy_contracts():
    valid = frozenset({"auto", "offset", "keyset"})
    assert policy.resolve_parallel_strategy(
        "offset",
        0,
        None,
        valid_parallel_pagination=valid,
        keyset_auto_min_size=1000,
    ) == "offset"
    assert policy.resolve_parallel_strategy(
        "keyset",
        0,
        100,
        valid_parallel_pagination=valid,
        keyset_auto_min_size=1000,
    ) == "keyset"
    assert policy.resolve_parallel_strategy(
        "keyset",
        10,
        100,
        valid_parallel_pagination=valid,
        keyset_auto_min_size=1000,
    ) == "offset"
    assert policy.resolve_parallel_strategy(
        "auto",
        0,
        None,
        valid_parallel_pagination=valid,
        keyset_auto_min_size=1000,
    ) == "keyset"
    assert policy.resolve_parallel_strategy(
        "auto",
        0,
        5000,
        valid_parallel_pagination=valid,
        keyset_auto_min_size=1000,
    ) == "keyset"
    assert policy.resolve_parallel_strategy(
        "auto",
        0,
        100,
        valid_parallel_pagination=valid,
        keyset_auto_min_size=1000,
    ) == "offset"
    with pytest.raises(ValueError):
        policy.resolve_parallel_strategy(
            "unknown",
            0,
            None,
            valid_parallel_pagination=valid,
            keyset_auto_min_size=1000,
        )


def test_normalize_order_mode_contracts():
    valid = frozenset({"ordered", "unordered", "window", "mostly_ordered"})
    assert policy.normalize_order_mode(None, default_order_mode="ordered", valid_order_modes=valid) == "ordered"
    assert policy.normalize_order_mode(True, default_order_mode="ordered", valid_order_modes=valid) == "ordered"
    assert policy.normalize_order_mode(False, default_order_mode="ordered", valid_order_modes=valid) == "unordered"
    assert policy.normalize_order_mode("window", default_order_mode="ordered", valid_order_modes=valid) == "window"
    with pytest.raises(ValueError):
        policy.normalize_order_mode("invalid", default_order_mode="ordered", valid_order_modes=valid)


def test_apply_parallel_profile_contracts():
    valid_profiles = frozenset({"default", "large_query", "unordered", "mostly_ordered"})
    assert policy.apply_parallel_profile(
        None,
        None,
        False,
        default_profile="default",
        valid_parallel_profiles=valid_profiles,
    ) == ("default", True, False)
    assert policy.apply_parallel_profile(
        "large_query",
        None,
        False,
        default_profile="default",
        valid_parallel_profiles=valid_profiles,
    ) == ("large_query", "window", True)
    assert policy.apply_parallel_profile(
        "unordered",
        None,
        True,
        default_profile="default",
        valid_parallel_profiles=valid_profiles,
    ) == ("unordered", False, True)
    assert policy.apply_parallel_profile(
        "mostly_ordered",
        None,
        False,
        default_profile="default",
        valid_parallel_profiles=valid_profiles,
    ) == ("mostly_ordered", "window", False)
    assert policy.apply_parallel_profile(
        "default",
        "unordered",
        False,
        default_profile="default",
        valid_parallel_profiles=valid_profiles,
    ) == ("default", "unordered", False)
    with pytest.raises(ValueError):
        policy.apply_parallel_profile(
            "invalid",
            None,
            False,
            default_profile="default",
            valid_parallel_profiles=valid_profiles,
        )


def test_resolve_prefetch_contracts():
    assert policy.resolve_prefetch(
        None,
        max_workers=4,
        large_query_mode=False,
        default_parallel_prefetch=8,
    ) == 8
    assert policy.resolve_prefetch(
        None,
        max_workers=4,
        large_query_mode=False,
        default_parallel_prefetch=None,
    ) == 4
    assert policy.resolve_prefetch(
        None,
        max_workers=4,
        large_query_mode=True,
        default_parallel_prefetch=None,
    ) == 8
    assert policy.resolve_prefetch(
        6,
        max_workers=4,
        large_query_mode=True,
        default_parallel_prefetch=None,
    ) == 6
    with pytest.raises(ValueError):
        policy.resolve_prefetch(
            0,
            max_workers=4,
            large_query_mode=False,
            default_parallel_prefetch=None,
        )


def test_resolve_inflight_limit_contracts():
    assert policy.resolve_inflight_limit(None, prefetch=5, default_parallel_inflight_limit=3) == 3
    assert policy.resolve_inflight_limit(None, prefetch=5, default_parallel_inflight_limit=None) == 5
    assert policy.resolve_inflight_limit(4, prefetch=5, default_parallel_inflight_limit=None) == 4
    with pytest.raises(ValueError):
        policy.resolve_inflight_limit(0, prefetch=5, default_parallel_inflight_limit=None)
