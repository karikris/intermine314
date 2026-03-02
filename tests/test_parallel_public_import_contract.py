from intermine314 import parallel
from intermine314.parallel import policy


def test_parallel_package_public_exports_are_explicit_and_canonical():
    expected = {
        "VALID_ORDER_MODES",
        "VALID_PARALLEL_PAGINATION",
        "VALID_PARALLEL_PROFILES",
        "resolve_parallel_strategy",
        "normalize_order_mode",
        "apply_parallel_profile",
    }
    assert set(parallel.__all__) == expected

    assert parallel.VALID_ORDER_MODES is policy.VALID_ORDER_MODES
    assert parallel.VALID_PARALLEL_PAGINATION is policy.VALID_PARALLEL_PAGINATION
    assert parallel.VALID_PARALLEL_PROFILES is policy.VALID_PARALLEL_PROFILES
    assert parallel.resolve_parallel_strategy is policy.resolve_parallel_strategy
    assert parallel.normalize_order_mode is policy.normalize_order_mode
    assert parallel.apply_parallel_profile is policy.apply_parallel_profile
    assert not hasattr(parallel, "require_int")
    assert not hasattr(parallel, "require_positive_int")
    assert not hasattr(parallel, "require_non_negative_int")
    assert not hasattr(parallel, "resolve_prefetch")
    assert not hasattr(parallel, "resolve_inflight_limit")
    assert not hasattr(parallel, "normalize_mode")
    assert not hasattr(parallel, "estimate_page_count")
    assert not hasattr(parallel, "iter_offset_pages")
