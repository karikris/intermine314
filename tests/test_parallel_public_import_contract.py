from intermine314 import parallel
from intermine314.parallel import helpers
from intermine314.parallel import policy


def test_parallel_package_public_exports_are_explicit_and_canonical():
    expected = {
        "VALID_ORDER_MODES",
        "VALID_PARALLEL_PAGINATION",
        "VALID_PARALLEL_PROFILES",
        "normalize_mode",
        "estimate_page_count",
        "iter_offset_pages",
        "require_int",
        "require_positive_int",
        "require_non_negative_int",
        "resolve_parallel_strategy",
        "normalize_order_mode",
        "apply_parallel_profile",
        "resolve_prefetch",
        "resolve_inflight_limit",
    }
    assert set(parallel.__all__) == expected

    assert parallel.VALID_ORDER_MODES is policy.VALID_ORDER_MODES
    assert parallel.VALID_PARALLEL_PAGINATION is policy.VALID_PARALLEL_PAGINATION
    assert parallel.VALID_PARALLEL_PROFILES is policy.VALID_PARALLEL_PROFILES
    assert parallel.require_int is policy.require_int
    assert parallel.require_positive_int is policy.require_positive_int
    assert parallel.require_non_negative_int is policy.require_non_negative_int
    assert parallel.resolve_parallel_strategy is policy.resolve_parallel_strategy
    assert parallel.normalize_order_mode is policy.normalize_order_mode
    assert parallel.apply_parallel_profile is policy.apply_parallel_profile
    assert parallel.resolve_prefetch is policy.resolve_prefetch
    assert parallel.resolve_inflight_limit is policy.resolve_inflight_limit
    assert parallel.normalize_mode is helpers.normalize_mode
    assert parallel.estimate_page_count is helpers.estimate_page_count
    assert parallel.iter_offset_pages is helpers.iter_offset_pages
