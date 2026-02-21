from intermine314 import parallel
from intermine314.parallel import policy
from intermine314.query import builder as query_builder


def test_parallel_policy_constants_are_canonical_across_modules():
    assert parallel.VALID_PARALLEL_PAGINATION is policy.VALID_PARALLEL_PAGINATION
    assert parallel.VALID_PARALLEL_PROFILES is policy.VALID_PARALLEL_PROFILES
    assert parallel.VALID_ORDER_MODES is policy.VALID_ORDER_MODES
    assert query_builder.VALID_PARALLEL_PAGINATION is policy.VALID_PARALLEL_PAGINATION
    assert query_builder.VALID_PARALLEL_PROFILES is policy.VALID_PARALLEL_PROFILES
    assert query_builder.VALID_ORDER_MODES is policy.VALID_ORDER_MODES


def test_parallel_package_exports_point_to_canonical_policy_functions():
    assert parallel.require_int is policy.require_int
    assert parallel.require_positive_int is policy.require_positive_int
    assert parallel.require_non_negative_int is policy.require_non_negative_int
    assert parallel.resolve_parallel_strategy is policy.resolve_parallel_strategy
    assert parallel.normalize_order_mode is policy.normalize_order_mode
    assert parallel.apply_parallel_profile is policy.apply_parallel_profile
    assert parallel.resolve_prefetch is policy.resolve_prefetch
    assert parallel.resolve_inflight_limit is policy.resolve_inflight_limit
