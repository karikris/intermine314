from __future__ import annotations

VALID_PARALLEL_PAGINATION = frozenset({"auto", "offset", "keyset"})
VALID_PARALLEL_PROFILES = frozenset({"default", "large_query", "unordered", "mostly_ordered"})
VALID_ORDER_MODES = frozenset({"ordered", "unordered", "window", "mostly_ordered"})


def require_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def require_positive_int(name, value):
    ivalue = require_int(name, value)
    if ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return ivalue


def require_non_negative_int(name, value):
    ivalue = require_int(name, value)
    if ivalue < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return ivalue


def resolve_parallel_strategy(pagination, start, size, *, valid_parallel_pagination, keyset_auto_min_size):
    strategy = str(pagination).lower()
    if strategy not in valid_parallel_pagination:
        choices = ", ".join(sorted(valid_parallel_pagination))
        raise ValueError("pagination must be one of: %s" % (choices,))

    if strategy == "offset":
        return "offset"
    if strategy == "keyset":
        return "keyset" if start == 0 else "offset"

    if start != 0:
        return "offset"
    if size is None:
        return "keyset"
    if size >= keyset_auto_min_size:
        return "keyset"
    return "offset"


def normalize_order_mode(ordered, *, default_order_mode, valid_order_modes):
    if ordered is None:
        return default_order_mode
    if isinstance(ordered, bool):
        return "ordered" if ordered else "unordered"
    mode = str(ordered).strip().lower()
    if mode not in valid_order_modes:
        choices = ", ".join(sorted(valid_order_modes))
        raise ValueError("ordered must be a bool or one of: %s" % (choices,))
    return mode


def apply_parallel_profile(profile, ordered, large_query_mode, *, default_profile, valid_parallel_profiles):
    if profile is None:
        profile = default_profile
    profile_name = str(profile).strip().lower()
    if profile_name not in valid_parallel_profiles:
        choices = ", ".join(sorted(valid_parallel_profiles))
        raise ValueError("profile must be one of: %s" % (choices,))

    effective_ordered = ordered
    effective_large = bool(large_query_mode)
    if profile_name == "large_query":
        effective_large = True
        if effective_ordered is None:
            effective_ordered = "window"
    elif profile_name == "unordered":
        if effective_ordered is None:
            effective_ordered = False
    elif profile_name == "mostly_ordered":
        if effective_ordered is None:
            effective_ordered = "window"
    else:
        if effective_ordered is None:
            effective_ordered = True

    return profile_name, effective_ordered, effective_large


def resolve_prefetch(prefetch, *, max_workers, large_query_mode, default_parallel_prefetch):
    if prefetch is None:
        if default_parallel_prefetch is not None:
            prefetch = default_parallel_prefetch
        else:
            prefetch = max_workers * 2 if large_query_mode else max_workers
    return require_positive_int("prefetch", prefetch)


def resolve_inflight_limit(inflight_limit, *, prefetch, default_parallel_inflight_limit):
    if inflight_limit is None:
        inflight_limit = (
            default_parallel_inflight_limit if default_parallel_inflight_limit is not None else prefetch
        )
    return require_positive_int("inflight_limit", inflight_limit)


__all__ = [
    "VALID_PARALLEL_PAGINATION",
    "VALID_PARALLEL_PROFILES",
    "VALID_ORDER_MODES",
    "require_int",
    "require_positive_int",
    "require_non_negative_int",
    "resolve_parallel_strategy",
    "normalize_order_mode",
    "apply_parallel_profile",
    "resolve_prefetch",
    "resolve_inflight_limit",
]
