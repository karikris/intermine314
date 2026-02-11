VALID_ORDER_MODES = frozenset({"ordered", "unordered", "window", "mostly_ordered"})


def normalize_mode(mode):
    if isinstance(mode, bool):
        return "ordered" if mode else "unordered"
    text = str(mode).strip().lower()
    if text not in VALID_ORDER_MODES:
        raise ValueError("mode must be ordered|unordered|window|mostly_ordered")
    return text
