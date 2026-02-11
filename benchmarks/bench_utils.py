from __future__ import annotations

import math
import statistics
from pathlib import Path


def parse_csv_tokens(text: str | None) -> list[str]:
    if text is None:
        return []
    return [token.strip() for token in str(text).split(",") if token.strip()]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_string_list(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return parse_csv_tokens(values)
    if isinstance(values, (list, tuple, set)):
        out: list[str] = []
        for value in values:
            token = str(value).strip()
            if token:
                out.append(token)
        return out
    return []


def merge_shallow_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def stat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        stddev = 0.0
    else:
        stddev = statistics.stdev(values)
    sorted_vals = sorted(values)

    def nearest_rank(p: float) -> float:
        idx = int(math.ceil(p * len(sorted_vals))) - 1
        idx = max(0, min(idx, len(sorted_vals) - 1))
        return sorted_vals[idx]

    trim_count = int(math.floor(len(sorted_vals) * 0.10))
    if trim_count > 0:
        trimmed = sorted_vals[: len(sorted_vals) - trim_count]
    else:
        trimmed = sorted_vals

    group_count = min(3, len(sorted_vals))
    chunk_size = int(math.ceil(len(sorted_vals) / group_count))
    means = []
    for idx in range(0, len(sorted_vals), chunk_size):
        chunk = sorted_vals[idx : idx + chunk_size]
        if chunk:
            means.append(statistics.fmean(chunk))

    mean_val = statistics.fmean(values)
    return {
        "n": float(len(values)),
        "mean": mean_val,
        "stddev": stddev,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "p90": nearest_rank(0.90),
        "p95": nearest_rank(0.95),
        "trimmed_mean_drop_high_10pct": statistics.fmean(trimmed),
        "median_of_means": statistics.median(means),
        "cv_pct": (stddev / mean_val * 100.0) if mean_val else 0.0,
    }
