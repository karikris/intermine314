#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_collected_tests(path: Path) -> tuple[int | None, str | None]:
    if not path.exists():
        return None, f"collect output not found: {path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(\d+)\s+tests?\s+collected", text)
    if match is None:
        return None, "unable to parse collected test count from collect output"
    return int(match.group(1)), None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _summarize_durations(durations: list[float]) -> dict[str, Any]:
    if not durations:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "total": 0.0,
            "histogram": [],
        }
    sorted_durations = sorted(float(value) for value in durations if value >= 0.0)
    if not sorted_durations:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "total": 0.0,
            "histogram": [],
        }

    edges = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    bins: list[dict[str, Any]] = []
    lower = 0.0
    for upper in edges:
        count = sum(1 for value in sorted_durations if lower <= value < upper)
        bins.append({"range": f"[{lower:.3f},{upper:.3f})", "count": count})
        lower = upper
    bins.append(
        {
            "range": f"[{edges[-1]:.3f},inf)",
            "count": sum(1 for value in sorted_durations if value >= edges[-1]),
        }
    )

    return {
        "count": len(sorted_durations),
        "min": sorted_durations[0],
        "max": sorted_durations[-1],
        "mean": statistics.fmean(sorted_durations),
        "median": _percentile(sorted_durations, 0.50),
        "p90": _percentile(sorted_durations, 0.90),
        "p95": _percentile(sorted_durations, 0.95),
        "p99": _percentile(sorted_durations, 0.99),
        "total": float(sum(sorted_durations)),
        "histogram": bins,
    }


def _parse_junit(path: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if not path.exists():
        return (
            {
                "executed_testcases": 0,
                "skipped_tests": 0,
                "skip_reasons": [],
                "runtime_distribution_seconds": _summarize_durations([]),
            },
            [f"junit xml not found: {path}"],
        )

    try:
        root = ET.parse(path).getroot()
    except Exception as exc:  # pragma: no cover - defensive
        return (
            {
                "executed_testcases": 0,
                "skipped_tests": 0,
                "skip_reasons": [],
                "runtime_distribution_seconds": _summarize_durations([]),
            },
            [f"failed to parse junit xml: {exc}"],
        )

    skip_reasons: Counter[str] = Counter()
    durations: list[float] = []
    testcases = list(root.findall(".//testcase"))
    for case in testcases:
        duration_raw = str(case.attrib.get("time", "0")).strip()
        try:
            duration = float(duration_raw)
        except Exception:
            duration = 0.0
            warnings.append(f"invalid testcase time '{duration_raw}'")
        durations.append(duration)

        skipped = case.find("skipped")
        if skipped is None:
            continue
        reason = (skipped.attrib.get("message") or (skipped.text or "")).strip()
        if not reason:
            reason = "unspecified skip reason"
        skip_reasons[reason] += 1

    skip_list = [{"reason": reason, "count": count} for reason, count in skip_reasons.most_common()]
    return (
        {
            "executed_testcases": len(testcases),
            "skipped_tests": int(sum(skip_reasons.values())),
            "skip_reasons": skip_list,
            "runtime_distribution_seconds": _summarize_durations(durations),
        },
        warnings,
    )


def _render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Test Inventory Baseline")
    lines.append("")
    lines.append(f"- generated_at_utc: `{summary['generated_at_utc']}`")
    lines.append(f"- collected_tests: `{summary.get('collected_tests')}`")
    lines.append(f"- executed_testcases: `{summary.get('executed_testcases')}`")
    lines.append(f"- skipped_tests: `{summary.get('skipped_tests')}`")
    lines.append("")

    lines.append("## Skip Reasons")
    skip_reasons = summary.get("skip_reasons", [])
    if skip_reasons:
        lines.append("| Count | Reason |")
        lines.append("| ---: | --- |")
        for item in skip_reasons:
            lines.append(f"| {item['count']} | {item['reason']} |")
    else:
        lines.append("No skipped tests recorded.")
    lines.append("")

    dist = summary.get("runtime_distribution_seconds", {})
    lines.append("## Runtime Distribution (seconds)")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key in ("min", "max", "mean", "median", "p90", "p95", "p99", "total"):
        value = float(dist.get(key, 0.0))
        lines.append(f"| {key} | {value:.6f} |")
    lines.append("")
    lines.append("### Histogram")
    histogram = dist.get("histogram", [])
    if histogram:
        lines.append("| Range | Count |")
        lines.append("| --- | ---: |")
        for bucket in histogram:
            lines.append(f"| {bucket['range']} | {bucket['count']} |")
    else:
        lines.append("No runtime histogram available.")
    lines.append("")

    warnings = summary.get("warnings", [])
    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build baseline test inventory summary from pytest artifacts.")
    parser.add_argument("--collect-out", required=True, help="Path to pytest --collect-only output text.")
    parser.add_argument("--junit-xml", required=True, help="Path to pytest junit XML report.")
    parser.add_argument("--json-out", required=True, help="Path for summary JSON output.")
    parser.add_argument("--md-out", required=False, help="Optional path for markdown summary output.")
    args = parser.parse_args(argv)

    collect_path = Path(args.collect_out)
    junit_path = Path(args.junit_xml)
    json_path = Path(args.json_out)
    md_path = Path(args.md_out) if args.md_out else None

    collected_tests, collect_warning = _parse_collected_tests(collect_path)
    junit_summary, junit_warnings = _parse_junit(junit_path)
    warnings: list[str] = []
    if collect_warning:
        warnings.append(collect_warning)
    warnings.extend(junit_warnings)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "collected_tests": collected_tests,
        **junit_summary,
        "warnings": warnings,
        "inputs": {
            "collect_out": str(collect_path),
            "junit_xml": str(junit_path),
        },
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
