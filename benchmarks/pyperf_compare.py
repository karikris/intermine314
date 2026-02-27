#!/usr/bin/env python3
"""Utility wrapper around pyperf command/compare_to for benchmark repeatability."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def _run_command_mode(args: argparse.Namespace) -> int:
    if not args.command:
        raise ValueError("command mode requires a benchmark command after '--'")
    cmd = [sys.executable, "-m", "pyperf", "command", "--output", args.output]
    if args.rigorous:
        cmd.append("--rigorous")
    if args.loops is not None:
        cmd.extend(["--loops", str(args.loops)])
    cmd.extend(args.command)
    print("pyperf_run " + shlex.join(cmd), flush=True)
    return subprocess.call(cmd)


def _run_compare_mode(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "pyperf", "compare_to", args.baseline, args.candidate]
    if args.table:
        cmd.append("--table")
    print("pyperf_compare " + shlex.join(cmd), flush=True)
    return subprocess.call(cmd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    run_parser = subparsers.add_parser("command", help="Run a command under pyperf and write JSON output.")
    run_parser.add_argument("--output", required=True, help="Target pyperf JSON output path.")
    run_parser.add_argument("--loops", type=int, default=None, help="Optional pyperf --loops value.")
    run_parser.add_argument(
        "--rigorous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pyperf --rigorous mode.",
    )
    run_parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to benchmark after '--'.")

    compare_parser = subparsers.add_parser("compare", help="Compare two pyperf JSON runs.")
    compare_parser.add_argument("--baseline", required=True, help="Baseline pyperf JSON file.")
    compare_parser.add_argument("--candidate", required=True, help="Candidate pyperf JSON file.")
    compare_parser.add_argument("--table", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "command":
        return _run_command_mode(args)
    if args.mode == "compare":
        return _run_compare_mode(args)
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
