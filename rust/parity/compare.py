#!/usr/bin/env python3
"""Compare two parity outputs (Python reference vs Rust port).

Two modes:

* default (core parity): every difference is a failure. Numbers are compared with
  an absolute tolerance; strings, booleans and null must match exactly.
* --shape-only (service parity): the engines run *different models*, so values are
  expected to differ. Only the STRUCTURE is checked (same keys, same types, same
  nesting); value differences are reported as informational.

Usage:
    python3 compare.py python_out.json rust_out.json [--shape-only] [--tol 1e-9]
                       [--output parity_report.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

MISSING = "<missing>"


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def walk(python, rust, path, tol, shape_only, differences, counters):
    if isinstance(python, dict) and isinstance(rust, dict):
        for key in sorted(set(python) | set(rust)):
            child = f"{path}.{key}" if path else key
            if key not in python:
                differences.append({"path": child, "kind": "extra_in_rust", "python": MISSING, "rust": rust[key]})
            elif key not in rust:
                differences.append({"path": child, "kind": "missing_in_rust", "python": python[key], "rust": MISSING})
            else:
                walk(python[key], rust[key], child, tol, shape_only, differences, counters)
        return
    if isinstance(python, list) and isinstance(rust, list):
        if len(python) != len(rust):
            differences.append({"path": path, "kind": "length", "python": len(python), "rust": len(rust)})
            return
        for index, (a, b) in enumerate(zip(python, rust)):
            walk(a, b, f"{path}[{index}]", tol, shape_only, differences, counters)
        return
    if is_number(python) and is_number(rust):
        delta = abs(float(python) - float(rust))
        counters["numeric"] += 1
        counters["max_delta"] = max(counters["max_delta"], delta)
        # NaN is never equal; infinities must match exactly
        if math.isnan(delta) or (math.isinf(float(python)) != math.isinf(float(rust))):
            differences.append({"path": path, "kind": "numeric", "python": python, "rust": rust, "delta": delta})
        elif delta > tol:
            differences.append({"path": path, "kind": "numeric", "python": python, "rust": rust, "delta": delta})
        return
    counters["scalar"] += 1
    if python != rust or type(python) is not type(rust):
        # A null on one side and a value on the other is a VALUE difference, not a
        # structural one: both are valid scalars (a model may abstain, another may not).
        if python is None or rust is None:
            kind = "value"
        elif type(python) is not type(rust):
            kind = "type"
        else:
            kind = "value"
        differences.append({"path": path, "kind": kind, "python": python, "rust": rust})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_out")
    parser.add_argument("rust_out")
    parser.add_argument("--shape-only", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--output")
    parser.add_argument(
        "--documented-prefix",
        action="append",
        default=[],
        help="path prefix whose differences are declared in the SPEC (reported, not blocking)",
    )
    args = parser.parse_args()

    python_out = json.loads(Path(args.python_out).read_text(encoding="utf-8"))
    rust_out = json.loads(Path(args.rust_out).read_text(encoding="utf-8"))

    def is_documented(path: str) -> bool:
        return any(path == prefix or path.startswith(prefix + ".") for prefix in args.documented_prefix)

    differences, failures, matched = [], [], 0
    counters = {"numeric": 0, "scalar": 0, "max_delta": 0.0}
    for case_id in sorted(set(python_out) | set(rust_out)):
        if case_id not in python_out or case_id not in rust_out:
            failures.append(case_id)
            differences.append({
                "case": case_id,
                "path": case_id,
                "kind": "missing_case",
                "python": case_id in python_out,
                "rust": case_id in rust_out,
            })
            continue
        case_differences = []
        # Paths are absolute (prefixed with the top-level key) so that --documented-prefix
        # can address whole subtrees such as `model` or `timing.peak_mlx_bytes`.
        walk(python_out[case_id], rust_out[case_id], case_id, args.tol, args.shape_only, case_differences, counters)
        for difference in case_differences:
            difference["case"] = case_id
        differences.extend(case_differences)
        if case_differences:
            failures.append(case_id)
        else:
            matched += 1

    documented = [d for d in differences if is_documented(d["path"])]
    remaining = [d for d in differences if d not in documented]
    informational = [d for d in remaining if args.shape_only and d["kind"] in ("value", "numeric")]
    blocking = [d for d in remaining if d not in informational] if args.shape_only else remaining

    report = {
        "mode": "shape-only" if args.shape_only else "exact",
        "numeric_tolerance": args.tol,
        "cases": len(set(python_out) | set(rust_out)),
        "cases_matched": matched,
        "cases_with_differences": sorted(set(failures)),
        "numeric_fields_compared": counters["numeric"],
        "scalar_fields_compared": counters["scalar"],
        "max_numeric_delta": counters["max_delta"],
        "blocking_differences": len(blocking),
        "documented_differences": len(documented),
        "informational_differences": len(informational),
        "differences": differences,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"mode: {report['mode']}  (numeric tolerance {args.tol})")
    print(f"cases: {report['cases']}  matched: {matched}  with differences: {len(set(failures))}")
    print(f"fields compared: {counters['numeric']} numeric, {counters['scalar']} scalar")
    print(f"max numeric delta: {counters['max_delta']:.3e}")
    print(f"blocking differences: {len(blocking)}   documented: {len(documented)}   informational: {len(informational)}")
    for difference in blocking[:25]:
        print(f"  - [{difference.get('case','?')}] {difference['path']}: {difference['kind']} "
              f"python={difference.get('python')!r} rust={difference.get('rust')!r}")
    if len(blocking) > 25:
        print(f"  ... and {len(blocking) - 25} more")
    if documented:
        print("documented differences (declared in the SPEC, backend-specific):")
        for difference in documented[:40]:
            print(f"  ~ [{difference.get('case','?')}] {difference['path']}: {difference['kind']}")
        if len(documented) > 40:
            print(f"  ... and {len(documented) - 40} more")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
