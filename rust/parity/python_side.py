#!/usr/bin/env python3
"""Reference side of the decision-core parity check.

Runs the ORIGINAL Rizzo Flow decoding (rizzo_flow.decisions.decode) over the golden
vectors, so the Rust port can be compared against it field by field.

Run it with the reference checkout on PYTHONPATH, e.g.

    cd /path/to/rizzo-flow && PYTHONPATH=src .venv/bin/python \
        /path/to/rizzo-flow-rs/parity/python_side.py core_vectors.json > python_out.json
"""

import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from rizzo_flow.decisions import decode
from rizzo_flow.schema import Question

adapter = TypeAdapter(Question)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python_side.py <core_vectors.json>", file=sys.stderr)
        return 2
    vectors = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = {}
    for case in vectors["cases"]:
        question = adapter.validate_python(case["question"])
        try:
            out[case["id"]] = {"ok": decode(question, case["logits"], case.get("temperature", 1.0))}
        except ValueError as error:
            out[case["id"]] = {"error": str(error)}
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
