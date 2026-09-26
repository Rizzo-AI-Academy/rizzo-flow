#!/usr/bin/env python3
"""Reference side of the end-to-end parity check.

Runs the ORIGINAL Rizzo Flow engine (Spark-X2.5 via MLX) over one request file and
prints the full response, so the Rust port's response can be compared with it.

Configuration: the reference runs with `bits=8` (the documented Q8 configuration;
bf16 is ~10 GB and does not fit the node's 8 GB). The two engines run DIFFERENT
models (Spark/MLX vs a GGUF on llama.cpp), so values are expected to differ: this
check validates the response STRUCTURE, not the answers.

    cd /path/to/rizzo-flow && PYTHONPATH=src .venv/bin/python \
        /path/to/rizzo-flow-rs/parity/python_engine.py request.json > python_decide.json
"""

import json
import sys
from pathlib import Path

from rizzo_flow.backend import SparkBackend
from rizzo_flow.config import DEFAULT_SIZE, MODELS
from rizzo_flow.engine import Engine
from rizzo_flow.schema import Request

BITS = 8
CTX = 4096


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python_engine.py <request.json>", file=sys.stderr)
        return 2
    request = Request.model_validate_json(Path(sys.argv[1]).read_text(encoding="utf-8"))
    backend = SparkBackend.load(MODELS[DEFAULT_SIZE].path, bits=BITS, device="auto", batch_size=4)
    engine = Engine(backend, ctx=CTX)
    response = engine.decide(request)
    print(json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
