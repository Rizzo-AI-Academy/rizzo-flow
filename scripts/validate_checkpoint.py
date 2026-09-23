"""Run real-weight checks and preserve raw evidence. Execute from the repository root."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from rizzo_flow.api import create_app
from rizzo_flow.cli import read_jsonl, write_json
from rizzo_flow.engine import Engine
from rizzo_flow.evaluation import evaluate
from rizzo_flow.loader import BACKENDS, load_backend
from rizzo_flow.prompts import Compiled, compile_request
from rizzo_flow.schema import Request

if TYPE_CHECKING:  # MLX is an optional install
    from rizzo_flow.backend import SparkBackend


def projection_delta(backend: SparkBackend, job: Compiled) -> float:
    """MLX only: the selected-row projection against the full vocabulary head."""
    import mlx.core as mx

    from rizzo_flow.backend import selected_logits

    cache = backend._prefill(job.tokens[:-1])
    hidden = backend.model.model(mx.array([job.tokens[-1:]]), cache=cache)[:, -1, :]
    optimized = selected_logits(backend.model, hidden, job.slots)
    head = backend.model.model.embedding
    native = head.as_linear(hidden)[:, mx.array(job.slots)].astype(mx.float32)
    mx.eval(optimized, native)
    delta = mx.max(mx.abs(optimized - native)).item()
    if delta > 0.125:
        raise RuntimeError(f"Selected projection diverged from full head: {delta}")
    return delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=BACKENDS, default="llama")
    parser.add_argument("--size", default="4b")
    parser.add_argument("--model", help="GGUF file (llama) or checkpoint directory (mlx)")
    parser.add_argument("--quant", help="llama: q8_0 (default), q4_k_m, bf16")
    parser.add_argument("--bits", type=int, choices=[4, 8], help="mlx only")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-long-state", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    backend = load_backend(
        args.backend,
        size=args.size,
        model=args.model,
        quant=args.quant,
        bits=args.bits,
        device=args.device,
    )
    engine = Engine(backend)
    request = Request.model_validate_json(Path("examples/ticket.json").read_text(encoding="utf-8"))
    _, jobs = compile_request(backend.tokenizer, request, 8192)
    # llama.cpp always projects the whole vocabulary: there is nothing to compare there.
    delta = projection_delta(backend, jobs[0]) if args.backend == "mlx" else None
    print(f"Projection max logit delta: {delta}", flush=True)
    with TestClient(create_app(engine)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/v1/decisions", json=request.model_dump())
        assert response.status_code == 200, response.text
        write_json(response.json(), out / "api-example.json")
    # State long enough to cross Spark's 512-token rotating attention window.
    long_request = request.model_dump()
    long_request["state"]["background"] = "Archived note: the office has blue walls. " * 150
    started = time.perf_counter()
    reports = {}
    suites = {
        "smoke": read_jsonl("benchmarks/smoke.jsonl"),
        "perturbations": read_jsonl("benchmarks/perturbations.jsonl"),
        "long-state": [{"id": "long-shared-state", "request": long_request}],
    }
    if args.skip_long_state:
        del suites["long-state"]
    for name, fixtures in suites.items():
        report = evaluate(
            engine, fixtures, repeats=2 if name == "long-state" else 1, compare_modes=True
        )
        write_json(report.model_dump(), out / f"{name}.json")
        reports[name] = report.summary.model_dump()
        print(name, json.dumps(reports[name], ensure_ascii=False), flush=True)
    write_json(
        {
            "model": backend.metadata.as_dict(),
            "selected_projection_max_logit_delta": delta,
            "validation_seconds": time.perf_counter() - started,
            "suites": reports,
        },
        out / "summary.json",
    )


if __name__ == "__main__":
    main()
