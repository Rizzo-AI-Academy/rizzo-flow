"""Speed and agreement of one backend over a grid of state lengths × questions per request.

Run it once per backend on the same machine, same checkpoint, one process with weights at a
time, then compare the two reports:

  .venv/bin/python scripts/backend_bench.py run --backend llama --quant bf16 --output A.json
  .venv/bin/python scripts/backend_bench.py run --backend max --output B.json
  .venv/bin/python scripts/backend_bench.py compare A.json B.json

The requests are synthetic and deterministic (a seeded event log as state; boolean, choice and
score questions about it), so both backends see the same prompts: the report records each
answer's `prompt_sha256` and `compare` refuses to pair different prompts. Timings are wall
clock around `Engine.decide` with the model warm; the first call per cell is a warm-up and is
not counted. Reports are create-only, like every benchmark artifact in this project.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

NAMES = ["Ada", "Bruno", "Chiara", "Dario", "Elena", "Fabio", "Giulia", "Hugo", "Irene", "Luca"]
ACTIONS = [
    "logged in from a new device",
    "reset the password",
    "opened a support ticket about a failed payment",
    "upgraded the subscription to the team plan",
    "downloaded the monthly invoice",
    "reported that the export to CSV times out",
    "invited two colleagues to the workspace",
    "cancelled a scheduled meeting",
    "asked for a refund of the last charge",
    "changed the billing address",
]
QUESTIONS = [
    {
        "type": "boolean",
        "instructions": "Did anyone in the log ask for a refund?",
    },
    {
        "type": "choice",
        "instructions": "Which team should look at the most recent event?",
        "options": [
            {"id": "support", "description": "Technical support: logins, errors, timeouts."},
            {"id": "billing", "description": "Billing: invoices, charges, refunds."},
            {"id": "sales", "description": "Sales: upgrades and new seats."},
        ],
    },
    {
        "type": "score",
        "instructions": "How busy was the account over the whole log?",
        "levels": ["A handful of events.", "Regular activity.", "Constant activity."],
    },
]


def state(tokens: int, seed: int = 0) -> dict:
    """An event log of roughly `tokens` tokens (about 16 per line)."""
    rng = random.Random(seed)
    lines = [
        f"{day // 24 + 1:03d}/{day % 24:02d}h {rng.choice(NAMES)} {rng.choice(ACTIONS)}"
        for day in range(max(1, tokens // 16))
    ]
    return {"account": "ACME-42", "events": "\n".join(lines)}


def request(tokens: int, questions: int) -> dict:
    return {
        "state": state(tokens),
        "questions": {f"q{i:02d}": QUESTIONS[i % len(QUESTIONS)] for i in range(questions)},
    }


def quantiles(values):
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]
    return {"p50": pick(0.5), "p95": pick(0.95), "min": ordered[0], "mean": statistics.mean(values)}


def measure(engine, payload, tokens, count, mode, repeats):
    first = engine.decide(payload)  # warm-up, also the recorded answers
    seconds, inference, prefill = [], [], []
    for _ in range(repeats):
        mark = time.perf_counter()
        response = engine.decide(payload)
        seconds.append(time.perf_counter() - mark)
        inference.append(response["timing"]["inference_seconds"])
        prefill.append(response["timing"]["prefill_seconds"])
    timing = response["timing"]
    return {
        "state_tokens": tokens,
        "questions": count,
        "mode": mode,
        "input_tokens": timing["logical_input_tokens"],
        "shared_prefix_tokens": timing["shared_prefix_tokens"],
        "seconds": quantiles(seconds),
        "inference_seconds": quantiles(inference),
        "prefill_seconds": quantiles(prefill),
        "decisions_per_second": count / statistics.median(seconds),
        "peak_device_bytes": timing.get("peak_device_bytes"),
        "answers": {
            key: {
                "prompt_sha256": answer["prompt_sha256"],
                "probabilities": answer["probabilities"],
            }
            for key, answer in first["answers"].items()
        },
    }


def run(args):
    from rizzo_flow.engine import Engine
    from rizzo_flow.loader import load_backend

    backend = load_backend(
        args.backend,
        size=args.size,
        quant=args.quant,
        weights=args.weights,
        device=args.device,
        ctx=args.ctx,
        batch_size=args.batch_size,
        prefill_chunk=args.prefill_chunk,
    )
    engine = Engine(backend, ctx=args.ctx)
    report = {
        "backend": args.backend,
        "model": {
            **backend.metadata,
            "batch_size": args.batch_size,
            "prefill_chunk": backend.prefill_chunk,
        },
        "repeats": args.repeats,
        "cells": [],
    }
    try:
        for tokens in args.states:
            for count in args.questions:
                for mode in args.modes:
                    payload = {**request(tokens, count), "mode": mode}
                    try:
                        cell = measure(engine, payload, tokens, count, mode, args.repeats)
                    except (ValueError, RuntimeError) as error:
                        # Out of device memory, typically: record it and keep the other cells.
                        cell = {"state_tokens": tokens, "questions": count, "mode": mode}
                        cell["error"] = str(error).splitlines()[0]
                        report["cells"].append(cell)
                        print(f"{tokens:>6} tok × {count:>3} q {mode:>6}: {cell['error']}")
                        continue
                    report["cells"].append(cell)
                    print(
                        f"{tokens:>6} tok × {count:>3} q {mode:>6}: "
                        f"p50 {cell['seconds']['p50'] * 1000:8.1f} ms, "
                        f"{cell['decisions_per_second']:7.2f} dec/s",
                        flush=True,
                    )
    finally:
        release = getattr(backend, "close", None)
        if release is not None:
            release()
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def compare(args):
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.a, args.b))
    key = lambda c: (c["state_tokens"], c["questions"], c["mode"])
    right = {key(c): c for c in b["cells"]}
    name = lambda r: f"{r['backend']} {r['model'].get('precision')} {r['model'].get('backend')}"
    print(f"A = {name(a)}, B = {name(b)}\n")
    print(
        "| tokens per question | questions | mode | A p50 ms | B p50 ms | B/A | A dec/s | B dec/s "
        "| argmax changes | max Δp |"
    )
    print("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cell in a["cells"]:
        other = right.get(key(cell))
        if other is None:
            continue
        if "error" in cell or "error" in other:
            failed = "A" if "error" in cell else "B"
            print(
                f"| {cell['state_tokens']} state | {cell['questions']} | {cell['mode']} "
                f"| {failed} failed: {(cell.get('error') or other['error'])[:60]} |"
            )
            continue
        changes, delta = 0, 0.0
        for q, answer in cell["answers"].items():
            twin = other["answers"][q]
            if answer["prompt_sha256"] != twin["prompt_sha256"]:
                raise ValueError(f"Different prompts for {key(cell)} {q}: not comparable")
            p, r = answer["probabilities"], twin["probabilities"]
            changes += max(p, key=p.get) != max(r, key=r.get)
            delta = max(delta, *(abs(p[o] - r[o]) for o in p))
        ta, tb = cell["seconds"]["p50"] * 1000, other["seconds"]["p50"] * 1000
        print(
            f"| {cell['input_tokens'] // max(1, cell['questions'])} | {cell['questions']} "
            f"| {cell['mode']} | {ta:.1f} | {tb:.1f} | {tb / ta:.2f} "
            f"| {cell['decisions_per_second']:.2f} | {other['decisions_per_second']:.2f} "
            f"| {changes}/{len(cell['answers'])} | {delta:.4f} |"
        )
    for label, report in (("A", a), ("B", b)):
        peaks = [c["peak_device_bytes"] for c in report["cells"] if c.get("peak_device_bytes")]
        if peaks:
            print(f"\n{label} peak device memory: {max(peaks) / 2**30:.2f} GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("run")
    bench.add_argument("--backend", choices=("llama", "mlx", "max"), required=True)
    bench.add_argument("--size", default="4b")
    bench.add_argument("--quant", help="llama.cpp GGUF: q8_0, q4_k_m, bf16")
    bench.add_argument("--weights", choices=("flow", "base"))
    bench.add_argument("--device", default="auto")
    bench.add_argument("--batch-size", type=int, default=4)
    bench.add_argument("--prefill-chunk", type=int, help="Tokens per prefill call (llama, max)")
    bench.add_argument("--ctx", type=int, default=12288, help="Above the longest state")
    bench.add_argument("--states", type=int, nargs="+", default=[512, 2048, 8192])
    bench.add_argument("--questions", type=int, nargs="+", default=[1, 8, 64])
    bench.add_argument("--modes", nargs="+", choices=("shared", "direct"), default=["shared"])
    bench.add_argument("--repeats", type=int, default=5, choices=range(1, 101), metavar="N")
    bench.add_argument("--output", required=True)
    pair = commands.add_parser("compare")
    pair.add_argument("a")
    pair.add_argument("b")
    args = parser.parse_args()
    (run if args.command == "run" else compare)(args)


if __name__ == "__main__":
    main()
