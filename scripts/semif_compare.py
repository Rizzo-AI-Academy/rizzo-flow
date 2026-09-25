"""Run SemIf's own fixtures through Rizzo Flow (llama.cpp or MLX) or SemIf's MLX backend.

Both systems see the same rows, are scored by SemIf's `benchmarks/evaluate.py`, and are timed
with the scope SemIf documents: warm model; prompt construction, tokenization, forward passes
and CPU readout included; model loading and file writes excluded. Each system keeps its own
prompt and model, so this compares complete systems, not checkpoints in isolation.

  .venv/bin/python scripts/semif_compare.py --system rizzo --semif /path/SemIf --output results/x
  .venv/bin/python scripts/semif_compare.py --system rizzo --backend mlx --bits 8 ...
  /path/semif-venv/bin/python scripts/semif_compare.py --system semif --semif /path/SemIf --output ...
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


def read(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write(path, value):
    # Create-only, like every other benchmark artifact in this project.
    with Path(path).open("x", encoding="utf-8") as stream:
        if isinstance(value, list):
            stream.writelines(json.dumps(row, allow_nan=False) + "\n" for row in value)
        else:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


class MlxMemory:
    key = "peak_mlx_bytes"

    def reset(self):
        import mlx.core as mx

        mx.clear_cache()
        mx.reset_peak_memory()

    def peak(self):
        import mlx.core as mx

        return mx.get_peak_memory()


class LlamaMemory:
    """Largest drop in free device memory since before the load; never resets."""

    key = "peak_device_bytes"

    def __init__(self, backend):
        self.backend = backend

    def reset(self):
        pass

    def peak(self):
        return self.backend.peak_device_bytes()


class Rizzo:
    def __init__(self, args):
        from rizzo_flow.engine import Engine
        from rizzo_flow.loader import load_backend

        backend = load_backend(
            args.backend,
            size=args.size,
            model=args.model,
            quant=args.quant,
            weights=args.weights,
            bits=args.bits,
            device=args.device,
            batch_size=args.batch_size,
            **({"kv_type": args.kv_type} if args.kv_type else {}),
        )
        self.engine = Engine(backend)
        self.metadata = {**backend.metadata, "batch_size": args.batch_size}
        self.memory = MlxMemory() if args.backend == "mlx" else LlamaMemory(backend)

    def _run(self, rows, mode):
        questions = {
            row["id"]: {
                "type": "choice",
                "instructions": row["question"],
                # Fixture options already include their own `insufficient` where relevant.
                "policy": {"allow_abstain": False},
                "options": [
                    {"id": f"o{index}", "description": option["description"]}
                    for index, option in enumerate(row["options"])
                ],
            }
            for row in rows
        }
        response = self.engine.decide(
            {"state": rows[0]["state"], "questions": questions, "mode": mode}
        )
        return [
            {
                "id": row["id"],
                "option_ids": [option["id"] for option in row["options"]],
                "probabilities": list(response["answers"][row["id"]]["probabilities"].values()),
                "input_tokens": response["answers"][row["id"]]["input_tokens"],
            }
            for row in rows
        ]

    def direct(self, row):
        return self._run([row], "direct")[0]

    def shared(self, rows):
        return self._run(rows, "shared")


class SemIf:
    def __init__(self, args):
        from semif_phase1 import mlx_backend

        self.backend = mlx_backend
        self.memory = MlxMemory()
        self.model, self.tokenizer, self.metadata = mlx_backend.load_model(
            args.model or "Qwen/Qwen3.5-4B", args.revision, args.bits
        )

    def direct(self, row):
        return self.backend.score(self.model, self.tokenizer, row, self.metadata)

    def shared(self, rows):
        return self.backend.score_shared(self.model, self.tokenizer, rows, self.metadata)[0]


def latency(seconds):
    ordered = sorted(seconds)
    return {
        "calls": len(ordered),
        "total_seconds": sum(ordered),
        "p50_seconds": statistics.median(ordered),
        "p95_seconds": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "max_seconds": ordered[-1],
    }


def stability(evaluate, gold, base, perturb_gold, perturbed):
    """Single-system version of SemIf's `evaluate_perturbations.py` (same definitions)."""

    def chosen(row):
        return row["option_ids"][row["probabilities"].index(max(row["probabilities"]))]

    def headline(result):
        return {
            "mean_family_balanced_accuracy": result["mean_family_balanced_accuracy"],
            "accuracy": 1 - len(result["errors"]) / result["evaluated"],
        }

    base = {row["id"]: row for row in base}
    perturbed = {row["id"]: row for row in perturbed}
    originals = [row for row in gold if row["provenance"]["variant"] == "original"]
    missing = [row for row in gold if row["provenance"]["variant"] == "missing"]
    report = {
        "base_original": headline(evaluate.evaluate(originals, [base[r["id"]] for r in originals])),
        "variants": {},
    }
    variants = defaultdict(list)
    for row in perturb_gold:
        variants[row["provenance"]["variant"]].append(row)
    for variant, rows in variants.items():
        shifts, flips = [], []
        for row in rows:
            reference, candidate = base[row["provenance"]["base_id"]], perturbed[row["id"]]
            left = dict(zip(reference["option_ids"], reference["probabilities"]))
            right = dict(zip(candidate["option_ids"], candidate["probabilities"]))
            if set(left) != set(right):
                raise ValueError("Semantic option IDs changed")
            shifts.append(max(abs(left[key] - right[key]) for key in left))
            if chosen(reference) != chosen(candidate):
                flips.append(row["id"])
        report["variants"][variant] = {
            **headline(evaluate.evaluate(rows, [perturbed[r["id"]] for r in rows])),
            "argmax_flips": len(flips),
            "flip_rows": flips,
            "mean_max_probability_movement": statistics.mean(shifts),
            "max_probability_movement": max(shifts),
        }
    predictions = [base[row["id"]] for row in missing]
    report["missing_evidence"] = {
        **headline(evaluate.evaluate(missing, predictions)),
        "confident_non_insufficient_at_0_8": sum(
            row["predicted_id"] != "insufficient" and row["confidence"] >= 0.8
            for row in evaluate.align(missing, predictions)
        ),
    }
    return report


def slim(prediction):
    keep = ("id", "option_ids", "probabilities", "option_logits", "input_tokens")
    return {key: prediction[key] for key in keep if key in prediction}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("rizzo", "semif"), required=True)
    parser.add_argument("--semif", type=Path, required=True, help="SemIf repository checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("llama", "mlx"), default="llama", help="Rizzo only")
    parser.add_argument("--size", default="4b", help="Rizzo checkpoint: 4b or 1.7b")
    parser.add_argument("--quant", help="Rizzo on llama.cpp: q8_0 (default), q4_k_m, bf16")
    parser.add_argument("--weights", choices=("flow", "base"), help="pinned GGUF; default flow")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", help="Rizzo: GGUF file or MLX directory; SemIf: repository")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--bits", type=int, choices=(4, 8), help="MLX quantization")
    parser.add_argument("--batch-size", type=int, default=4, help="Rizzo suffix microbatch")
    parser.add_argument("--kv-type", choices=("f16", "q8_0", "q4_0"), help="Rizzo on llama.cpp")
    parser.add_argument("--shape-states", type=int, default=37, help="Shared-mode states")
    parser.add_argument("--direct-states", type=int, default=3, help="Fresh-mode states (slow)")
    args = parser.parse_args()
    sys.path[:0] = [str(args.semif / "benchmarks"), str(args.semif / "src")]
    import evaluate

    args.output.mkdir(parents=True)
    data = args.semif / "benchmarks" / "data"
    system = (Rizzo if args.system == "rizzo" else SemIf)(args)
    report = {"system": args.system, "model": system.metadata, "quality": {}, "shape": {}}

    # Quality: one fresh call per row, exactly as SemIf's MLX quality suite does.
    golds, scored = {}, {}
    for name in ("authored144", "perturbations108"):
        gold = golds[name] = read(data / f"{name}.jsonl")
        system.direct(gold[0])
        system.memory.reset()
        predictions, seconds = [], []
        for index, row in enumerate(gold):
            mark = time.perf_counter()
            predictions.append(slim(system.direct(row)))
            seconds.append(time.perf_counter() - mark)
            if (index + 1) % 36 == 0:
                print(f"{name}: {index + 1}/{len(gold)}", flush=True)
        result = evaluate.evaluate(gold, predictions)
        report["quality"][name] = {
            "mean_family_balanced_accuracy": result["mean_family_balanced_accuracy"],
            "mean_family_macro_f1": result["mean_family_macro_f1"],
            "families": {
                family: {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                for family, metrics in result["family_results"].items()
            },
            "errors": len(result["errors"]),
            "latency": latency(seconds),
            system.memory.key: system.memory.peak(),
        }
        scored[name] = predictions
        write(args.output / f"{name}.jsonl", predictions)
        print(name, json.dumps(report["quality"][name]["latency"]), flush=True)
    report["stability"] = stability(
        evaluate,
        golds["authored144"],
        scored["authored144"],
        golds["perturbations108"],
        scored["perturbations108"],
    )

    # Systems: ~2k-token states with 21 binary criteria each; no gold labels.
    groups = defaultdict(list)
    for row in read(data / "shape777.jsonl"):
        groups[row["group_id"]].append(row)
    groups = list(groups.values())
    runs = {}
    for mode, count in (("shared", args.shape_states), ("direct", args.direct_states)):
        selected = groups[:count]
        if not selected:
            continue

        def run(group, mode=mode):
            if mode == "shared":
                return system.shared(group)
            return [system.direct(row) for row in group]

        run(groups[0])
        system.memory.reset()
        predictions, seconds = [], []
        for index, group in enumerate(selected):
            mark = time.perf_counter()
            scored = run(group)
            seconds.append(time.perf_counter() - mark)
            predictions.extend(slim(p) for p in scored)
            print(f"shape/{mode}: {index + 1}/{len(selected)} {seconds[-1]:.2f}s", flush=True)
        runs[mode] = predictions
        report["shape"][mode] = {
            "states": len(selected),
            "decisions": len(predictions),
            "decisions_per_second": len(predictions) / sum(seconds),
            "state_latency": latency(seconds),
            "input_tokens_per_decision": statistics.mean(p["input_tokens"] for p in predictions),
            system.memory.key: system.memory.peak(),
        }
        write(args.output / f"shape777-{mode}.jsonl", predictions)
    if "direct" in runs and "shared" in runs:
        fresh = {p["id"]: p["probabilities"] for p in runs["direct"]}
        pairs = [(fresh[p["id"]], p["probabilities"]) for p in runs["shared"] if p["id"] in fresh]
        report["shape"]["shared_vs_direct"] = {
            "rows": len(pairs),
            "argmax_flips": sum(a.index(max(a)) != b.index(max(b)) for a, b in pairs),
            "max_probability_difference": max(abs(x - y) for a, b in pairs for x, y in zip(a, b)),
        }
    write(args.output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("quality", "shape")}, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
