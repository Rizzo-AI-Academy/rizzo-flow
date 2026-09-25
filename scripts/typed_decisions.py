"""Score the local model on LocalLLaMA/typed-decisions (Hugging Face), zero-shot.

Every row is replayed as the body of a `POST /v1/systemone` request (`state` + `questions`,
unchanged), in process, through `compat.to_native` / `from_native`: five questions over one
shared state, one request per case. The time per case covers wire validation, translation,
prompt compilation, tokenization, forward passes and readout on a warm model; it has no HTTP
hop, unlike the client-side numbers in the dataset card.

The card names its metrics but not their formulas. The ones below are ours and are checked by
`--baselines`, which recomputes the card's Uniform and Prior rows without loading any model.

  python scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --baselines \
      --train .research/typed-decisions/all/train.jsonl
  python scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --quant q8_0 \
      --output results/local-typed-decisions/q8_0.json
"""

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

EPS = 1e-6  # probability floor for KL; the card does not say which one it uses


def load(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def labels(question):
    if question["type"] == "noul":
        return ["false", "true"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def argmax(dist, keys):
    return max(keys, key=lambda k: dist[k])  # ties go to the first key


def macro_f1(pairs, keys):
    scores = []
    for key in keys:
        tp = sum(1 for g, p in pairs if g == key and p == key)
        fp = sum(1 for g, p in pairs if g != key and p == key)
        fn = sum(1 for g, p in pairs if g == key and p != key)
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return statistics.fmean(scores)


def metrics(rows, predictions):
    """`predictions[case_id][question]` is a distribution over the gold label keys."""
    per = []
    by_question = defaultdict(list)
    for row in rows:
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for name, question in questions.items():
            keys = labels(question)
            g = {k: float(gold[name]["probabilities"].get(k, 0.0)) for k in keys}
            p = predictions[row["id"]][name]
            label = gold[name]["label"]
            guess = argmax(p, keys)
            clipped = {k: max(p[k], EPS) for k in keys}
            total = sum(clipped.values())
            entry = {
                "question": f"{row['workflow']}/{name}",
                "type": question["type"],
                "gold": label,
                "guess": guess,
                "correct": guess == label,
                "soft": g[guess],
                "kl": sum(g[k] * math.log(g[k] / (clipped[k] / total)) for k in keys if g[k] > 0),
                "tv": 0.5 * sum(abs(g[k] - p[k]) for k in keys),
                "brier": sum(
                    (g[k] - p[k]) ** 2 for k in keys
                ),  # summed over classes, as in the card
                "peak": p[guess],
            }
            if question["type"] == "score":
                expected = sum(int(k) * p[k] for k in keys)
                entry["mae"] = abs(expected - gold[name]["score"])
                entry["within1"] = abs(expected - gold[name]["score"]) <= 1
            per.append(entry)
            by_question[entry["question"]].append(entry)

    def summary(entries):
        bins = defaultdict(list)
        for e in entries:
            bins[min(int(e["peak"] * 10), 9)].append(e)
        ece = sum(
            len(b)
            / len(entries)
            * abs(
                statistics.fmean(e["peak"] for e in b) - statistics.fmean(e["correct"] for e in b)
            )
            for b in bins.values()
        )
        scored = [e for e in entries if "mae" in e]
        out = {
            "decisions": len(entries),
            "accuracy": statistics.fmean(e["correct"] for e in entries),
            "soft_accuracy": statistics.fmean(e["soft"] for e in entries),
            "kl": statistics.fmean(e["kl"] for e in entries),
            "tv": statistics.fmean(e["tv"] for e in entries),
            "brier": statistics.fmean(e["brier"] for e in entries),
            "ece": ece,
        }
        if scored:
            out["score_mae"] = statistics.fmean(e["mae"] for e in scored)
            out["within_1_level"] = statistics.fmean(e["within1"] for e in scored)
        return out

    overall = summary(per)
    f1s = []
    questions_out = {}
    for name, entries in sorted(by_question.items()):
        keys = sorted({e["gold"] for e in entries} | {e["guess"] for e in entries})
        f1 = macro_f1([(e["gold"], e["guess"]) for e in entries], keys)
        f1s.append(f1)
        questions_out[name] = summary(entries) | {"macro_f1": f1}
    overall["macro_f1"] = statistics.fmean(f1s)
    types = {t: summary([e for e in per if e["type"] == t]) for t in ("noul", "choice", "score")}
    workflows = {
        w: summary([e for e in per if e["question"].startswith(w + "/")])
        for w in sorted({r["workflow"] for r in rows})
    }
    return {
        "overall": overall,
        "by_type": types,
        "by_workflow": workflows,
        "by_question": questions_out,
    }


def baselines(rows, train):
    uniform = {}
    frequencies = defaultdict(Counter)
    for row in train:
        gold = json.loads(row["gold"])
        for name, answer in gold.items():
            frequencies[(row["workflow"], name)][answer["label"]] += 1
    prior = {}
    for row in rows:
        questions = json.loads(row["questions"])
        uniform[row["id"]] = {}
        prior[row["id"]] = {}
        for name, question in questions.items():
            keys = labels(question)
            uniform[row["id"]][name] = dict.fromkeys(keys, 1 / len(keys))
            counts = frequencies[(row["workflow"], name)]
            prior[row["id"]][name] = {k: counts[k] / sum(counts.values()) for k in keys}
    return {"uniform": metrics(rows, uniform)["overall"], "prior": metrics(rows, prior)["overall"]}


def run_model(rows, args):
    from rizzo_flow import compat
    from rizzo_flow.engine import Engine
    from rizzo_flow.loader import load_backend

    backend = load_backend(
        "llama",
        size=args.size,
        model=args.model,
        quant=None if args.model else args.quant,
        weights=None if args.model else args.weights,
        device=args.device,
        ctx=args.ctx,
    )
    engine = Engine(backend, ctx=args.ctx)
    served = compat.model_name(backend.metadata)

    def call(row):
        request = compat.SystemOneRequest.model_validate(
            {
                "state": json.loads(row["state"]),
                "model": served,
                "questions": json.loads(row["questions"]),
            }
        )
        native, options = compat.to_native(request)
        return compat.from_native(request, engine.decide(native), options, served)

    call(rows[0])  # warm-up, not timed
    predictions, seconds, cases = {}, [], []
    for index, row in enumerate(rows, 1):
        mark = time.perf_counter()
        response = call(row)
        seconds.append(time.perf_counter() - mark)
        predictions[row["id"]] = {}
        for name, answer in response["answers"].items():
            if answer["type"] == "noul":
                dist = {"false": 1 - answer["noul"], "true": answer["noul"]}
            else:
                dist = answer["probabilities"]
            predictions[row["id"]][name] = dist
        cases.append(
            {
                "id": row["id"],
                "ms": round(seconds[-1] * 1000, 2),
                "input_tokens": response["usage"]["input_tokens"],
                "answers": predictions[row["id"]],
            }
        )
        if index % 50 == 0:
            print(
                f"{index}/{len(rows)} cases, p50 {statistics.median(seconds) * 1000:.0f} ms",
                flush=True,
            )
    ordered = sorted(seconds)
    timing = {
        "p50_ms_per_case": statistics.median(seconds) * 1000,
        "p95_ms_per_case": ordered[int(0.95 * (len(ordered) - 1))] * 1000,
        "mean_ms_per_case": statistics.fmean(seconds) * 1000,
        "decisions_per_second": sum(len(c["answers"]) for c in cases) / sum(seconds),
        "input_tokens_p50": statistics.median(c["input_tokens"] for c in cases),
    }
    return served, backend.metadata, predictions, timing, cases


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", help="test split as JSONL (one parquet row per line)")
    parser.add_argument("--train", help="train split as JSONL, for the Prior baseline")
    parser.add_argument("--baselines", action="store_true", help="only Uniform and Prior, no model")
    parser.add_argument("--size", default="4b", choices=("4b", "1.7b"))
    parser.add_argument("--quant", default="q8_0")
    parser.add_argument("--weights", choices=("flow", "base"), help="pinned GGUF; default flow")
    parser.add_argument(
        "--model", help="a GGUF file instead of the pinned one (e.g. a merged LoRA)"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = load(args.data)
    if args.baselines:
        print(json.dumps(baselines(rows, load(args.train)), indent=2))
        return
    served, metadata, predictions, timing, cases = run_model(rows, args)
    report = {
        "dataset": "LocalLLaMA/typed-decisions",
        "data": str(Path(args.data)),
        "model": served,
        "gguf": args.model,
        "kind": (
            "fine-tuned (rizzo-flow LoRA; no typed-decisions workflows in training)"
            if metadata.get("weights") == "flow"
            else "general, zero-shot"
        ),
        "metadata": {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))},
        "kl_floor": EPS,
        "timing": timing,
        "metrics": metrics(rows, predictions),
        "cases": cases,
    }
    print(
        json.dumps(
            {"model": served, "timing": timing, "overall": report["metrics"]["overall"]}, indent=2
        )
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "x", encoding="utf-8") as stream:  # results are create-only
            json.dump(report, stream, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
