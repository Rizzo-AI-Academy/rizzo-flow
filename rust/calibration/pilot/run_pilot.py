#!/usr/bin/env python3
"""End-to-end calibration pilot: run -> rows -> fit -> apply -> verify.

Exercises the whole calibration path that the unit tests cannot cover:

  1. run the engine on the pilot questions (uncalibrated) and keep the response;
  2. build `rows.jsonl` pairing each answer's option logits with its ground-truth
     label, in candidate order;
  3. fit temperatures with the `calibrate` subcommand;
  4. run again *with* the calibration and check that it is applied (per-answer
     temperature, probability_status, the `calibration` block) and that the
     probabilities actually move;
  5. negative test: a calibration whose fingerprint does not match must be refused.

Usage:
    calibration/pilot/run_pilot.py [--out DIR]
Environment:
    RIZZO_BIN    binary (default ./target/release/rizzo-flow-rs)
    RIZZO_MODEL  GGUF (default the pinned Qwen3.5-4B-SingleTurn)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN = os.environ.get("RIZZO_BIN", str(HERE.parent.parent / "target/release/rizzo-flow-rs"))
MODEL = os.environ.get(
    "RIZZO_MODEL",
    "/home/alforiva/inference/models/Qwen3.5-4B-Instruct-SingleTurn.Q4_K_M.gguf",
)


def run(args: list[str], expect_ok: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True)
    if expect_ok and result.returncode != 0:
        print(f"!! command failed: {' '.join(args)}", file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(1)
    return result


def decide(request: dict, extra: list[str] | None = None) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(request, handle)
        path = handle.name
    try:
        result = run([BIN, "decide", path, "--model", MODEL, "--ctx", "4096", "--ngl", "99"]
                     + (extra or []))
        return json.loads(result.stdout)
    finally:
        Path(path).unlink(missing_ok=True)


def candidate_order(question: dict) -> list[str]:
    """Candidate order both implementations agree on (see parity/core_vectors.json)."""
    if question["type"] == "choice":
        order = [option["id"] for option in question["options"]]
    else:
        order = ["false", "true"]
    if question.get("policy", {}).get("allow_abstain", True):
        order.append("__insufficient__")
    return order


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/calibration-pilot")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not Path(BIN).exists():
        print(f"!! binary not found: {BIN}", file=sys.stderr)
        return 2
    if not Path(MODEL).exists():
        print(f"!! model not found: {MODEL}", file=sys.stderr)
        return 2

    pilot = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    request, labels = pilot["request"], pilot["labels"]
    questions = request["questions"]

    print("1. run without calibration")
    baseline = decide(request)
    (out / "response_uncalibrated.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fingerprint = baseline["model"]["fingerprint"]
    print(f"   fingerprint: {fingerprint[:16]}…")
    correct = 0
    for qid, expected in labels.items():
        answer = baseline["answers"][qid]
        predicted = answer.get("choice", answer.get("value"))
        hit = str(predicted).lower() == expected.lower()
        correct += hit
    print(f"   pilot accuracy (uncalibrated): {correct}/{len(labels)}")

    print("2. build rows.jsonl from the run's own option logits")
    rows = []
    for qid, question in questions.items():
        answer = baseline["answers"][qid]
        order = candidate_order(question)
        logits = [answer["option_logits"][candidate] for candidate in order]
        label = labels[qid]
        if label not in order:
            print(f"!! label {label} is not a candidate of {qid}", file=sys.stderr)
            return 1
        rows.append({"type": answer["type"], "logits": logits, "label_index": order.index(label)})
    with (out / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"   {len(rows)} rows "
          f"({sum(1 for r in rows if r['type'] == 'choice')} choice, "
          f"{sum(1 for r in rows if r['type'] == 'boolean')} boolean)")

    print("3. fit temperatures")
    calibration_path = out / "calibration.json"
    calibration_path.unlink(missing_ok=True)
    run([BIN, "calibrate", str(out / "rows.jsonl"),
         "--fingerprint", fingerprint, "--output", str(calibration_path)])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    print(f"   temperatures: {json.dumps(calibration['temperatures'])}")
    print(f"   status: {calibration['status']}")

    print("4. run WITH the calibration")
    calibrated = decide(request, ["--calibration", str(calibration_path)])
    (out / "response_calibrated.json").write_text(
        json.dumps(calibrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    applied = {answer["temperature"] for answer in calibrated["answers"].values()}
    statuses = {answer["probability_status"] for answer in calibrated["answers"].values()}
    print(f"   calibration block present: {calibrated['calibration'] is not None}")
    print(f"   temperatures applied: {sorted(applied)}")
    print(f"   probability_status: {sorted(statuses)}")
    moved = 0
    max_shift = 0.0
    for qid, answer in calibrated["answers"].items():
        before = baseline["answers"][qid]["probabilities"]
        after = answer["probabilities"]
        shift = max(abs(after[k] - before[k]) for k in after)
        max_shift = max(max_shift, shift)
        moved += shift > 1e-9
    print(f"   probabilities moved in {moved}/{len(calibrated['answers'])} answers "
          f"(max shift {max_shift:.4f})")
    correct_after = 0
    for qid, expected in labels.items():
        answer = calibrated["answers"][qid]
        predicted = answer.get("choice", answer.get("value"))
        correct_after += str(predicted).lower() == expected.lower()
    print(f"   pilot accuracy (calibrated): {correct_after}/{len(labels)}")

    print("5. negative test: calibration for a different model must be refused")
    wrong = dict(calibration)
    wrong["fingerprint"] = "0" * 64
    wrong_path = out / "calibration_wrong.json"
    wrong_path.write_text(json.dumps(wrong), encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(request, handle)
        request_path = handle.name
    try:
        result = subprocess.run(
            [BIN, "decide", request_path, "--model", MODEL, "--ctx", "4096", "--ngl", "99",
             "--calibration", str(wrong_path)],
            capture_output=True, text=True)
    finally:
        Path(request_path).unlink(missing_ok=True)
    refused = result.returncode != 0 and "different model" in result.stderr
    print(f"   refused: {refused} (exit {result.returncode})")
    print(f"   message: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else '-'}")

    summary = {
        "fingerprint": fingerprint,
        "pilot_rows": len(rows),
        "accuracy_uncalibrated": f"{correct}/{len(labels)}",
        "accuracy_calibrated": f"{correct_after}/{len(labels)}",
        "temperatures": calibration["temperatures"],
        "fit_metrics": calibration["fit_metrics"],
        "probabilities_moved": f"{moved}/{len(calibrated['answers'])}",
        "max_probability_shift": max_shift,
        "calibration_block_present": calibrated["calibration"] is not None,
        "wrong_fingerprint_refused": refused,
    }
    (out / "pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nsummary written to {out / 'pilot_summary.json'}")
    return 0 if (moved > 0 and refused) else 1


if __name__ == "__main__":
    raise SystemExit(main())
