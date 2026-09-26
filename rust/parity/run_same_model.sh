#!/usr/bin/env bash
# Layer C — the SAME model on both engines.
#
# Layer B compares only the response *shape*, because the reference runs Spark-X2.5 and the
# published crate cannot load that architecture. With a vendored llama.cpp (§3.11 of
# REPORT.md) and the built-in Spark format, the same model can now run on both sides and the
# *values* can be compared — which is what this does:
#
#   * prompt_sha256 must be identical: the two engines must send the same prompt;
#   * the answers must be identical;
#   * the probability deltas are reported (they are not zero: the two sides run different
#     quantisations of the same model — GGUF Q8_0 against MLX bits=8).
#
# Usage:
#   parity/run_same_model.sh [request.json]
#
# Environment:
#   RIZZO_BIN        our binary, built against the vendored llama.cpp
#   RIZZO_MODEL      Spark GGUF (default the pinned Q8_0)
#   RIZZO_CHAT_FORMAT  file containing the built-in format id (default: created here)
#   RIZZO_REFERENCE  the Python checkout (default /mnt/disco1/rizzo-flow)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE="${RIZZO_REFERENCE:-/mnt/disco1/rizzo-flow}"
BIN="${RIZZO_BIN:-$REPO/target/release/rizzo-flow-rs}"
MODEL="${RIZZO_MODEL:-/home/alforiva/inference/models/Spark-X2.5-4B-Q8_0.gguf}"
REQUEST="${1:-$REFERENCE/rizzo_test_6.json}"
OUT="${PARITY_OUT:-/tmp/parity}"
PY="$REFERENCE/.venv/bin/python"
FORMAT_FILE="${RIZZO_CHAT_FORMAT:-$OUT/spark-format.txt}"

mkdir -p "$OUT"
printf 'spark-x2.5-nonthinking' > "$FORMAT_FILE"

for path in "$BIN" "$MODEL" "$REQUEST"; do
  [[ -e "$path" ]] || { echo "!! missing: $path" >&2; exit 2; }
done

echo "=== Layer C · same model on both engines"
echo "  model:   $MODEL"
echo "  request: $REQUEST"

# The reference: Spark on MLX, thinking disabled (the configuration it uses).
if [[ -x "$PY" ]]; then
  echo "  running the reference (MLX)…"
  (cd "$REFERENCE" && PYTHONPATH=src "$PY" "$REPO/parity/python_engine.py" "$REQUEST") \
    > "$OUT/reference_same_model.json" 2>"$OUT/reference_same_model.log" || true
else
  echo "!! reference virtualenv not found at $PY" >&2
  exit 2
fi

echo "  running this port (llama.cpp)…"
"$BIN" decide "$REQUEST" --model "$MODEL" --chat-template "$FORMAT_FILE" \
  --ctx 4096 --ngl 99 > "$OUT/rust_same_model.json" 2>"$OUT/rust_same_model.log" || {
    echo "!! our run failed; see $OUT/rust_same_model.log" >&2; exit 1; }

python3 - "$OUT/reference_same_model.json" "$OUT/rust_same_model.json" <<'PY'
import json, sys
reference = json.loads(open(sys.argv[1]).read())
ours = json.loads(open(sys.argv[2]).read())
questions = list(reference["answers"])
same_prompt = sum(reference["answers"][q]["prompt_sha256"] == ours["answers"][q]["prompt_sha256"] for q in questions)
def answer_of(a):
    return a.get("choice", a.get("value"))
same_answer = sum(str(answer_of(reference["answers"][q])) == str(answer_of(ours["answers"][q])) for q in questions)
worst = 0.0
for q in questions:
    a, b = reference["answers"][q]["probabilities"], ours["answers"][q]["probabilities"]
    worst = max(worst, max(abs(a[k] - b[k]) for k in a))
print()
print(f"  prompts identical : {same_prompt}/{len(questions)}")
print(f"  answers identical : {same_answer}/{len(questions)}")
print(f"  max probability delta: {worst:.3e}   (different quantisations of the same model)")
for q in questions:
    r, o = answer_of(reference["answers"][q]), answer_of(ours["answers"][q])
    mark = "OK  " if str(r) == str(o) else "DIFF"
    print(f"    {mark} {q:<16} reference={str(r):<14} ours={o}")
print()
ok = same_prompt == len(questions) and same_answer == len(questions)
print("SAME-MODEL PARITY OK" if ok else "SAME-MODEL PARITY FAILED")
sys.exit(0 if ok else 1)
PY
