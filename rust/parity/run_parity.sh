#!/usr/bin/env bash
# Parity suite: the Rust port against the original Rizzo Flow (Python).
#
#   Layer A (always)        decision-core parity, model-free and exact:
#                           same question + same logits -> same answer, field by field.
#   Layer B (--end-to-end)  response-shape parity through both full engines.
#                           The engines run DIFFERENT models (Spark/MLX vs GGUF/llama.cpp),
#                           so values differ by design: only the structure is checked.
#
# Usage:
#   parity/run_parity.sh [--end-to-end]
#
# Environment:
#   RIZZO_REFERENCE  path of the original Python checkout (default /mnt/disco1/rizzo-flow)
#   RIZZO_MODEL      GGUF used for layer B
#   RIZZO_REQUEST    request file used for layer B (default <reference>/rizzo_test_6.json)
#   PARITY_OUT       output directory (default /tmp/parity)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE="${RIZZO_REFERENCE:-/mnt/disco1/rizzo-flow}"
MODEL="${RIZZO_MODEL:-/home/alforiva/inference/models/Qwen3.5-4B-Instruct-SingleTurn.Q4_K_M.gguf}"
REQUEST="${RIZZO_REQUEST:-$REFERENCE/rizzo_test_6.json}"
OUT="${PARITY_OUT:-/tmp/parity}"
PY="$REFERENCE/.venv/bin/python"
FAILED=0

mkdir -p "$OUT"

echo "parity suite"
echo "  repo:      $REPO"
echo "  reference: $REFERENCE"
echo "  output:    $OUT"

if [[ ! -x "$PY" ]]; then
  echo "!! reference virtualenv not found at $PY" >&2
  exit 2
fi

echo
echo "=== Layer A · decision core (exact parity, model-free)"
cargo build --release --example parity_vectors --manifest-path "$REPO/Cargo.toml" >/dev/null
"$REPO/target/release/examples/parity_vectors" "$REPO/parity/core_vectors.json" > "$OUT/rust_core.json"
(cd "$REFERENCE" && PYTHONPATH=src "$PY" "$REPO/parity/python_side.py" "$REPO/parity/core_vectors.json") > "$OUT/python_core.json"
python3 "$REPO/parity/compare.py" \
  "$OUT/python_core.json" "$OUT/rust_core.json" \
  --tol 1e-9 --output "$OUT/report_core.json" || FAILED=1

if [[ "${1:-}" == "--end-to-end" ]]; then
  echo
  echo "=== Layer B · end-to-end response shape (different models: values differ by design)"
  if [[ ! -f "$MODEL" ]]; then
    echo "!! model not found: $MODEL" >&2
    exit 2
  fi
  if pgrep -x rizzo-flow-rs >/dev/null 2>&1; then
    echo "!! a rizzo-flow-rs server is running and holds VRAM; stop it first" >&2
    exit 2
  fi
  # The MLX-CUDA reference aborts during interpreter teardown (after the response has
  # been written: "terminate called without an active exception"). Tolerate the exit
  # status and validate the artefact instead — that is the thing we compare.
  (cd "$REFERENCE" && PYTHONPATH=src "$PY" "$REPO/parity/python_engine.py" "$REQUEST") \
    > "$OUT/python_decide.json" 2>"$OUT/python_engine.log" || true
  python3 - "$OUT/python_decide.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    response = json.loads(path.read_text(encoding="utf-8"))
except Exception as error:  # noqa: BLE001
    print(f"!! the reference produced no usable response ({error}); see python_engine.log", file=sys.stderr)
    raise SystemExit(2)
expected = {"model", "mode", "answers", "calibration", "timing"}
missing = expected - set(response)
if missing:
    print(f"!! reference response is missing {sorted(missing)}", file=sys.stderr)
    raise SystemExit(2)
print(f"reference response: {len(response['answers'])} answers, "
      f"inference {response['timing']['inference_seconds']:.2f}s")
PY
  "$REPO/target/release/rizzo-flow-rs" decide "$REQUEST" \
    --model "$MODEL" --ctx 4096 --ngl 99 > "$OUT/rust_decide.json"
  # Declared differences (SPEC §1-bis point 4): the model identity reports the
  # backend actually used, and the timing carries backend-specific counters.
  python3 "$REPO/parity/compare.py" \
    "$OUT/python_decide.json" "$OUT/rust_decide.json" \
    --shape-only \
    --documented-prefix model \
    --documented-prefix timing.evaluated_tokens_including_padding \
    --documented-prefix timing.peak_mlx_bytes \
    --documented-prefix timing.load_seconds \
    --output "$OUT/report_shape.json" || FAILED=1
fi

echo
if [[ "$FAILED" -eq 0 ]]; then
  echo "PARITY OK — reports in $OUT"
else
  echo "PARITY FAILED — inspect $OUT" >&2
fi
exit "$FAILED"
