#!/usr/bin/env bash
# Vendor `llama-cpp-sys-2` with a newer llama.cpp, so this port can load models whose
# architecture the published crate predates (today: Spark-X2.5 / `spark2_5`).
#
# Why this is needed
# ------------------
# `llama-cpp-2` 0.1.156 (the latest) vendors a llama.cpp older than the `spark2_5`
# architecture. The project this port follows uses llama.cpp release b11081, which has it.
# Replacing the vendored sources and adapting three call sites in the crate's C++ wrapper
# is enough — see REPORT.md §3.11 for the measured result.
#
# What it does
# ------------
#   1. clones llama.cpp at the pinned tag into the vendor directory;
#   2. copies the crate from the cargo registry and swaps its `llama.cpp/` for that clone;
#   3. applies the three wrapper adaptations (idempotent: safe to re-run);
#   4. prints the `[patch.crates-io]` block to paste into Cargo.toml.
#
# Caveats, stated plainly
# -----------------------
#   - this is a *local patch*, not a published crate: it must be re-applied after a cargo
#     registry clean, and it is not what the delivered code uses by default;
#   - the build is long (a full llama.cpp CUDA build took ~59 minutes on 4 cores);
#   - the patch tracks a specific llama.cpp tag; a different one may need more work.
#
# Usage:
#   tools/vendor_llama_cpp.sh [vendor-dir] [llama.cpp-tag]
# Defaults: vendor dir `vendor/llama-cpp-sys-2`, tag `b11081`.
set -euo pipefail

VENDOR_DIR="${1:-vendor/llama-cpp-sys-2}"
LLAMA_TAG="${2:-b11081}"
CRATE_NAME="llama-cpp-sys-2"
CRATE_VERSION="0.1.156"

echo "vendoring $CRATE_NAME $CRATE_VERSION with llama.cpp $LLAMA_TAG -> $VENDOR_DIR"

# 1. llama.cpp sources at the pinned tag
if [[ ! -d "$VENDOR_DIR/llama.cpp/.git" ]]; then
  work="$(mktemp -d)"
  echo "  cloning llama.cpp $LLAMA_TAG"
  git clone --quiet --depth 1 --branch "$LLAMA_TAG" \
      https://github.com/ggml-org/llama.cpp "$work/llama.cpp"
else
  echo "  llama.cpp already present, leaving it alone"
fi

# 2. the crate, from the cargo registry
if [[ ! -d "$VENDOR_DIR/src" ]]; then
  crate_path="$(find "${CARGO_HOME:-$HOME/.cargo}/registry/src" -maxdepth 2 \
      -type d -name "$CRATE_NAME-$CRATE_VERSION" | head -1)"
  if [[ -z "$crate_path" ]]; then
    echo "!! $CRATE_NAME-$CRATE_VERSION not found in the cargo registry." >&2
    echo "   Add it as a dependency and build once, or unpack it manually." >&2
    exit 2
  fi
  echo "  copying crate from $crate_path"
  mkdir -p "$(dirname "$VENDOR_DIR")"
  cp -r "$crate_path" "$VENDOR_DIR"
  chmod -R u+w "$VENDOR_DIR"
  rm -rf "$VENDOR_DIR/llama.cpp"
fi
if [[ ! -d "$VENDOR_DIR/llama.cpp" && -n "${work:-}" ]]; then
  mv "$work/llama.cpp" "$VENDOR_DIR/llama.cpp"
  rm -rf "$work"
fi

# 3. the three wrapper adaptations
WRAPPER="$VENDOR_DIR/wrapper_common.cpp"
python3 - "$WRAPPER" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
changes = []

# (a) b11081 inserted `const common_fit_extra_model * extra` before log_level.
old = "        n_ctx_min,\n        log_level));"
new = ("        n_ctx_min,\n"
       "        nullptr, // newer llama.cpp added `const common_fit_extra_model * extra` here\n"
       "        log_level));")
if old in text:
    text = text.replace(old, new)
    changes.append("common_fit_params: pass nullptr for the new extra model parameter")

# (b) json_schema_to_grammar takes llama.cpp's own common_json (which is not nlohmann::json).
old = "const auto schema = nlohmann::json::parse(schema_json);"
if old in text:
    text = text.replace(old, "")
if "json_schema_to_grammar(schema, force_gbnf)" in text:
    text = text.replace(
        "json_schema_to_grammar(schema, force_gbnf)",
        "json_schema_to_grammar(common_json::parse(schema_json), force_gbnf)")
    changes.append("json_schema_to_grammar: parse with common_json, as llama.cpp itself does")

path.write_text(text)
if changes:
    for change in changes:
        print(f"  adaptation: {change}")
else:
    print("  adaptations already applied (nothing to do)")
PY

# 4. the patch block
cat <<EOF

Now add this to the project's Cargo.toml:

  [patch.crates-io]
  $CRATE_NAME = { path = "$VENDOR_DIR" }

Then build. The CUDA build of llama.cpp takes a while (about an hour on 4 cores):

  CARGO_TARGET_DIR=/path/with/space cargo build --release --features cuda

Verify with a model the published crate cannot load (Spark-X2.5):

  rizzo-flow-rs decide request.json --model Spark-X2.5-4B-Q8_0.gguf \\
      --chat-template templates/spark-x2.5-nonthinking.jinja
EOF
