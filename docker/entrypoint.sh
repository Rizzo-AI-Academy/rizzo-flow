#!/bin/sh
# Rizzo Flow API inside the container.
#
# First start: fetch the llama.cpp build and the GGUF into /data (sha256-verified, resumable,
# skipped when the file is already there), then serve on 0.0.0.0:8017 — the compose port
# mapping decides who can reach it. Every setting comes from the environment documented in
# .env.example.
#
# Any argument replaces the command:
#   docker compose run --rm rizzo rizzo decide /app/examples/ticket.json
set -eu

say() { printf 'rizzo: %s\n' "$*" >&2; }

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

size="${RIZZO_SIZE:-4b}"
quant="${RIZZO_QUANT:-q8_0}"
weights="${RIZZO_WEIGHTS:-flow}"
runtime="${RIZZO_RUNTIME:-cpu}"
device="${RIZZO_DEVICE:-auto}"
ctx="${RIZZO_CTX:-8192}"
batch="${RIZZO_BATCH_SIZE:-4}"
port=8017

# Same checks the CLI makes, without hashing gigabytes on every start: a runtime directory
# that holds libllama, a GGUF file at the pinned path. A truncated download fails loudly when
# the weights are loaded, and `rizzo download` verifies the sha256 before writing it.
have_runtime() {
    python3 - "$runtime" <<'PY'
import sys

from rizzo_flow import llama_release

sys.exit(0 if llama_release.find_library(llama_release.install_dir(sys.argv[1])) else 1)
PY
}

have_weights() {
    python3 - "$size" "$quant" "$weights" <<'PY'
import sys

from rizzo_flow.config import gguf_spec

sys.exit(0 if gguf_spec(sys.argv[1], sys.argv[2], sys.argv[3]).path.is_file() else 1)
PY
}

if [ "${RIZZO_SKIP_DOWNLOAD:-0}" = "1" ]; then
    say "RIZZO_SKIP_DOWNLOAD=1: not checking runtime and weights"
else
    if have_runtime; then
        say "llama.cpp $runtime runtime already unpacked in /data/runtimes"
    else
        say "downloading the llama.cpp $runtime runtime into /data/runtimes"
        rizzo download --only runtime --runtime "$runtime"
    fi
    if have_weights; then
        say "$size $quant ($weights) weights already in /data/models"
    else
        say "downloading the $size $quant ($weights) GGUF into /data/models; this can take a while"
        rizzo download --only weights --size "$size" --quant "$quant" --weights "$weights"
    fi
fi

set -- serve \
    --size "$size" \
    --quant "$quant" \
    --weights "$weights" \
    --device "$device" \
    --ctx "$ctx" \
    --batch-size "$batch" \
    --host 0.0.0.0 \
    --port "$port"

if [ -n "${RIZZO_THREADS:-}" ]; then
    set -- "$@" --threads "$RIZZO_THREADS"
fi
if [ -n "${RIZZO_KV_TYPE:-}" ]; then
    set -- "$@" --kv-type "$RIZZO_KV_TYPE"
fi
if [ -n "${RIZZO_CALIBRATION:-}" ]; then
    set -- "$@" --calibration "$RIZZO_CALIBRATION"
fi

say "serving rizzo-flow $size $weights $quant on 0.0.0.0:$port (device: $device)"
# exec keeps PID 1 on uvicorn, so `docker compose stop` is a clean SIGTERM.
exec rizzo "$@"
