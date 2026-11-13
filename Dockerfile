# Rizzo Flow API — llama.cpp backend (no MLX), CPU or GPU.
#
# The image carries the source of this checkout and its Python environment, so there is no
# commit to pin and no clone at build time: what you build is what you have. The llama.cpp
# binaries (~30 MB, or ~570 MB with CUDA) and the GGUF weights are fetched by the entrypoint
# into /data on first start, so the image stays small and the heavy files survive rebuilds,
# `docker compose down` and `docker image prune`.
#
# bookworm (glibc 2.36) runs the official ubuntu-x64 llama.cpp builds (max GLIBC 2.34).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/data/hf \
    PATH="/app/.venv/bin:$PATH"

# libgomp1: hard dependency of libggml-base.so.
# libvulkan1 + Mesa: the `vulkan` llama.cpp plug-in dlopens libvulkan.so.1 and the host GPU's
#   ICD (RADV for AMD, ANV for Intel). Unused with RIZZO_RUNTIME=cpu; adds ~40 MB.
# ca-certificates: HTTPS downloads (Hugging Face, GitHub releases) at first start.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Only what `uv sync` needs: pyproject.toml + uv.lock + the package source. Copying them
# separately keeps the dependency layer cached while the code changes.
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Small JSON requests for a smoke test without mounting anything (the container runs in
# /data, so the path is absolute):
#   docker compose run --rm rizzo rizzo decide /app/examples/ticket.json
COPY examples ./examples

COPY docker/entrypoint.sh /usr/local/bin/rizzo-entrypoint
RUN chmod 0755 /usr/local/bin/rizzo-entrypoint

# Everything writable lives on the volume: models/, runtimes/, hf/ (see docker-compose.yml).
WORKDIR /data
EXPOSE 8017

ENTRYPOINT ["/usr/local/bin/rizzo-entrypoint"]
# No CMD: the entrypoint serves the API, and any argument it is given runs instead
# (`docker compose run --rm rizzo rizzo decide /app/examples/ticket.json`).
CMD []
