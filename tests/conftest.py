"""Test-wide setup, applied before any test module imports MLX."""

import os
from pathlib import Path

import pytest

# TF32 rounding on NVIDIA GPUs exceeds the exact-equivalence tolerances of test_mlx.py.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import rizzo_flow  # noqa: F401  (prepares the MLX stack on Windows)

# Real-weight llama.cpp tests are opt-in: they load a 4 GB checkpoint, so they must never run
# in the default fast suite. `RIZZO_LLAMA_TEST=1 .venv/bin/pytest -q -m llama`.
LLAMA_LIBRARY = Path(
    os.environ.get("RIZZO_LLAMA_LIB", "/home/snorcini/git/llama.cpp-amd/build/bin")
)
LLAMA_GGUF = Path(
    os.environ.get("RIZZO_LLAMA_GGUF", "/mnt/model-cache/rizzo-flow/Spark-X2.5-4B-Q8_0.gguf")
)


@pytest.fixture
def llama_runtime():
    """A loaded Spark GGUF on the pinned llama.cpp build, or a skip."""
    ready = (
        os.environ.get("RIZZO_LLAMA_TEST") == "1"
        and (LLAMA_LIBRARY / "libllama.so").exists()
        and LLAMA_GGUF.is_file()
    )
    if not ready:
        pytest.skip("set RIZZO_LLAMA_TEST=1 with libllama and a GGUF to run real-weight tests")
    from rizzo_flow.llama_runtime import LlamaRuntime

    runtime = LlamaRuntime.load(LLAMA_LIBRARY, LLAMA_GGUF, n_ctx=4096, n_seq_max=5, device="auto")
    try:
        yield runtime
    finally:
        runtime.close()
