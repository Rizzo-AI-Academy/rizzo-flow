"""Backend selection: the original MLX stack or the llama.cpp GGUF backend.

Both behind the same surface `Engine` consumes. This module is the only place that knows
which backend is active, so the decision layer stays untouched.
"""

import importlib.util
from pathlib import Path

from .config import DEFAULT_SIZE, GGUF_MODELS, MODELS, gguf_path
from .llama_runtime import DEVICES as LLAMA_DEVICES
from .runtime import DEVICES as MLX_DEVICES

BACKENDS = ("auto", "mlx", "llama")
# Union of both stacks; each loader keeps only the names it understands.
DEVICE_CHOICES = tuple(dict.fromkeys((*MLX_DEVICES, *LLAMA_DEVICES)))
DEFAULT_QUANT = "q8_0"


def llama_available() -> bool:
    from .backend_llama import library_dir

    return (library_dir() / "libllama.so").exists()


def mlx_available() -> bool:
    return importlib.util.find_spec("mlx") is not None and (
        importlib.util.find_spec("spark_mlx_llm") is not None
    )


def available_backends() -> list[str]:
    return [name for name, ready in (("mlx", mlx_available), ("llama", llama_available)) if ready()]


def select_backend(requested: str, gguf, available: list[str] | None = None) -> str:
    """Resolve `auto`: a GGUF means llama.cpp, otherwise prefer MLX and fall back to llama.cpp."""
    if requested not in BACKENDS:
        raise ValueError(f"Backend must be one of: {', '.join(BACKENDS)}")
    if requested != "auto":
        return requested
    if gguf:
        return "llama"
    available = available_backends() if available is None else available
    if "mlx" in available:
        return "mlx"
    if "llama" in available:
        return "llama"
    raise ValueError(
        "No backend available: install the MLX extra or build llama.cpp and set RIZZO_LLAMA_LIB"
    )


def ensure_options(backend: str, *, bits, device: str) -> None:
    """Reject combinations that would otherwise fail deep inside a loader."""
    names = LLAMA_DEVICES if backend == "llama" else MLX_DEVICES
    if device not in names:
        raise ValueError(
            f"--device {device} does not apply to the {backend} backend; use: {', '.join(names)}"
        )
    if backend == "llama" and bits is not None:
        raise ValueError(
            "--bits quantizes MLX weights in memory; with llama.cpp choose a quantized GGUF "
            "instead (--gguf PATH or --quant " + "|".join(GGUF_MODELS) + ")"
        )


def load_backend(
    backend: str = "auto",
    *,
    size: str = DEFAULT_SIZE,
    model: Path | None = None,
    gguf: Path | None = None,
    quant: str = DEFAULT_QUANT,
    bits: int | None = None,
    device: str = "auto",
    ctx: int = 8192,
    batch_size: int = 4,
    prefill_chunk: int = 512,
    threads: int | None = None,
):
    backend = select_backend(backend, gguf)
    ensure_options(backend, bits=bits, device=device)
    if backend == "llama":
        from .backend_llama import LlamaBackend

        if gguf:
            path, expected = Path(gguf), None
        elif size in MODELS:
            path, expected = gguf_path(quant), GGUF_MODELS[quant].sha256
        else:
            raise ValueError(f"Unknown size {size}; available: {', '.join(MODELS)}")
        return LlamaBackend.load(
            path,
            device=device,
            ctx=ctx,
            batch_size=batch_size,
            prefill_chunk=prefill_chunk,
            threads=threads,
            expected_sha256=expected,
        )
    from .backend import SparkBackend

    if quant != DEFAULT_QUANT and not gguf:
        raise ValueError("--quant applies to the llama backend only")
    return SparkBackend.load(
        model or MODELS[size].path,
        bits=bits,
        device=device,
        batch_size=batch_size,
        prefill_chunk=prefill_chunk,
    )


def describe() -> dict:
    """What `rizzo devices` prints: both stacks, without importing MLX on a llama-only box."""
    from .backend_llama import library_dir

    report = {
        "available": available_backends(),
        "llama": {
            "library": str(library_dir()),
            "present": llama_available(),
            "devices": list(LLAMA_DEVICES),
        },
    }
    if mlx_available():
        from .runtime import describe as describe_mlx

        report["mlx"] = describe_mlx()
    else:
        report["mlx"] = {"available": False}
    return report
