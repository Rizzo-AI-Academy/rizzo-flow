"""Choose and load the scoring backend: llama.cpp by default, MLX or MAX on request."""

from pathlib import Path

from .config import DEFAULT_SIZE, checkpoint_path, gguf_spec

BACKENDS = ("llama", "mlx", "max")
# `auto`, `gpu` and `cpu` work everywhere. The other names ask for one GPU family: `mlx` and
# `cuda` also keep their meaning for the MLX backend, the rest exist in llama.cpp only.
DEVICES = ("auto", "gpu", "cpu", "cuda", "metal", "vulkan", "rocm", "sycl", "mlx")
MLX_DEVICES = ("auto", "gpu", "cpu", "cuda", "mlx")
MAX_DEVICES = ("auto", "gpu", "cpu")


def load_backend(
    backend="llama",
    *,
    size=DEFAULT_SIZE,
    model=None,
    quant=None,
    weights=None,
    bits=None,
    device="auto",
    ctx=8192,
    batch_size=4,
    threads=None,
    kv_type=None,
    prefill_chunk=None,
    max_resident=None,
):
    if backend not in BACKENDS:
        raise ValueError(f"Backend must be one of: {', '.join(BACKENDS)}")
    if backend == "max":
        if quant or bits:
            raise ValueError("--quant and --bits do not apply to the MAX backend (BF16 only)")
        if device not in MAX_DEVICES:
            raise ValueError(f"--device {device}: the MAX backend takes auto, gpu or cpu")
        if kv_type:
            raise ValueError("--kv-type exists only in the llama backend")
        if model and weights:
            raise ValueError(
                "--weights picks a pinned checkpoint; with --model the files are yours"
            )
        from .backend_max import MaxBackend

        return MaxBackend.load(
            model or checkpoint_path(size, weights),
            device=device,
            ctx=ctx,
            batch_size=batch_size,
            **({"prefill_chunk": prefill_chunk} if prefill_chunk else {}),
            **({"max_resident": max_resident} if max_resident else {}),
        )
    if max_resident:
        raise ValueError("--max-resident exists only in the MAX backend")
    if backend == "mlx":
        if prefill_chunk:
            raise ValueError("prefill_chunk is not configurable on the MLX backend")
        if quant:
            raise ValueError("--quant selects a GGUF file (llama backend); with MLX use --bits 4|8")
        if device not in MLX_DEVICES:
            raise ValueError(f"--device {device} exists only in the llama backend")
        if kv_type:
            raise ValueError("--kv-type exists only in the llama backend")
        from .backend import SparkBackend

        if model and weights:
            raise ValueError(
                "--weights picks a pinned checkpoint; with --model the files are yours"
            )
        return SparkBackend.load(
            model or checkpoint_path(size, weights),
            bits=bits,
            device=device,
            batch_size=batch_size,
        )
    if bits:
        raise ValueError(
            "--bits quantizes MLX weights in memory (--backend mlx); llama.cpp loads a "
            "quantized file instead: --quant q8_0 (default), q4_k_m or bf16"
        )
    if device == "mlx":
        raise ValueError("--device mlx needs --backend mlx; with llama.cpp use --device metal")
    if model and weights:
        raise ValueError("--weights picks a pinned file; with --model the file is yours")
    if model and Path(model).is_dir():
        raise ValueError(
            f"{model} is a checkpoint directory (MLX); llama.cpp needs a .gguf file. "
            "Pass --backend mlx, or a GGUF path, or drop --model to use the pinned file."
        )
    from .backend_llama import LlamaBackend

    return LlamaBackend.load(
        model or gguf_spec(size, quant, weights).path,
        device=device,
        ctx=ctx,
        batch_size=batch_size,
        threads=threads,
        kv_type=kv_type,
        **({"prefill_chunk": prefill_chunk} if prefill_chunk else {}),
    )


def describe() -> dict:
    """What `rizzo devices` prints: the llama.cpp runtime and its devices, MLX and MAX if present."""
    from . import llama_release
    from .llama_cpp import Library, choose_device

    report = {
        "llama.cpp": {
            "release": llama_release.RELEASE,
            "host": "/".join(llama_release.host()),
            "packages": llama_release.supported(),
            "recommended": None,
            "installed": llama_release.installed(),
        }
    }
    section = report["llama.cpp"]
    try:
        section["recommended"] = llama_release.pick("auto")
        section["directory"] = str(llama_release.locate())
        devices = Library.open().devices()
        chosen = choose_device(devices, "auto")
        section["devices"] = [device.public() for device in devices]
        section["auto_selects"] = chosen.name if chosen else "CPU"
    except (ValueError, OSError) as error:
        section["error"] = str(error)
    try:
        from .runtime import describe as describe_mlx

        report["mlx"] = describe_mlx()
    except ImportError:
        report["mlx"] = {"installed": False}
    try:
        from .backend_max import describe as describe_max

        report["max"] = describe_max()
    except ImportError:
        report["max"] = {"installed": False}
    return report
