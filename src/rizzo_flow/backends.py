"""Backend selection: which compute stack runs the model, and on which device.

Two stacks exist — the original MLX one (`backend.SparkBackend`) and the llama.cpp one
(`backend_llama.LlamaBackend`) — behind the single surface `Engine` consumes. This module is
the only place that decides between them, so the decision layer stays untouched.

`route()` answers the question a user actually asks: "run on whatever GPU this machine has".
It combines three inputs neither stack can supply alone:

* the accelerators physically present (`hardware.probe`),
* what each stack can drive on this install (`mlx_devices` / `llama_devices`),
* the user's preference (`--device`, `--backend`, `--gguf`).

Capabilities, not install presence, decide the winner: a machine with `mlx-cpu` installed and
an AMD card routes to llama.cpp, because MLX there has no accelerator to offer.
"""

import importlib.util
import warnings
from dataclasses import dataclass
from pathlib import Path

from . import hardware
from .config import DEFAULT_SIZE, GGUF_MODELS, MODELS, gguf_path
from .llama_runtime import DEVICES as LLAMA_DEVICES

BACKENDS = ("auto", "mlx", "llama")
# User-facing device names. The vendor names describe hardware, `gpu` means "any
# accelerator", and `cuda` is accepted as the colloquial NVIDIA synonym.
DEVICES = ("auto", "gpu", "apple", "nvidia", "amd", "intel", "cpu")
ACCELERATOR_ALIASES = {"cuda": "nvidia"}
# Legacy names that pin the implementation rather than the hardware; kept working.
STACK_DEVICES = {"mlx": "mlx", "vulkan": "llama", "hip": "llama"}
DEVICE_CHOICES = tuple(dict.fromkeys((*DEVICES, *STACK_DEVICES)))
# The quant to download when only `--size` is given; derived so it cannot drift from the pins.
DEFAULT_QUANT = next(iter(GGUF_MODELS))
# On CPU, llama.cpp is the validated path and MLX's CPU backend is documented as unusable.
CPU_STACK_PREFERENCE = ("llama", "mlx")


@dataclass(frozen=True)
class Routing:
    """A resolved choice, with the reason, so callers can explain it to the user."""

    backend: str
    device: str
    reason: str
    downgraded: bool = False  # the request implied a GPU but only CPU is usable


def mlx_importable() -> bool:
    """The MLX wheel is installed. Cheap: no import, so a llama-only box stays untouched."""
    return importlib.util.find_spec("mlx") is not None and (
        importlib.util.find_spec("spark_mlx_llm") is not None
    )


def mlx_devices() -> dict[str, str]:
    """Accelerator -> MLX device name. Empty when MLX is absent or has no usable GPU.

    This asks for a *usable accelerator*, not for the package: `mlx-cpu` imports on any
    machine, and routing to it where a real GPU exists costs minutes per decision.
    """
    if not mlx_importable():
        return {}
    from .runtime import accelerator, import_mlx

    try:
        gpu = accelerator(import_mlx())
    except ImportError:
        return {}  # present but broken: prefer a stack that works
    devices = {"cpu": "cpu"}
    if gpu == "mlx":
        devices["apple"] = "mlx"
    elif gpu == "cuda":
        devices["nvidia"] = "cuda"
    return devices


def llama_devices() -> dict[str, str]:
    """Accelerator -> llama.cpp device name, from the ggml backends this build ships."""
    from .backend_llama import library_dir, library_file
    from .llama_runtime import detect_backends

    base = library_dir()
    if library_file(base) is None:
        return {}
    shipped = detect_backends(base)
    if not shipped:
        # A CPU-only build still runs, just slowly.
        # NOTICED: llama.cpp on macOS uses Metal (libggml-metal.dylib), which detect_backends
        # does not look for, so a Mac asking for --gguf routes to CPU. MLX is the Mac path.
        return {"cpu": "cpu"}
    # detect_backends is already in preference order, so the first entry is the one
    # `llama_runtime.resolve` would pick: keeping the two in step avoids a surprise downgrade.
    served = {
        "cuda": {"nvidia": "cuda"},
        "vulkan": {"nvidia": "vulkan", "amd": "vulkan", "intel": "vulkan"},
        "hip": {"amd": "hip"},
    }[shipped[0]]
    return {**served, "cpu": "cpu"}


def llama_present(directory: Path | None = None) -> bool:
    """libllama is where we expect it, under this platform's own file name."""
    from .backend_llama import library_dir, library_file

    return library_file(directory or library_dir()) is not None


def available_backends() -> list[str]:
    """Stacks this install has at all; `describe` uses it, routing uses the capabilities."""
    return [name for name, ready in (("mlx", mlx_importable), ("llama", llama_present)) if ready()]


def candidate_stacks(
    present: list[str], mlx_caps: dict[str, str], llama_caps: dict[str, str]
) -> list[tuple[str, str, str]]:
    """(accelerator, backend, device) in the order `auto` tries them, best accelerator first.

    MLX comes before llama.cpp for the same accelerator where both can drive it: that is the
    documented default, not a measured result — no MLX-vs-llama figure exists for one GPU.
    """
    table = []
    for accelerator in present:
        if accelerator == "cpu":
            continue
        if accelerator in mlx_caps:
            table.append((accelerator, "mlx", mlx_caps[accelerator]))
        if accelerator in llama_caps:
            table.append((accelerator, "llama", llama_caps[accelerator]))
    return table


def _unavailable(device, backend, present, mlx_caps, llama_caps) -> ValueError:
    stacks = [name for name, caps in (("mlx", mlx_caps), ("llama", llama_caps)) if caps]
    wanted = f"--device {device}" + (f" --backend {backend}" if backend else "")
    return ValueError(
        f"No backend can serve {wanted} on this machine "
        f"(accelerators: {', '.join(present) or 'none'}; "
        f"installed stacks: {', '.join(stacks) or 'none'}). Install the MLX extra "
        "(uv sync --extra mlx|cuda|cpu) or build llama.cpp and set RIZZO_LLAMA_LIB."
    )


def _cpu_stack(backend, mlx_caps, llama_caps) -> str | None:
    if backend:
        caps = mlx_caps if backend == "mlx" else llama_caps
        return backend if "cpu" in caps else None
    caps = {"mlx": mlx_caps, "llama": llama_caps}
    return next((name for name in CPU_STACK_PREFERENCE if "cpu" in caps[name]), None)


def route(
    device: str = "auto",
    *,
    backend: str | None = None,
    gguf=None,
    hardware_present: list[str] | None = None,
    mlx_caps: dict[str, str] | None = None,
    llama_caps: dict[str, str] | None = None,
) -> Routing:
    """Resolve a user preference into (backend, device), or explain why it cannot.

    `auto` walks the accelerators present and returns the first stack that can drive one. It
    never silently drops to CPU: that outcome carries `downgraded=True` and a reason, and
    `announce` turns it into a warning. An explicit `--device <vendor>` or `--backend` that
    cannot be honoured raises instead, so a typo never looks like a slow success.
    """
    original = device
    device = ACCELERATOR_ALIASES.get(device, device)
    if device not in DEVICES and device not in STACK_DEVICES:
        raise ValueError(f"Device must be one of: {', '.join(DEVICE_CHOICES)}")
    if backend == "auto":
        backend = None
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"Backend must be one of: {', '.join(BACKENDS)}")

    pinned = STACK_DEVICES.get(device)
    if pinned and backend is not None and backend != pinned:
        raise ValueError(
            f"--device {original} selects the {pinned} backend, but --backend {backend} was "
            "also given; drop one of them"
        )
    if pinned:
        backend, device = pinned, "auto"  # the stack is pinned; the accelerator is not
    if gguf:
        backend = "llama"  # a GGUF is a llama.cpp artifact

    present = hardware.probe() if hardware_present is None else list(hardware_present)
    mlx_caps = mlx_devices() if mlx_caps is None else mlx_caps
    llama_caps = llama_devices() if llama_caps is None else llama_caps

    if backend is not None and not (mlx_caps if backend == "mlx" else llama_caps):
        # The pinned stack is not installed, so there is nothing to inspect. Hand over to the
        # loader: it knows the library path and prints a far better error than the router can.
        return Routing(backend, "auto", f"--backend {backend}: availability checked at load")

    if device == "cpu":
        chosen = _cpu_stack(backend, mlx_caps, llama_caps)
        if chosen is None:
            raise _unavailable(device, backend, present, mlx_caps, llama_caps)
        return Routing(chosen, "cpu", f"--device cpu via {chosen}")

    if device not in ("auto", "gpu") and device not in present:
        raise ValueError(
            f"--device {device}: no {device} accelerator on this machine "
            f"(found: {', '.join(present) or 'none'})"
        )

    table = candidate_stacks(present, mlx_caps, llama_caps)
    if backend is not None:
        table = [row for row in table if row[1] == backend]
    if device not in ("auto", "gpu"):
        table = [row for row in table if row[0] == device]
    if table:
        accelerator, chosen, stack_device = table[0]
        return Routing(chosen, stack_device, f"{accelerator} via {chosen}")

    if device not in ("auto", "cpu"):
        # An explicit accelerator was asked for (a vendor name or `gpu`): a silent CPU
        # downgrade would look like a slow success, so refuse instead.
        raise _unavailable(device, backend, present, mlx_caps, llama_caps)

    # `auto`, or a stack pinned without a device: fall back to CPU for that stack, visibly.
    chosen = _cpu_stack(backend, mlx_caps, llama_caps)
    if chosen is None:
        raise _unavailable(device, backend, present, mlx_caps, llama_caps)
    return Routing(
        chosen,
        "cpu",
        f"no usable GPU among {', '.join(present)}; using {chosen} on CPU",
        downgraded=True,
    )


def announce(routing: Routing) -> None:
    """Surface a CPU downgrade where the user can see it. `route` itself stays pure."""
    if routing.downgraded:
        warnings.warn(routing.reason, RuntimeWarning, stacklevel=2)


def ensure_options(backend: str, *, bits) -> None:
    """Reject combinations that would otherwise fail deep inside a loader."""
    if backend == "llama" and bits is not None:
        raise ValueError(
            "--bits quantizes MLX weights in memory; with llama.cpp choose a quantized GGUF "
            "instead (--gguf PATH or --quant " + "|".join(GGUF_MODELS) + ")"
        )


def _reject_quant_for_other_stacks(backend: str | None, quant: str) -> None:
    if backend != "llama" and quant != DEFAULT_QUANT:
        raise ValueError("--quant applies to the llama backend only")


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
    _reject_quant_for_other_stacks(backend, quant)
    routing = route(device, backend=backend, gguf=gguf)
    announce(routing)
    _reject_quant_for_other_stacks(routing.backend, quant)
    ensure_options(routing.backend, bits=bits)
    if routing.backend == "llama":
        from .backend_llama import LlamaBackend

        if gguf:
            path, expected = Path(gguf), None
        elif size in MODELS:
            path, expected = gguf_path(quant, size), GGUF_MODELS[quant].sha256
        else:
            raise ValueError(f"Unknown size {size}; available: {', '.join(MODELS)}")
        return LlamaBackend.load(
            path,
            device=routing.device,
            ctx=ctx,
            batch_size=batch_size,
            prefill_chunk=prefill_chunk,
            threads=threads,
            expected_sha256=expected,
        )
    from .backend import SparkBackend

    return SparkBackend.load(
        model or MODELS[size].path,
        bits=bits,
        device=routing.device,
        batch_size=batch_size,
        prefill_chunk=prefill_chunk,
    )


def describe() -> dict:
    """What `rizzo devices` prints: the hardware, both stacks, and what `auto` would pick."""
    from .backend_llama import library_dir
    from .llama_runtime import detect_backends

    report = {
        "hardware": hardware.probe(),
        "available": available_backends(),
        "llama": {
            "library": str(library_dir()),
            "present": llama_present(),
            "devices": list(LLAMA_DEVICES),
            "ggml_backends": detect_backends(library_dir()),
        },
    }
    if mlx_importable():
        from .runtime import describe as describe_mlx

        report["mlx"] = describe_mlx()
    else:
        report["mlx"] = {"available": False}
    try:
        chosen = route("auto")
        report["auto"] = {
            "backend": chosen.backend,
            "device": chosen.device,
            "reason": chosen.reason,
            "downgraded": chosen.downgraded,
        }
    except ValueError as error:
        report["auto"] = {"error": str(error)}
    return report
