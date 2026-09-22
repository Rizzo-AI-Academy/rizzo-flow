"""Pick and prepare the MLX compute backend: Metal (Apple), CUDA (NVIDIA) or CPU."""

import os
import sys
import sysconfig
import types
from dataclasses import asdict, dataclass
from pathlib import Path

# User-facing names. `gpu` means whichever accelerator this install has.
DEVICES = ("auto", "gpu", "mlx", "cuda", "cpu")
INSTALL_HINT = (
    "Install exactly one runtime: `uv sync --extra mlx` (Apple Silicon), "
    "`uv sync --extra cuda` (NVIDIA GPU) or `uv sync --extra cpu` (no GPU)."
)


def prepare():
    """Make the MLX stack importable on Windows; does nothing elsewhere. Safe to call twice."""
    if sys.platform != "win32":
        return
    # mlx-lm raises RLIMIT_NOFILE at import time through the Unix-only `resource` module.
    if "resource" not in sys.modules:
        stub = types.ModuleType("resource")
        stub.RLIMIT_NOFILE = 0
        stub.setrlimit = lambda *args: None
        sys.modules["resource"] = stub
    # mlx-cuda loads the CUDA DLLs of the nvidia-* wheels lazily; they are not on PATH.
    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    folders = [nvidia / "cu13" / "bin" / "x86_64", nvidia / "cudnn" / "bin"]
    found = [str(f) for f in folders if f.is_dir() and str(f) not in os.environ.get("PATH", "")]
    if found:
        os.environ["PATH"] = os.pathsep.join([*found, os.environ.get("PATH", "")])


def import_mlx():
    prepare()
    try:
        import mlx.core as mx
    except ImportError as error:
        raise ImportError(f"MLX has no compute backend here ({error}). {INSTALL_HINT}") from error
    return mx


def accelerator(mx) -> str | None:
    """Name of the GPU backend of this install, if it has a usable one."""
    if not mx.is_available(mx.gpu):
        return None
    return "mlx" if sys.platform == "darwin" else "cuda"


def resolve(device: str):
    """Map a user-facing device name to (mlx device, backend name: mlx | cuda | cpu)."""
    if device not in DEVICES:
        raise ValueError(f"Device must be one of: {', '.join(DEVICES)}")
    mx = import_mlx()
    gpu = accelerator(mx)
    if device == "cpu" or (device == "auto" and gpu is None):
        return mx.cpu, "cpu"
    if gpu is None:
        raise ValueError(f"--device {device}: this install has no usable GPU. {INSTALL_HINT}")
    if device in ("mlx", "cuda") and device != gpu:
        raise ValueError(f"--device {device}: the GPU backend of this install is `{gpu}`.")
    return mx.gpu, gpu


@dataclass(frozen=True)
class MlxReport:
    """What `rizzo devices` prints about MLX, when MLX is installed at all."""

    mlx_version: str
    platform: str
    available: list[str]
    auto_selects: str

    def as_dict(self) -> dict:
        return asdict(self)


def describe() -> MlxReport:
    """The compute backends this install can actually use."""
    mx = import_mlx()
    gpu = accelerator(mx)
    return MlxReport(
        mlx_version=mx.__version__,
        platform=sys.platform,
        available=[name for name in (gpu, "cpu") if name],
        auto_selects=gpu or "cpu",
    )
