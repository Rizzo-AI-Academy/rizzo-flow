"""Which accelerator this machine actually has.

Both compute stacks probe their own install — can `mlx` be imported, is a `libggml-*`
sitting next to `libllama` — and neither answers the question a router needs: which GPU
is physically present. This module answers that one, so `--device auto` stops inferring
the vendor from whichever artifacts happen to be on disk.

Detection is best-effort and never raises: an unknown machine reports `cpu`, which is
always the last (and safe) fallback.
"""

import platform
import shutil
import sys
from pathlib import Path

# Best first. The router walks this order, so Apple Silicon wins on a Mac and CPU is the floor.
ACCELERATORS = ("apple", "nvidia", "amd", "intel", "cpu")
# PCI vendor ids exactly as the Linux DRM tree spells them.
PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}
DRM_ROOT = Path("/sys/class/drm")
# Device nodes the proprietary driver creates; present even when nvidia-smi is not on PATH.
NVIDIA_NODES = (Path("/dev/nvidiactl"), Path("/dev/nvidia0"))


def drm_vendors(drm: Path = DRM_ROOT) -> set[str]:
    """Accelerator names readable from a Linux DRM tree; empty on any other OS."""
    found = set()
    for vendor in Path(drm).glob("card[0-9]*/device/vendor"):
        try:
            value = vendor.read_text().strip().lower()
        except OSError:
            continue
        if value in PCI_VENDORS:
            found.add(PCI_VENDORS[value])
    return found


def apple_platform() -> bool:
    """True only for Apple Silicon, where MLX has a Metal backend."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def nvidia_driver() -> bool:
    """An NVIDIA driver is loaded. A proxy: the nodes exist or nvidia-smi is installed."""
    return any(node.exists() for node in NVIDIA_NODES) or shutil.which("nvidia-smi") is not None


def probe() -> list[str]:
    """Accelerators present on this machine, best first, always ending with `cpu`.

    macOS (non-Apple-Silicon) is not covered: the DRM tree is Linux-only and there is no
    portable vendor probe, so such a machine reports `cpu`. MLX is Apple-Silicon-only and
    Vulkan needs MoltenVK there, so this is a limitation, not a mis-detection.
    """
    present = drm_vendors()
    if apple_platform():
        # Apple Silicon has no discrete NVIDIA/AMD card to probe for.
        present.add("apple")
    elif nvidia_driver():
        # `drm_vendors` may also have found an AMD card: hybrid machines have both.
        present.add("nvidia")
    present.add("cpu")
    return [name for name in ACCELERATORS if name in present]
