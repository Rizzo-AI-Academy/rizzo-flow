"""Backend identity and fingerprint as records instead of hand-built dicts.

The fingerprint binds a calibration file to the exact weights, runtime and prompt that
produced it. It is a sha256 over the fields marked with `fingerprinted()`, so **renaming
one of those fields, or adding or dropping one, invalidates every calibration file on
disk**. `tests/test_formats.py` pins that promise to fingerprints recorded in `results/`.

The two backends do not report the same things, so they get one flat record each rather
than a shared shape with holes in it: MLX names its runtime and quantization group size,
llama.cpp names its GGUF source, release and commit.
"""

import hashlib
from dataclasses import dataclass, field, fields
from typing import Any

from .prompts import canonical


def fingerprinted(**kwargs: Any) -> Any:
    """Mark a field as covered by the fingerprint. Order of declaration is the JSON order."""
    return field(metadata={"fingerprint": True}, **kwargs)


@dataclass(frozen=True)
class Metadata:
    """Shared behaviour only: no fields here, so each backend keeps its own key order."""

    def identity(self) -> dict:
        """Exactly what the fingerprint is computed over, `None` values included."""
        return {
            f.name: getattr(self, f.name) for f in fields(self) if f.metadata.get("fingerprint")
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical(self.identity()).encode()).hexdigest()

    def as_dict(self) -> dict:
        """The public `model` block: identity, then the fingerprint, then the rest."""
        identity = self.identity()
        extra = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in identity}
        return {**identity, "fingerprint": self.fingerprint, **extra}


@dataclass(frozen=True)
class MlxMetadata(Metadata):
    source: str = fingerprinted()
    requested_revision: str = fingerprinted()
    runtime_revision: str = fingerprinted()
    source_files: dict[str, str] = fingerprinted()
    precision: str = fingerprinted()
    quantization_group_size: int | None = fingerprinted()
    device: str = fingerprinted()
    backend: str = fingerprinted()
    mlx: str = fingerprinted()
    mlx_lm: str = fingerprinted()
    prompt_version: str = fingerprinted()
    load_seconds: float = 0.0


@dataclass(frozen=True)
class LlamaMetadata(Metadata):
    source: str = fingerprinted()
    requested_revision: str = fingerprinted()
    gguf_source: str | None = fingerprinted()
    gguf_revision: str | None = fingerprinted()
    source_files: dict[str, str] = fingerprinted()
    precision: str = fingerprinted()
    device: str = fingerprinted()
    backend: str = fingerprinted()
    runtime: str = fingerprinted()
    llama_cpp_release: str = fingerprinted()
    llama_cpp_commit: str = fingerprinted()
    prompt_version: str = fingerprinted()
    device_name: str | None = None
    context_cells: int = 0
    load_seconds: float = 0.0
