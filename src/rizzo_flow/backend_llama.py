"""llama.cpp backend: the same contract as `SparkBackend`, over a pinned GGUF checkpoint.

The decision layer above (`Engine`, `prompts`, `decisions`) is untouched: this class only
provides a tokenizer, an auditable `metadata` dictionary and `score()`. Both provenance
layers are recorded, because the artifact is a third-party conversion of the pinned original
weights: the model identity (`source`, `requested_revision`) comes from `config.MODELS`, the
artifact (`gguf_source`, `gguf_revision`, `gguf_sha256`) from `config.GGUF_MODELS`.
"""

import hashlib
import os
import time
from pathlib import Path

from .config import find_gguf_pin, identify
from .llama_runtime import LLAMA_COMMIT, LLAMA_VERSION, LlamaRuntime
from .llama_tokenizer import LlamaTokenizer
from .prompts import PROMPT_VERSION, canonical

LIBRARY_ENV = "RIZZO_LLAMA_LIB"
# Matches the build directory documented in SPEC.md; override with RIZZO_LLAMA_LIB.
DEFAULT_LIBRARY = Path("llama.cpp-amd/build/bin")


def library_dir() -> Path:
    return Path(os.environ.get(LIBRARY_ENV, DEFAULT_LIBRARY))


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class LlamaBackend:
    def __init__(self, runtime, tokenizer, metadata, batch_size=4, prefill_chunk=512):
        if not 1 <= batch_size <= 16 or not 1 <= prefill_chunk <= 2048:
            raise ValueError("batch_size must be 1–16 and prefill_chunk 1–2048")
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.batch_size = batch_size
        self.prefill_chunk = prefill_chunk

    @classmethod
    def from_runtime(
        cls,
        runtime,
        *,
        gguf,
        gguf_sha256,
        batch_size=4,
        prefill_chunk=512,
        load_seconds=None,
    ) -> "LlamaBackend":
        architecture = runtime.architecture
        if architecture != "spark2_5":
            raise ValueError(f"Only the Spark2.5 architecture is supported (found {architecture})")
        raw = runtime.metadata()
        try:
            hidden_size = int(raw["spark2_5.embedding_length"])
        except (KeyError, ValueError) as error:
            raise ValueError("GGUF does not expose spark2_5.embedding_length") from error
        spec = identify({"hidden_size": hidden_size})
        pin = find_gguf_pin(Path(gguf).name)
        identity = {
            "backend": "llama",
            "compute_backend": runtime.compute_backend,
            "device": runtime.device,
            "source": spec.repo,
            "requested_revision": spec.revision,
            "architecture": architecture,
            "context_length": int(raw["spark2_5.context_length"])
            if raw.get("spark2_5.context_length")
            else None,
            "precision": runtime.file_type_name.lower(),
            "gguf_file": Path(gguf).name,
            "gguf_sha256": gguf_sha256,
            "gguf_source": pin.repo if pin else None,
            "gguf_revision": pin.revision if pin else None,
            "llama_cpp_version": LLAMA_VERSION,
            "llama_cpp_commit": LLAMA_COMMIT,
            "prompt_version": PROMPT_VERSION,
        }
        metadata = {
            **identity,
            "fingerprint": hashlib.sha256(canonical(identity).encode()).hexdigest(),
        }
        if load_seconds is not None:
            metadata["load_seconds"] = load_seconds
        return cls(
            runtime, LlamaTokenizer.from_runtime(runtime), metadata, batch_size, prefill_chunk
        )

    @classmethod
    def load(
        cls,
        gguf,
        *,
        library=None,
        device="auto",
        ctx=8192,
        batch_size=4,
        prefill_chunk=512,
        threads=None,
        expected_sha256=None,
    ) -> "LlamaBackend":
        if ctx < 1:
            raise ValueError("ctx must be positive")
        gguf = Path(gguf).resolve()
        if not gguf.is_file():
            raise ValueError(f"GGUF file not found: {gguf} (run `rizzo download --format gguf`)")
        started = time.perf_counter()
        digest = sha256_file(gguf)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(
                f"GGUF sha256 mismatch for {gguf.name}: expected {expected_sha256}, got {digest}"
            )
        runtime = LlamaRuntime.load(
            library or library_dir(),
            gguf,
            n_ctx=ctx,
            n_batch=ctx,
            n_ubatch=min(prefill_chunk, ctx),
            n_seq_max=batch_size + 1,
            device=device,
            threads=threads,
        )
        return cls.from_runtime(
            runtime,
            gguf=gguf,
            gguf_sha256=digest,
            batch_size=batch_size,
            prefill_chunk=prefill_chunk,
            load_seconds=time.perf_counter() - started,
        )

    def close(self) -> None:
        close = getattr(self.runtime, "close", None)
        if close:
            close()
