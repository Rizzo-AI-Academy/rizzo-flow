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

from .config import find_gguf_pin, identify, sha256_file
from .llama_runtime import LLAMA_COMMIT, LLAMA_VERSION, LlamaRuntime, build_batch_spec
from .llama_tokenizer import LlamaTokenizer
from .prompts import PROMPT_VERSION, Compiled, canonical

LIBRARY_ENV = "RIZZO_LLAMA_LIB"
# Matches the build directory documented in SPEC.md; override with RIZZO_LLAMA_LIB.
DEFAULT_LIBRARY = Path("llama.cpp-amd/build/bin")


def library_dir() -> Path:
    return Path(os.environ.get(LIBRARY_ENV, DEFAULT_LIBRARY))


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
        sequence_limit = getattr(runtime, "n_seq_max", None)
        if sequence_limit is not None and sequence_limit < batch_size + 1:
            raise ValueError(
                f"Runtime allows {sequence_limit} sequences but batch_size={batch_size} needs "
                f"{batch_size + 1}; llama_memory_seq_cp would abort the process"
            )
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
        # Branching duplicates the KV cache, so llama.cpp divides n_ctx across sequences:
        # request one sequence's worth per branch and the per-sequence context stays >= ctx.
        sequences = batch_size + 1
        runtime = LlamaRuntime.load(
            library or library_dir(),
            gguf,
            n_ctx=ctx * sequences,
            n_batch=ctx * sequences,
            n_ubatch=min(prefill_chunk, ctx),
            n_seq_max=sequences,
            device=device,
            threads=threads,
        )
        if runtime.n_ctx_seq < ctx:
            runtime.close()
            raise ValueError(
                f"llama.cpp granted {runtime.n_ctx_seq} tokens per sequence, below the requested "
                f"--ctx {ctx}; lower --batch-size or raise --ctx"
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

    # --- scoring --------------------------------------------------------------------------

    def _decode_tokens(self, tokens: list[int], seq_id: int = 0) -> int:
        """Prefill a run of tokens in chunks, without materialising their logits."""
        chunks = 0
        for offset in range(0, len(tokens), self.prefill_chunk):
            chunk = tokens[offset : offset + self.prefill_chunk]
            self.runtime.decode(build_batch_spec(offset, [chunk], [seq_id], produce_logits=False))
            chunks += 1
        return chunks

    def _groups(self, jobs, prefix_length: int):
        """Chunk jobs into microbatch-sized groups that also fit the context."""
        capacity = getattr(self.runtime, "n_ctx", 0) or 1 << 30
        groups, current, total = [], [], 0
        for job in jobs:
            length = len(job.tokens) - prefix_length
            if current and (len(current) >= self.batch_size or total + length > capacity):
                groups.append(current)
                current, total = [], 0
            current.append(job)
            total += length
        if current:
            groups.append(current)
        return groups

    def score(self, prefix: list[int], jobs: list[Compiled], mode="shared"):
        if mode not in ("shared", "direct"):
            raise ValueError("Unknown execution mode")
        if not jobs:
            raise ValueError("No decisions supplied")
        if any(
            job.tokens[: len(prefix)] != prefix or len(job.tokens) <= len(prefix) for job in jobs
        ):
            raise ValueError("Invalid shared prefix")
        started = time.perf_counter()
        result = {}
        prefix_seconds = 0.0
        evaluated_tokens = 0
        batches = 0
        if mode == "direct" or not prefix:
            for job in jobs:
                # No reuse: every job starts from an empty cache, one forward pass for its last token.
                self.runtime.memory_clear()
                chunks = self._decode_tokens(job.tokens[:-1])
                spec = build_batch_spec(len(job.tokens) - 1, [job.tokens[-1:]], [0])
                self.runtime.decode(spec)
                result[job.id] = self.runtime.logits(spec.finals[0], job.slots)
                evaluated_tokens += len(job.tokens)
                batches += chunks + 1
        else:
            mark = time.perf_counter()
            self.runtime.memory_clear()
            batches += self._decode_tokens(prefix)
            prefix_seconds = time.perf_counter() - mark
            evaluated_tokens += len(prefix)
            ordered = sorted(jobs, key=lambda job: len(job.tokens))
            for group in self._groups(ordered, len(prefix)):
                suffixes = [job.tokens[len(prefix) :] for job in group]
                seq_ids = list(range(1, len(group) + 1))
                # Branch the retained prefix once per job, score all suffixes in one decode.
                for seq_id in seq_ids:
                    self.runtime.seq_copy(0, seq_id)
                spec = build_batch_spec(len(prefix), suffixes, seq_ids)
                self.runtime.decode(spec)
                for job, final in zip(group, spec.finals, strict=True):
                    result[job.id] = self.runtime.logits(final, job.slots)
                for seq_id in seq_ids:
                    self.runtime.seq_remove(seq_id)
                evaluated_tokens += len(spec.tokens)
                batches += 1
        return result, {
            "inference_seconds": time.perf_counter() - started,
            "prefill_seconds": prefix_seconds,
            "shared_prefix_tokens": len(prefix) if mode == "shared" else 0,
            "evaluated_tokens_including_padding": evaluated_tokens,
            "logical_input_tokens": sum(len(job.tokens) for job in jobs),
            "batches": batches,
            "generated_tokens": 0,
            "context_tokens": getattr(self.runtime, "n_ctx", 0),
            "context_tokens_per_sequence": getattr(self.runtime, "n_ctx_seq", 0),
        }
