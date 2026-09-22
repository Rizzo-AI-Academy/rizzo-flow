"""llama.cpp scoring: no generation, flat unpadded batches, prefix branches that share KV cells.

Same contract as `backend.SparkBackend` (tokenizer, `metadata`, `score`), so `Engine`, the
prompt and the decision layer do not know which runtime is underneath. llama.cpp drives Metal,
CUDA, Vulkan (AMD, Intel, NVIDIA), ROCm, SYCL and plain CPUs from the same GGUF file.
"""

import hashlib
import time
from pathlib import Path

from . import llama_release
from .config import GGUF, identify
from .llama_cpp import Session
from .prompts import PROMPT_VERSION, Compiled, canonical

N_BATCH = 2048  # most tokens handed to one llama_decode call
# general.file_type of the quantizations pinned in config.GGUF (enum llama_ftype)
ARCHITECTURE = "spark2_5"
FILE_TYPES = {"1": "f16", "7": "q8_0", "15": "q4_k_m", "32": "bf16"}


class LlamaTokenizer:
    """The two tokenizer calls `prompts.compile_request` makes, served by the GGUF itself:
    its chat template rendered the way transformers renders it, its vocabulary for encoding."""

    def __init__(self, session, template: str):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message):
            raise ValueError(message)

        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.globals["raise_exception"] = raise_exception
        self.template = environment.from_string(template)
        self.session = session
        self.pad_token_id = session.pad_token
        self.eos_token_id = session.eos_token

    def apply_chat_template(self, messages, tokenize=False, **variables) -> str:
        if tokenize:
            raise ValueError("Render the text, then call encode()")
        return self.template.render(messages=messages, **variables)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.session.tokenize(text, add_special_tokens)


class LlamaBackend:
    def __init__(self, session, tokenizer, metadata, batch_size=4, prefill_chunk=512):
        if not 1 <= batch_size <= 16 or not 1 <= prefill_chunk <= 2048:
            raise ValueError("batch_size must be 1–16 and prefill_chunk 1–2048")
        self.session = session
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.batch_size = batch_size
        self.prefill_chunk = prefill_chunk
        self._lowest_free = None

    @classmethod
    def load(
        cls,
        path,
        device="auto",
        ctx=8192,
        batch_size=4,
        prefill_chunk=512,
        threads=None,
        runtime_dir=None,
    ):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"GGUF file not found at {path}. Run `rizzo download` first.")
        if ctx < 1:
            raise ValueError("ctx must be positive")
        started = time.perf_counter()
        # Hash the weights once at startup for auditability and calibration binding.
        with path.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        pin = next((spec for spec in GGUF.values() if spec.sha256 == sha256), None)
        # The longest question is `ctx` tokens; a microbatch adds at most N_BATCH suffix tokens
        # on top of the prefix they share, so this many cells always suffice.
        session = Session.load(
            path,
            directory=runtime_dir or llama_release.locate(device),
            device=device,
            n_ctx=ctx + N_BATCH,
            n_batch=N_BATCH,
            n_ubatch=prefill_chunk,
            n_seq_max=batch_size + 1,
            threads=threads,
        )
        try:
            architecture = session.meta("general.architecture")
            if architecture != ARCHITECTURE:
                raise ValueError(f"Only the Spark2.5 architecture is supported, not {architecture}")
            spec = identify(int(session.meta(f"{ARCHITECTURE}.embedding_length") or 0))
            template = session.chat_template()
            if not template:
                raise ValueError("The GGUF file carries no chat template")
            tokenizer = LlamaTokenizer(session, template)
        except Exception:
            session.close()
            raise
        file_type = session.meta("general.file_type")
        chosen = session.device
        identity = {
            "source": spec.repo,
            "requested_revision": spec.revision,
            "gguf_source": pin.repo if pin else None,  # None: not one of the pinned files
            "gguf_revision": pin.revision if pin else None,
            "source_files": {path.name: sha256},
            "precision": pin.quant if pin else FILE_TYPES.get(file_type, f"ftype{file_type}"),
            "device": "gpu" if chosen else "cpu",
            "backend": chosen.backend.lower() if chosen else "cpu",
            "runtime": "llama.cpp",
            "llama_cpp_release": llama_release.RELEASE,
            "llama_cpp_commit": llama_release.COMMIT,
            "prompt_version": PROMPT_VERSION,
        }
        metadata = {
            **identity,
            "fingerprint": hashlib.sha256(canonical(identity).encode()).hexdigest(),
            "device_name": chosen.description if chosen else None,
            "context_cells": session.n_ctx,
            "load_seconds": time.perf_counter() - started,
        }
        return cls(session, tokenizer, metadata, batch_size, prefill_chunk)

    def _feed(self, tokens, start, sequence, want_logits) -> int | None:
        """Run `tokens` on one sequence, N_BATCH per call; index of the final logits row.
        llama.cpp splits each call into `prefill_chunk` micro-batches on its own."""
        last = None
        for offset in range(0, len(tokens), N_BATCH):
            piece = tokens[offset : offset + N_BATCH]
            final = want_logits and offset + len(piece) == len(tokens)
            self.session.decode(
                piece,
                range(start + offset, start + offset + len(piece)),
                [sequence] * len(piece),
                [len(piece) - 1] if final else (),
            )
            last = len(piece) - 1 if final else None
        return last

    def _groups(self, jobs, prefix_length):
        """Microbatches of up to `batch_size` suffixes that fit one llama_decode call."""
        group, used = [], 0
        for job in jobs:
            size = len(job.tokens) - prefix_length
            if group and (len(group) == self.batch_size or used + size > N_BATCH):
                yield group
                group, used = [], 0
            group.append(job)
            used += size
        if group:
            yield group

    def _track_memory(self):
        free = self.session.free_bytes()
        if free is not None:
            self._lowest_free = free if self._lowest_free is None else min(self._lowest_free, free)

    def peak_device_bytes(self) -> int | None:
        """Largest drop in free device memory since before the load. Other processes using
        the same GPU are counted too; None on the CPU."""
        if self._lowest_free is None:
            return None
        return max(self.session.idle_free - self._lowest_free, 0)

    def score(self, prefix: list[int], jobs: list[Compiled], mode="shared"):
        if mode not in ("shared", "direct"):
            raise ValueError("Unknown execution mode")
        if not jobs:
            raise ValueError("No decisions supplied")
        if any(
            job.tokens[: len(prefix)] != prefix or len(job.tokens) <= len(prefix) for job in jobs
        ):
            raise ValueError("Invalid shared prefix")
        session = self.session
        started = time.perf_counter()
        result = {}
        prefix_seconds = 0.0
        evaluated_tokens = 0
        batches = 0
        # A lone question has nobody to share the prefix with: one pass is the same computation
        # as `direct` and saves a call, which is a third of the latency of a short request.
        reuse = mode == "shared" and bool(prefix) and len(jobs) > 1
        if not reuse:
            for job in jobs:
                session.clear()
                row = self._feed(job.tokens, 0, 0, True)
                result[job.id] = session.logits(row, job.slots)
                evaluated_tokens += len(job.tokens)
                batches += 1
        else:
            mark = time.perf_counter()
            session.clear()
            self._feed(prefix, 0, 0, False)
            session.synchronize()
            prefix_seconds = time.perf_counter() - mark
            evaluated_tokens += len(prefix)
            start = len(prefix)
            for group in self._groups(sorted(jobs, key=lambda j: len(j.tokens)), start):
                # Branches reference the prefix cells of sequence 0; nothing is copied, and
                # dropping a branch leaves the prefix untouched for the next microbatch.
                for sequence in range(1, len(group) + 1):
                    session.branch(0, sequence)
                suffixes = [job.tokens[start:] for job in group]
                if len(group) == 1:
                    rows = [self._feed(suffixes[0], start, 1, True)]  # any length, in chunks
                else:
                    # Suffixes lie end to end: no padding, each reads its own last position.
                    tokens, positions, sequences, rows = [], [], [], []
                    for sequence, suffix in enumerate(suffixes, start=1):
                        tokens += suffix
                        positions += range(start, start + len(suffix))
                        sequences += [sequence] * len(suffix)
                        rows.append(len(tokens) - 1)
                    session.decode(tokens, positions, sequences, rows)
                for job, row in zip(group, rows, strict=True):
                    result[job.id] = session.logits(row, job.slots)
                for sequence in range(1, len(group) + 1):
                    session.drop(sequence)
                evaluated_tokens += sum(map(len, suffixes))
                batches += 1
        session.synchronize()
        self._track_memory()
        timing = {
            "inference_seconds": time.perf_counter() - started,
            "prefill_seconds": prefix_seconds,
            "shared_prefix_tokens": len(prefix) if reuse else 0,
            "evaluated_tokens_including_padding": evaluated_tokens,  # llama.cpp never pads
            "logical_input_tokens": sum(len(j.tokens) for j in jobs),
            "batches": batches,
            "generated_tokens": 0,
        }
        if self._lowest_free is not None:
            timing["peak_device_bytes"] = self.peak_device_bytes()
        return result, timing
