"""Spark-native MLX scoring: no generation, selected output rows, shared prefix branches."""

import copy
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from .config import RUNTIME_REVISION, identify
from .metadata import MlxMetadata
from .prompts import PROMPT_VERSION, Compiled
from .protocols import Tokenizer
from .responses import Timing
from .runtime import resolve


def quantize_model(model, bits):
    from mlx import nn

    native_predicate = model.quant_predicate
    nn.quantize(
        model,
        group_size=64,
        bits=bits,
        class_predicate=lambda path, module: (
            hasattr(module, "to_quantized") and native_predicate(path, module)
        ),
    )


def selected_logits(model, hidden, slots):
    """Project only declared answer tokens; works with BF16 and affine quantization."""
    import mlx.core as mx

    head = model.model.embedding if model.args.tie_word_embeddings else model.lm_head
    indices = mx.array(slots)
    weights = head.weight[indices]
    if hasattr(head, "scales"):
        biases = head.biases[indices] if "biases" in head else None
        output = mx.quantized_matmul(
            hidden,
            weights,
            head.scales[indices],
            biases,
            transpose=True,
            group_size=head.group_size,
            bits=head.bits,
            mode=head.mode,
        )
    else:
        output = hidden @ weights.T
    if "bias" in head:
        output = output + head.bias[indices]
    return output.astype(mx.float32)


def branch_cache(prefix_cache, batch_size):
    """Copy native cache objects and arrays, including rotating offsets; never mutate the prefix."""
    import mlx.core as mx

    result = []
    for original in prefix_cache:
        branch = copy.copy(original)
        branch.state = tuple(mx.repeat(a, batch_size, axis=0) for a in original.state)
        result.append(branch)
    return result


class SparkBackend:
    def __init__(
        self,
        model,
        tokenizer: Tokenizer,
        metadata: MlxMetadata,
        batch_size: int = 4,
        prefill_chunk: int = 512,
    ) -> None:
        if not 1 <= batch_size <= 16 or not 1 <= prefill_chunk <= 2048:
            raise ValueError("batch_size must be 1–16 and prefill_chunk 1–2048")
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.batch_size = batch_size
        self.prefill_chunk = prefill_chunk

    @classmethod
    def load(cls, path, bits=None, device="auto", batch_size=4, prefill_chunk=512):
        path = Path(path).resolve()
        if not path.is_dir():
            raise ValueError(f"Model not found at {path}. Run `rizzo download` first.")
        if bits not in (None, 4, 8):
            raise ValueError("Supported precisions: BF16, 8-bit, 4-bit")
        target, backend = resolve(device)
        import mlx.core as mx
        from spark_mlx_llm import load

        mx.set_default_device(target)
        mx.set_cache_limit(256 * 1024**2)
        started = time.perf_counter()
        # Hash checkpoint contents once at startup for auditability and calibration binding.
        files = sorted(path.glob("*.safetensors")) + [
            path / "config.json",
            path / "tokenizer.json",
            path / "tokenizer_config.json",
        ]
        files += list(path.glob("*.jinja"))
        hashes = {}
        for file in files:
            with file.open("rb") as stream:
                hashes[file.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if not any(name.endswith(".safetensors") for name in hashes):
            raise ValueError("Model directory contains no safetensors weights")
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "spark2_5":
            raise ValueError("Only the Spark2.5 architecture is supported")
        spec = identify(config.get("hidden_size"))
        model, tokenizer = load(
            path,
            lazy=True,
            strict=True,
            dtype="bfloat16",
            tokenizer_config={"trust_remote_code": False},
        )
        if bits:
            quantize_model(model, bits)
        model.eval()
        mx.eval(model.parameters())
        mx.synchronize()
        metadata = MlxMetadata(
            source=spec.repo,
            requested_revision=spec.revision,
            runtime_revision=RUNTIME_REVISION,
            source_files=hashes,
            precision=f"q{bits}" if bits else "bf16",
            quantization_group_size=64 if bits else None,
            device="cpu" if backend == "cpu" else "gpu",
            backend=backend,
            mlx=mx.__version__,
            mlx_lm=importlib.metadata.version("mlx-lm"),
            prompt_version=PROMPT_VERSION,
            load_seconds=time.perf_counter() - started,
        )
        return cls(model, tokenizer, metadata, batch_size, prefill_chunk)

    def _prefill(self, tokens):
        import mlx.core as mx

        cache = self.model.make_cache()
        for offset in range(0, len(tokens), self.prefill_chunk):
            self.model.model(mx.array([tokens[offset : offset + self.prefill_chunk]]), cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    def score(
        self, prefix: list[int], jobs: list[Compiled], mode: str = "shared"
    ) -> tuple[dict[str, list[float]], Timing]:
        import mlx.core as mx

        if mode not in ("shared", "direct"):
            raise ValueError("Unknown execution mode")
        if not jobs:
            raise ValueError("No decisions supplied")
        if any(
            job.tokens[: len(prefix)] != prefix or len(job.tokens) <= len(prefix) for job in jobs
        ):
            raise ValueError("Invalid shared prefix")
        started = time.perf_counter()
        mx.reset_peak_memory()
        result = {}
        prefix_seconds = 0.0
        evaluated_tokens = 0
        batches = 0
        if mode == "direct" or not prefix:
            for job in jobs:
                # Chunk prefill bounds attention intermediates; project only the last position.
                cache = self._prefill(job.tokens[:-1])
                hidden = self.model.model(mx.array([job.tokens[-1:]]), cache=cache)[:, -1, :]
                logits = selected_logits(self.model, hidden, job.slots)[0]
                mx.eval(logits)
                result[job.id] = logits.tolist()
                evaluated_tokens += len(job.tokens)
                batches += 1
                del cache, hidden, logits
        else:
            mark = time.perf_counter()
            cache = self._prefill(prefix)
            mx.synchronize()
            prefix_seconds = time.perf_counter() - mark
            evaluated_tokens += len(prefix)
            ordered = sorted(jobs, key=lambda j: len(j.tokens))
            for offset in range(0, len(ordered), self.batch_size):
                group = ordered[offset : offset + self.batch_size]
                suffixes = [j.tokens[len(prefix) :] for j in group]
                width = max(map(len, suffixes))
                # Right padding is causal future context; read only each real final position.
                # These branches are discarded after one call, never reused for generation.
                pad = self.tokenizer.pad_token_id
                if pad is None:
                    pad = self.tokenizer.eos_token_id
                inputs = mx.array([s + [pad] * (width - len(s)) for s in suffixes])
                branch = branch_cache(cache, len(group))
                hidden = self.model.model(inputs, cache=branch)
                final = hidden[mx.arange(len(group)), mx.array([len(s) - 1 for s in suffixes])]
                union = sorted({slot for job in group for slot in job.slots})
                logits = selected_logits(self.model, final, union)
                mx.eval(logits)
                rows = logits.tolist()
                for job, row in zip(group, rows, strict=True):
                    result[job.id] = [row[union.index(slot)] for slot in job.slots]
                evaluated_tokens += len(group) * width
                batches += 1
                del branch, hidden, final, logits
        mx.synchronize()
        return result, Timing(
            inference_seconds=time.perf_counter() - started,
            prefill_seconds=prefix_seconds,
            shared_prefix_tokens=len(prefix) if mode == "shared" else 0,
            evaluated_tokens_including_padding=evaluated_tokens,
            logical_input_tokens=sum(len(j.tokens) for j in jobs),
            batches=batches,
            generated_tokens=0,
            peak_mlx_bytes=mx.get_peak_memory(),
        )
