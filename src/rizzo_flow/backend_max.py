"""MAX scoring: Spark2.5 as compiled MAX graphs, no generation, candidate rows only.

Same contract as `backend.SparkBackend` and `backend_llama.LlamaBackend` (tokenizer, `metadata`,
`score`). The weights are the BF16 safetensors checkpoint that MLX loads, bound by name from the
files. The prefix K/V stays on the device between calls; branches read it, never write it.
"""

import hashlib
import importlib.metadata
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .chat_template import compile_template
from .config import FLOW_CHECKPOINTS, checkpoint_hashes, identify
from .max_spark import SLIDING, SparkConfig, SparkGraphs
from .prompts import PROMPT_VERSION, Compiled, canonical


class MaxTokenizer:
    """The two tokenizer calls `prompts.compile_request` makes: the checkpoint's chat template
    rendered as transformers renders it, and its `tokenizer.json` for encoding."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        # transformers prefers the standalone template file over the config entry.
        template_file = path / "chat_template.jinja"
        if template_file.is_file():
            template = template_file.read_text(encoding="utf-8")
        else:
            template = config.get("chat_template")
        if not template:
            raise ValueError("The checkpoint carries no chat template")
        self.render = compile_template(template)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.pad_token_id = self._special(config.get("pad_token"))
        self.eos_token_id = self._special(config.get("eos_token"))

    def _special(self, token):
        if isinstance(token, dict):
            token = token.get("content")
        return self.tokenizer.token_to_id(token) if token else None

    def apply_chat_template(self, messages, tokenize=False, **variables) -> str:
        if tokenize:
            raise ValueError("Render the text, then call encode()")
        return self.render(messages=messages, **variables)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


def pick_device(device="auto"):
    from max.driver import CPU, Accelerator, accelerator_count

    if device not in ("auto", "gpu", "cpu"):
        raise ValueError("The MAX backend takes --device auto, gpu or cpu")
    if device == "cpu":
        return CPU()
    if accelerator_count():
        return Accelerator()
    if device == "gpu":
        raise ValueError("MAX sees no GPU on this machine; use --device cpu")
    return CPU()


def describe() -> dict:
    """The MAX section of `rizzo devices`; ImportError without the max extra."""
    from max.driver import Accelerator, accelerator_count

    accelerators = [Accelerator(i) for i in range(accelerator_count())]
    return {
        "installed": importlib.metadata.version("max"),
        "devices": [
            {
                "api": d.api,
                "architecture": d.architecture_name,
                "memory": d.stats.get("total_memory"),
            }
            for d in accelerators
        ],
        "auto_selects": "gpu" if accelerators else "cpu",
    }


FAR = 1 << 40  # position of unused cache rows: after every query, so never visible


class KVCache:
    """The K/V of a prefix: per layer, K then V, one [rows, D] device buffer per KV head.
    Full-attention layers hold `capacity` rows (a size bucket), sliding layers a ring of the
    last `ring` positions; `positions` give each row's absolute position (FAR when unused)."""

    def __init__(self, buffers, capacity, ring):
        self.buffers = buffers
        self.capacity = capacity
        self.length = 0
        self.positions = np.full(capacity, FAR, dtype=np.int64)
        self.ring_positions = np.full(ring, FAR, dtype=np.int64)


class SparkRunner:
    """Compiled graphs, one per size bucket and built on first use, and the device buffers
    they exchange."""

    def __init__(
        self,
        config: SparkConfig,
        weights,
        device,
        dtype,
        batch_size=4,
        chunk=512,
        block=128,
        max_resident=16,
        min_view=1024,
        score_tokens=1024,
    ):
        from max.driver import CPU, Buffer
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef

        self.config = config
        self.device = device
        self.dtype = dtype
        graphs = SparkGraphs(config, dtype, DeviceRef.from_device(device))
        missing = set(graphs.weight_shapes()) - set(weights)
        if missing:
            raise ValueError(f"Checkpoint lacks {len(missing)} tensors, e.g. {min(missing)}")
        # The graphs type every weight input; a mismatch would only surface on the first call.
        for name, shape in graphs.weight_shapes().items():
            found = tuple(int(d) for d in weights[name].shape)
            kind = weights[name].dtype  # max DType (safetensors) or numpy dtype (tests)
            kind = kind if isinstance(kind, DType) else DType.from_numpy(kind)
            if found != shape:
                raise ValueError(f"{name} has shape {found}, the config implies {shape}")
            if kind != dtype:
                raise ValueError(f"{name} is {kind}, the graphs expect {dtype}")
        # Uploaded once and passed to both graphs on every call. Fused projections are
        # concatenated on the host first, so the device never holds the separate halves.
        host = CPU()
        concat = InferenceSession(devices=[host]).load(
            self._concat_graph(DeviceRef.from_device(host))
        )
        self._weights = []
        for key, (names, shape) in graphs.input_shapes().items():
            if len(names) == 1:
                buffer = self._host_buffer(weights[names[0]])
            else:
                parts = [self._host_buffer(weights[name]) for name in names]
                padding = shape[0] - sum(part.shape[0] for part in parts)
                parts.append(Buffer.zeros([padding, shape[1]], dtype, device=host))
                buffer = concat.execute(*parts)[0]
            if graphs.weight_device(key).is_gpu():
                buffer = buffer.to(device)
            self._weights.append(buffer)
        # Sizes are multiples of `block` (128: what MAX's tensor-core matmul needs). A prefill
        # chunk must also divide the sliding ring, so its writes never wrap.
        self.block = block
        self.ring = graphs.ring_size(block)
        if chunk % block or self.ring % chunk:
            raise ValueError(
                f"prefill_chunk must be a multiple of {block} dividing {self.ring}, not {chunk}"
            )
        self.chunk = chunk
        self.min_view = min_view
        # Score attention is quadratic in rows * width (every row's new tokens are keys for all
        # rows, block-masked): a microbatch of 4 suffixes of ~1.3k tokens needs ~2 GB. Calls are
        # capped at this many tokens; `fits` tells the backend when to split.
        self.score_tokens = score_tokens
        # Score graphs have a static batch: one row for a lone suffix, `batch_size` rows for
        # microbatches, smaller groups padded with copies.
        self.batch_size = batch_size
        self._graphs = graphs
        self._session = InferenceSession(devices=[device])
        # Each compiled graph holds its own activation memory on the device: keep only the
        # most recently used ones (the others reload from MAX's disk cache in ~10 s).
        self._compiled = OrderedDict()
        self.max_resident = max_resident
        self._sliding = [kind == SLIDING for kind in config.layer_types]

    def _concat_graph(self, device):
        from max.graph import Graph, TensorType, ops

        hidden = self.config.hidden_size
        types = [TensorType(self.dtype, [rows, hidden], device=device) for rows in "ABC"]
        with Graph("spark_fuse", input_types=types) as graph:
            graph.output(ops.concat([v.tensor for v in graph.inputs], axis=0))
        return graph

    @staticmethod
    def _host_buffer(array):
        from max.driver import Buffer

        if isinstance(array, np.ndarray):
            return Buffer.from_numpy(np.ascontiguousarray(array))
        return Buffer.from_dlpack(array)  # safetensors WeightData (BF16)

    def _buffer(self, array):
        return self._host_buffer(array).to(self.device)

    def _model(self, kind, *sizes):
        """The compiled graph for a size bucket, built (and cached by MAX on disk) on first use."""
        key = (kind, *sizes)
        if key in self._compiled:
            self._compiled.move_to_end(key)
        else:
            while len(self._compiled) >= self.max_resident:
                self._compiled.popitem(last=False)
            self._compiled[key] = self._session.load(getattr(self._graphs, kind)(*sizes))
        return self._compiled[key]

    def _bucket(self, rows, unit):
        """The smallest `unit * 2**k` holding `rows`."""
        size = unit
        while size < rows:
            size *= 2
        return size

    def _layer_buffers(self, rows_full):
        from max.driver import Buffer

        c = self.config
        buffers = []
        for sliding in self._sliding:
            rows = self.ring if sliding else rows_full
            for _ in range(2 * c.num_kv_heads):
                buffers.append(Buffer.zeros([rows, c.head_dim], self.dtype, device=self.device))
        return buffers

    def _views(self, cache, rows):
        """The cache buffers, full-attention layers cut to their first `rows` rows."""
        per_layer = 2 * self.config.num_kv_heads
        return [
            buffer if self._sliding[index // per_layer] else buffer[:rows, :]
            for index, buffer in enumerate(cache.buffers)
        ]

    def _append(self, cache, outputs, count):
        """Write the first `count` rows of a prefill's new K/V into the cache."""
        start, per_layer = cache.length, 2 * self.config.num_kv_heads
        at = start % self.ring  # a chunk divides the ring, so this never wraps
        for index, (buffer, new) in enumerate(zip(cache.buffers, outputs, strict=True)):
            row = at if self._sliding[index // per_layer] else start
            buffer[row : row + count, :].inplace_copy_from(new[:count, :])
        cache.positions[start : start + count] = np.arange(start, start + count)
        cache.ring_positions[at : at + count] = np.arange(start, start + count)
        cache.length += count

    def prefill(self, tokens, slots=(0,), pad=0, keep=True):
        """(cache of `tokens`, fp32 logits of `slots` after the last token), fed one static
        chunk at a time. With `keep=False` the last chunk is not written to the cache (a
        `direct` question only needs its logits)."""
        from max.driver import CPU

        # Allocated once for the whole prefix; each chunk only attends to a view of the filled
        # part, in buckets of at least `min_view` rows: early chunks do not pay for the empty
        # rows, and a request uses two or three prefill graphs, not one per doubling.
        capacity = self._bucket(len(tokens), self.block)
        cache = KVCache(self._layer_buffers(capacity), capacity, self.ring)
        candidates = self._host_buffer(np.array(slots, dtype=np.int64))
        logits = None
        for start in range(0, len(tokens), self.chunk):
            piece = tokens[start : start + self.chunk]
            padded = np.full(self.chunk, pad, dtype=np.int64)
            padded[: len(piece)] = piece
            view = min(capacity, max(self.min_view, self._bucket(start, self.block)))
            model = self._model("prefill", view, self.ring, self.chunk)
            *outputs, logits = model.execute(
                self._host_buffer(padded),
                self._buffer(np.arange(start, start + self.chunk, dtype=np.int64)),
                self._buffer(np.array([len(piece) - 1], dtype=np.int64)),
                candidates,
                self._buffer(cache.positions[:view]),
                self._buffer(cache.ring_positions),
                *self._views(cache, view),
                *self._weights,
            )
            if keep or start + self.chunk < len(tokens):
                self._append(cache, outputs, len(piece))
        return cache, None if logits is None else logits.to(CPU()).to_numpy()[0]

    def _shape(self, count, longest):
        """(rows, width bucket) of the score graph for `count` suffixes: one row alone, else a
        full `batch_size` microbatch; rows * width must be a multiple of the block."""
        batch = 1 if count == 1 else self.batch_size
        return batch, self._bucket(longest, self.block // math.gcd(batch, self.block))

    def fits(self, count, longest):
        """Whether `count` suffixes, the longest `longest` tokens, fit one score call."""
        batch, width = self._shape(count, longest)
        return batch * width <= self.score_tokens

    def score(self, cache, suffixes, slots, pad):
        """fp32 logits [len(suffixes), len(slots)] after each suffix, all continuing `cache`."""
        from max.driver import CPU

        count = len(suffixes)
        if not 1 <= count <= self.batch_size:
            raise ValueError(f"score takes 1 to {self.batch_size} suffixes")
        batch, width = self._shape(count, max(map(len, suffixes)))
        rows = suffixes + [suffixes[0]] * (batch - count)  # padding rows, discarded
        tokens = np.full((batch, width), pad, dtype=np.int64)
        for row, suffix in enumerate(rows):
            tokens[row, : len(suffix)] = suffix
        # Right padding is causal future context: each row reads its own last real position.
        last = np.array([r * width + len(s) - 1 for r, s in enumerate(rows)], dtype=np.int64)
        start = cache.length
        model = self._model("score", cache.capacity, self.ring, batch, width)
        (logits,) = model.execute(
            self._host_buffer(tokens),
            self._buffer(np.arange(start, start + width, dtype=np.int64)),
            self._buffer(last),
            self._host_buffer(np.array(slots, dtype=np.int64)),
            self._buffer(cache.positions),
            self._buffer(cache.ring_positions),
            *cache.buffers,
            *self._weights,
        )
        return logits.to(CPU()).to_numpy()[:count]

    def free_bytes(self):
        if self.device.is_host:
            return None
        return self.device.stats.get("free_memory")


class MaxBackend:
    def __init__(
        self, runner, tokenizer, metadata, batch_size=4, prefill_chunk=512, idle_free=None
    ):
        if not 1 <= batch_size <= 16 or not 1 <= prefill_chunk <= 2048:
            raise ValueError("batch_size must be 1–16 and prefill_chunk 1–2048")
        self.runner = runner
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.batch_size = batch_size
        self.prefill_chunk = prefill_chunk
        self._idle_free = idle_free  # free device memory before the weights were loaded
        self._lowest_free = None

    @classmethod
    def load(cls, path, device="auto", ctx=8192, batch_size=4, prefill_chunk=512, max_resident=16):
        from max.dtype import DType
        from max.graph.weights import load_weights

        path = Path(path).resolve()
        if not path.is_dir():
            raise ValueError(f"Model not found at {path}. Run `rizzo download --backend max`.")
        if ctx < 1:
            raise ValueError("ctx must be positive")
        started = time.perf_counter()
        target = pick_device(device)
        idle_free = None if target.is_host else target.stats.get("free_memory")
        hashes = checkpoint_hashes(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        spark = SparkConfig.from_hf(config)
        spec = identify(config)
        weights = load_weights(sorted(path.glob("*.safetensors")))
        registry = {name: weight.data() for name, weight in weights.items()}
        runner = SparkRunner(
            spark,
            registry,
            target,
            DType.bfloat16,
            batch_size,
            prefill_chunk,
            max_resident=max_resident,
        )
        identity = {
            "source": spec.repo,
            "requested_revision": spec.revision,
            "source_files": hashes,
            "precision": "bf16",
            "device": "cpu" if target.is_host else "gpu",
            "backend": "cpu" if target.is_host else target.api,
            "runtime": "max",
            "max": importlib.metadata.version("max"),
            "prompt_version": PROMPT_VERSION,
        }
        # Only the fine-tune adds a key, as with the other backends.
        flow = FLOW_CHECKPOINTS.get(spec.size)
        if flow and flow.weights and all(hashes.get(k) == v for k, v in flow.weights.items()):
            identity["weights"] = "flow"
        metadata = {
            **identity,
            "fingerprint": hashlib.sha256(canonical(identity).encode()).hexdigest(),
            "device_name": None if target.is_host else target.architecture_name,
            "load_seconds": time.perf_counter() - started,
        }
        return cls(runner, MaxTokenizer(path), metadata, batch_size, prefill_chunk, idle_free)

    def _track_memory(self):
        free = self.runner.free_bytes()
        if free is not None:
            self._lowest_free = free if self._lowest_free is None else min(self._lowest_free, free)

    def peak_device_bytes(self) -> int | None:
        """Largest drop in free device memory since before the load (weights included). Other
        processes using the same GPU are counted too; None on the CPU."""
        if self._idle_free is None or self._lowest_free is None:
            return None
        return max(self._idle_free - self._lowest_free, 0)

    def _groups(self, ordered, prefix_length):
        """Microbatches of up to `batch_size` suffixes within the runner's token cap, and the
        jobs whose suffix alone exceeds it."""
        groups, alone, group = [], [], []
        for job in ordered:
            length = len(job.tokens) - prefix_length
            if not self.runner.fits(1, length):
                alone.append(job)
                continue
            if group and (
                len(group) == self.batch_size or not self.runner.fits(len(group) + 1, length)
            ):
                groups.append(group)
                group = []
            group.append(job)
        if group:
            groups.append(group)
        return groups, alone

    def score(self, prefix: list[int], jobs: list[Compiled], mode="shared"):
        if mode not in ("shared", "direct"):
            raise ValueError("Unknown execution mode")
        if not jobs:
            raise ValueError("No decisions supplied")
        if any(
            job.tokens[: len(prefix)] != prefix or len(job.tokens) <= len(prefix) for job in jobs
        ):
            raise ValueError("Invalid shared prefix")
        runner = self.runner
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id or 0
        started = time.perf_counter()
        result = {}
        prefix_seconds = 0.0
        evaluated_tokens = 0
        batches = 0
        # A lone question has nobody to share the prefix with: one pass, as with llama.cpp.
        reuse = mode == "shared" and bool(prefix) and len(jobs) > 1
        if not reuse:
            for job in jobs:
                # The prefill's own last position gives the answer: one pass, no score call.
                _, logits = runner.prefill(job.tokens, job.slots, pad, keep=False)
                result[job.id] = logits.tolist()
                self._track_memory()
                evaluated_tokens += len(job.tokens)
                batches += 1
        else:
            mark = time.perf_counter()
            cache, _ = runner.prefill(prefix, pad=pad)
            prefix_seconds = time.perf_counter() - mark
            evaluated_tokens += len(prefix)
            groups, alone = self._groups(sorted(jobs, key=lambda j: len(j.tokens)), len(prefix))
            for job in alone:
                # A suffix too long for one score call: the chunked prefill bounds its memory.
                _, logits = runner.prefill(job.tokens, job.slots, pad, keep=False)
                result[job.id] = logits.tolist()
                self._track_memory()
                evaluated_tokens += len(job.tokens)
                batches += 1
            for group in groups:
                suffixes = [job.tokens[len(prefix) :] for job in group]
                union = sorted({slot for job in group for slot in job.slots})
                logits = runner.score(cache, suffixes, union, pad)
                self._track_memory()  # prefix K/V plus this microbatch
                for job, row in zip(group, logits, strict=True):
                    result[job.id] = [float(row[union.index(slot)]) for slot in job.slots]
                evaluated_tokens += len(group) * max(map(len, suffixes))
                batches += 1
            del cache
        timing = {
            "inference_seconds": time.perf_counter() - started,
            "prefill_seconds": prefix_seconds,
            "shared_prefix_tokens": len(prefix) if reuse else 0,
            "evaluated_tokens_including_padding": evaluated_tokens,
            "logical_input_tokens": sum(len(j.tokens) for j in jobs),
            "batches": batches,
            "generated_tokens": 0,
        }
        if self._lowest_free is not None:
            timing["peak_device_bytes"] = self.peak_device_bytes()
        return result, timing
