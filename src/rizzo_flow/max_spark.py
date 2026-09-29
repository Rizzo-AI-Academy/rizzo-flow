"""Spark2.5 as MAX graphs: prefill a shared prefix, then read letter logits after short suffixes.

Two kinds of compiled graph. The weights are trailing graph inputs, not constants, so every
graph reads the same device buffers (as constants, each compiled model would hold its own copy):

- `prefill`: one chunk of new tokens after the cached ones; returns the chunk's K/V, which the
  runner writes into the cache, and the candidate logits after its last real token.
- `score`: right-padded suffixes that all continue the same cache; returns the logits of the
  requested token ids at each suffix's last real position. Nothing is written back.

All shapes are static, so graphs are built per size bucket (cache capacity, suffix width): MAX
only runs its tensor-core kernels on static sizes. Attention is written with 2-D matmuls and an
fp32 mask and softmax (like MLX's fused kernel). RoPE angles are computed in fp32 from the
positions, as MLX does, rather than read from a BF16 table.
"""

from dataclasses import dataclass

import numpy as np

SLIDING = "sliding_attention"
EMBEDDING = "model.embedding.weight"
# Graph weight role -> the checkpoint roles concatenated (along rows) to build it.
FUSED = {"qkvg": ("qkv", "gate"), "mlp_gate_up": ("mlp_gate", "mlp_up")}


@dataclass(frozen=True)
class SparkConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    sliding_window: int
    rms_norm_eps: float
    layer_types: tuple[str, ...]
    rope: dict  # layer type -> (theta, rotary dims)

    @classmethod
    def from_hf(cls, config: dict):
        if config.get("model_type") != "spark2_5":
            raise ValueError("Only the Spark2.5 architecture is supported")
        if not config.get("tie_word_embeddings", True):
            raise ValueError("Spark2.5 without tied embeddings is not supported")
        if config.get("hidden_act", "gelu") != "gelu" or config.get("attention_bias"):
            raise ValueError("Unsupported Spark2.5 variant (activation or attention bias)")
        head_dim = config["head_dim"]
        rope = {}
        for kind, params in config["rope_parameters"].items():
            dims = int(head_dim * params.get("partial_rotary_factor", 1.0))
            rope[kind] = (float(params.get("rope_theta", 10000.0)), dims)
        return cls(
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=head_dim,
            vocab_size=config["vocab_size"],
            sliding_window=config["sliding_window"],
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            layer_types=tuple(config["layer_types"]),
            rope=rope,
        )


class SparkGraphs:
    """Builds the graphs. Both take the weights as trailing inputs, in `input_shapes()` order,
    after their own inputs and the per-layer K/V. Projections that read the same input are
    fused into one matmul: Q, K, V and the output gate; the MLP's gate and up."""

    def __init__(self, config: SparkConfig, dtype, device):
        self.config = config
        self.dtype = dtype
        self.device = device

    # Short role -> (checkpoint name suffix, shape) for each block.
    def _block_weights(self):
        c = self.config
        qkv = (c.num_heads + 2 * c.num_kv_heads) * c.head_dim
        return {
            "input_norm": ("input_layernorm.weight", (c.hidden_size,)),
            "post_norm": ("post_attention_layernorm.weight", (c.hidden_size,)),
            "qkv": ("self_attn.q_k_v_proj.weight", (qkv, c.hidden_size)),
            "gate": ("self_attn.g_proj.weight", (c.num_heads, c.hidden_size)),
            "out": ("self_attn.out_proj.weight", (c.hidden_size, c.num_heads * c.head_dim)),
            "mlp_gate": ("mlp.gate_proj.weight", (c.intermediate_size, c.hidden_size)),
            "mlp_up": ("mlp.up_proj.weight", (c.intermediate_size, c.hidden_size)),
            "mlp_down": ("mlp.down_proj.weight", (c.hidden_size, c.intermediate_size)),
        }

    def weight_shapes(self) -> dict[str, tuple[int, ...]]:
        """Every tensor the graphs read, by checkpoint name."""
        c = self.config
        shapes = {
            "model.embedding.weight": (c.vocab_size, c.hidden_size),
            "model.norm.weight": (c.hidden_size,),
        }
        for i in range(c.num_layers):
            for name, shape in self._block_weights().values():
                shapes[f"model.layers.{i}.{name}"] = shape
        return shapes

    def input_shapes(self) -> dict[str, tuple[list[str], tuple[int, ...]]]:
        """The graphs' weight inputs, in order: key -> (checkpoint tensors to concatenate along
        rows, then zero rows up to the resulting shape)."""
        c = self.config
        parts = self._block_weights()
        fused = {part for group in FUSED.values() for part in group}
        inputs = {
            "model.embedding.weight": (["model.embedding.weight"], (c.vocab_size, c.hidden_size)),
            "model.norm.weight": (["model.norm.weight"], (c.hidden_size,)),
        }
        for i in range(c.num_layers):
            prefix = f"model.layers.{i}."
            for role, (name, shape) in parts.items():
                if role not in fused:
                    inputs[prefix + role] = ([prefix + name], shape)
            for role, group in FUSED.items():
                rows = sum(parts[part][1][0] for part in group)
                names = [prefix + parts[part][0] for part in group]
                # Zero rows up to a multiple of 128: MAX only picks its tensor-core matmul
                # when N is one (qkvg: 3080 -> 3200 on the 1.7B), else cuBLAS with a memset.
                inputs[prefix + role] = (names, (-(-rows // 128) * 128, c.hidden_size))
        return inputs

    def weight_device(self, key):
        """The embedding table stays in host memory: the graphs only need the rows of the
        input tokens and of the candidate letters, gathered there and moved to the device.
        On a small GPU that is 0.5 GiB (1.7B) or 0.6 GiB (4B) for the activations instead."""
        from max.graph import DeviceRef

        return DeviceRef.CPU() if key == EMBEDDING else self.device

    def _weight_types(self):
        from max.graph import TensorType

        return [
            TensorType(self.dtype, shape, device=self.weight_device(key))
            for key, (_, shape) in self.input_shapes().items()
        ]

    def _head(self, final, slots):
        """fp32 logits [rows, N]: the tied embedding's rows of the candidates only."""
        from max.dtype import DType
        from max.graph import ops

        return ops.cast(ops.matmul(final, ops.transpose(self._embed(slots), 0, 1)), DType.float32)

    def _embed(self, ids):
        """Rows of the host-side embedding table for int64 ids (on the host), on the device."""
        from max.graph import ops

        return ops.gather(self.embedding, ids, axis=0).to(self.device)

    def _bind_weights(self, values):
        named = dict(zip(self.input_shapes(), values, strict=True))
        self.embedding = named[EMBEDDING]
        self.norm = named["model.norm.weight"]
        fused = {part for group in FUSED.values() for part in group}
        roles = [role for role in self._block_weights() if role not in fused]
        self.layers = [
            {role: named[f"model.layers.{i}.{role}"] for role in [*roles, *FUSED]}
            for i in range(self.config.num_layers)
        ]

    # -- building blocks ---------------------------------------------------------------------

    def _linear(self, x, w):
        from max.graph import ops

        return ops.matmul(x, ops.transpose(w, 0, 1))

    def _rms_norm(self, x, w):
        from max.graph import ops

        # Scale by the weight in fp32 before rounding to BF16, as MLX's fused kernel does.
        return ops.rms_norm(x, w, self.config.rms_norm_eps, multiply_before_cast=True)

    def _rope(self, x, positions, kind):
        """Rotate the first `dims` channels of `x` [..., S, heads, head_dim] (half-split pairs),
        pass the rest through. `positions` is int64 [S]."""
        from max.dtype import DType
        from max.graph import ops

        theta, dims = self.config.rope[kind]
        half = dims // 2
        inv_freq = (1.0 / theta ** (np.arange(0, dims, 2, dtype=np.float64) / dims)).astype(
            np.float32
        )
        freq = ops.constant(inv_freq, DType.float32, device=self.device)
        angles = ops.outer(ops.cast(positions, DType.float32), freq)  # [S, half]
        cos = ops.unsqueeze(ops.cos(angles), 1)  # [S, 1, half]
        sin = ops.unsqueeze(ops.sin(angles), 1)
        xf = ops.cast(x, DType.float32)
        x1 = xf[..., :half]
        x2 = xf[..., half:dims]
        parts = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
        if dims < self.config.head_dim:
            parts.append(xf[..., dims:])
        return ops.cast(ops.concat(parts, axis=-1), x.dtype)

    def _masks(self, rows, positions, past_positions, window_positions):
        """Additive fp32 masks [rows, 1, S, cached + rows * S] per layer kind, built once per
        graph, for `rows` sequences of S new tokens that all continue the same cached ones.

        Keys are the cache (C rows on full-attention layers, a ring of Wc rows on sliding ones),
        then all rows' new tokens, of which a query sees its own row's. Everything is causal on
        absolute positions, so unused cache rows, which hold a far-future position, and padding
        tokens, which come after the real ones, are never seen by a real query."""
        from max.dtype import DType
        from max.graph import ops

        window = ops.constant(self.config.sliding_window, DType.int64, device=self.device)
        zero = ops.constant(0.0, DType.float32, device=self.device)
        neg = ops.constant(-np.inf, DType.float32, device=self.device)
        s = int(positions.shape[0])
        qpos = ops.unsqueeze(positions, 1)  # [S, 1]

        def visible(keys, sliding):  # keys [..., K] against qpos [S, 1] -> bool [S, K]
            allowed = ops.greater_equal(qpos, keys)
            if sliding:
                allowed = ops.logical_and(allowed, ops.greater(keys + window, qpos))
            return allowed

        row_of = ops.constant(np.arange(rows).reshape(rows, 1), DType.int64, device=self.device)
        key_row = ops.reshape(ops.broadcast_to(row_of, [rows, s]), [rows * s])
        key_pos = ops.reshape(ops.broadcast_to(ops.unsqueeze(positions, 0), [rows, s]), [rows * s])
        same_row = ops.equal(ops.unsqueeze(ops.unsqueeze(row_of, 2), 3), key_row)  # [R,1,1,RS]
        masks = {}
        for kind, cached in ((SLIDING, window_positions), ("full_attention", past_positions)):
            sliding = kind == SLIDING
            on_cache = ops.where(visible(ops.unsqueeze(cached, 0), sliding), zero, neg)
            on_new = visible(ops.unsqueeze(key_pos, 0), sliding)  # [S, RS]
            on_new = ops.where(ops.logical_and(same_row, on_new), zero, neg)  # [R, 1, S, RS]
            shape = [rows, 1, s, int(cached.shape[0])]
            masks[kind] = ops.concat([ops.broadcast_to(on_cache, shape), on_new], axis=-1)
        return masks

    def _attention(self, q, k, v, past_k, past_v, mask):
        """Grouped-query attention of q [R, S, H, D] over the shared cache (past K and V, one
        [C, D] tensor per KV head) and the rows' own new keys k, v [R, S, Hkv, D], with an
        additive `mask` [R, 1, S, C + R*S] from `_masks`. Returns [R, S, H, D].

        Per KV head, one matmul pair for all rows' queries at once, against the cached keys
        followed by every row's new keys, block-masked to each row: the kernel count does not
        grow with the batch and the cache is never copied per row. Every size is static and a
        multiple of 128, so MAX picks its tensor-core matmul: with symbolic lengths it fell back
        to cuBLAS (plus a 32 MiB memset per call) or to a naive batched kernel, and its flash
        kernel has no fast path for 256-wide heads before Hopper. Scores and softmax are fp32,
        as in MLX (BF16 scores moved probabilities by up to 0.03); the CPU uses fp32 throughout,
        MAX's BF16 matmul being slow there."""
        from max.dtype import DType
        from max.graph import ops

        c = self.config
        group = c.num_heads // c.num_kv_heads
        rows, s = int(q.shape[0]), int(q.shape[1])
        keys = int(past_k[0].shape[0]) + rows * s
        compute = q.dtype if self.device.is_gpu() else DType.float32
        scale = c.head_dim**-0.5
        heads = []
        for j in range(c.num_kv_heads):
            # Query heads j*G .. j*G+G-1 read KV head j, as in MLX and transformers.
            qj = ops.permute(q[:, :, j * group : (j + 1) * group, :], [0, 2, 1, 3])  # [R,G,S,D]
            qj = ops.reshape(qj, [rows * group * s, c.head_dim])
            kj, vj = (
                ops.concat([past[j], ops.reshape(x[:, :, j, :], [rows * s, c.head_dim])], axis=0)
                for past, x in ((past_k, k), (past_v, v))
            )  # [C + R*S, D]
            qf, kf = ops.cast(qj, DType.float32), ops.cast(kj, DType.float32)
            scores = ops.matmul(qf, ops.transpose(kf, 0, 1)) * scale
            probs = ops.softmax(ops.reshape(scores, [rows, group, s, keys]) + mask)
            probs = ops.cast(ops.reshape(probs, [rows * group * s, keys]), compute)
            out = ops.matmul(probs, ops.cast(vj, compute))  # [R*G*S, D]
            out = ops.permute(ops.reshape(out, [rows, group, s, c.head_dim]), [0, 2, 1, 3])
            heads.append(out)  # [R, S, G, D]
        return ops.cast(ops.concat(heads, axis=2), q.dtype)

    def _layer(self, i, h, positions, past_k, past_v, masks):
        """One block on h [R, S, hidden]; past K and V (one [C, D] tensor per KV head) shared by
        the rows. Returns (h, k, v) with k, v the rows' new keys and values [R, S, Hkv, D]."""
        from max.dtype import DType
        from max.graph import ops

        c = self.config
        w = self.layers[i]
        kind = c.layer_types[i]
        b, s = h.shape[0], h.shape[1]
        x = self._rms_norm(h, w["input_norm"])
        qkvg = self._linear(x, w["qkvg"])
        q_size = c.num_heads * c.head_dim
        kv_size = c.num_kv_heads * c.head_dim
        end = q_size + 2 * kv_size
        q = ops.reshape(qkvg[..., :q_size], [b, s, c.num_heads, c.head_dim])
        k = ops.reshape(qkvg[..., q_size : q_size + kv_size], [b, s, c.num_kv_heads, c.head_dim])
        v = ops.reshape(qkvg[..., q_size + kv_size : end], [b, s, c.num_kv_heads, c.head_dim])
        q = self._rope(q, positions, kind)
        k = self._rope(k, positions, kind)
        attn = self._attention(q, k, v, past_k, past_v, masks[kind])
        gate = qkvg[..., end : end + c.num_heads]  # then the zero padding rows
        gate = ops.cast(ops.sigmoid(ops.cast(gate, DType.float32)), h.dtype)
        attn = attn * ops.unsqueeze(gate, -1)
        h = h + self._linear(ops.reshape(attn, [b, s, c.num_heads * c.head_dim]), w["out"])
        x = self._rms_norm(h, w["post_norm"])
        gate_up = self._linear(x, w["mlp_gate_up"])
        mlp = ops.gelu(gate_up[..., : c.intermediate_size]) * gate_up[..., c.intermediate_size :]
        return h + self._linear(mlp, w["mlp_down"]), k, v

    def ring_size(self, block):
        """Rows of a sliding layer's ring cache: the window, rounded up to whole blocks."""
        return -(-self.config.sliding_window // block) * block

    def _cache_types(self, capacity, ring):
        """Per layer, K then V, one [rows, D] tensor per KV head: `capacity` rows on
        full-attention layers, `ring` rows on sliding ones (they only read the window)."""
        from max.graph import TensorType

        c = self.config
        types = []
        for kind in c.layer_types:
            rows = ring if kind == SLIDING else capacity
            head = TensorType(self.dtype, [rows, c.head_dim], device=self.device)
            types += [head] * (2 * c.num_kv_heads)
        return types

    def _split_cache(self, values):
        """Graph cache inputs -> per layer (K heads, V heads)."""
        per_layer = 2 * self.config.num_kv_heads
        heads = self.config.num_kv_heads
        return [
            (
                values[i * per_layer : i * per_layer + heads],
                values[i * per_layer + heads : (i + 1) * per_layer],
            )
            for i in range(self.config.num_layers)
        ]

    # -- graphs ------------------------------------------------------------------------------
    # Every shape is static (MAX's fast kernels need it): graphs are built per size bucket.

    def prefill(self, capacity: int, ring: int, chunk: int):
        """Inputs: tokens [chunk] (right-padded), positions [chunk], index of the last real
        token [1], candidate token ids [N], cache positions [capacity] and [ring], the cache,
        weights. Outputs: the new K and V per layer and KV head [chunk, D] (the runner writes
        the real rows into the cache), then the fp32 logits [1, N] of the candidates after the
        last real token, so a single question needs no separate `score` call."""
        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops

        c = self.config
        inputs = [
            TensorType(DType.int64, [chunk], device=DeviceRef.CPU()),
            TensorType(DType.int64, [chunk], device=self.device),
            TensorType(DType.int64, [1], device=self.device),
            TensorType(DType.int64, ["N"], device=DeviceRef.CPU()),
            TensorType(DType.int64, [capacity], device=self.device),
            TensorType(DType.int64, [ring], device=self.device),
            *self._cache_types(capacity, ring),
            *self._weight_types(),
        ]
        cached = 2 * c.num_layers * c.num_kv_heads
        with Graph(f"spark_prefill_{capacity}_{ring}_{chunk}", input_types=inputs) as graph:
            values = [v.tensor for v in graph.inputs]
            tokens, positions, last, slots, past_positions, window_positions = values[:6]
            cache = self._split_cache(values[6 : 6 + cached])
            self._bind_weights(values[6 + cached :])
            masks = self._masks(1, positions, past_positions, window_positions)
            h = ops.unsqueeze(self._embed(tokens), 0)
            outputs = []
            for i in range(c.num_layers):
                h, k, v = self._layer(i, h, positions, *cache[i], masks)
                outputs += [k[0, :, j, :] for j in range(c.num_kv_heads)]
                outputs += [v[0, :, j, :] for j in range(c.num_kv_heads)]
            final = self._rms_norm(ops.gather(h[0], last, axis=0), self.norm)  # [1, hidden]
            graph.output(*outputs, self._head(final, slots))
        return graph

    def score(self, capacity: int, ring: int, batch: int, width: int):
        """Inputs: tokens [batch, width] right-padded, positions [width], each row's last real
        token as a flat index `r * width + last` [batch], candidate token ids [N], cache
        positions [capacity] and [ring], the cache, weights.
        Output: fp32 logits [batch, N] from the tied embedding rows of the candidates only."""
        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops

        c = self.config
        inputs = [
            TensorType(DType.int64, [batch, width], device=DeviceRef.CPU()),
            TensorType(DType.int64, [width], device=self.device),
            TensorType(DType.int64, [batch], device=self.device),
            TensorType(DType.int64, ["N"], device=DeviceRef.CPU()),
            TensorType(DType.int64, [capacity], device=self.device),
            TensorType(DType.int64, [ring], device=self.device),
            *self._cache_types(capacity, ring),
            *self._weight_types(),
        ]
        cached = 2 * c.num_layers * c.num_kv_heads
        name = f"spark_score_{capacity}_{ring}_{batch}_{width}"
        with Graph(name, input_types=inputs) as graph:
            values = [v.tensor for v in graph.inputs]
            tokens, positions, last, slots, past_positions, window_positions = values[:6]
            cache = self._split_cache(values[6 : 6 + cached])
            self._bind_weights(values[6 + cached :])
            masks = self._masks(batch, positions, past_positions, window_positions)
            h = self._embed(tokens)  # [batch, width, hidden]
            for i in range(c.num_layers):
                h, _, _ = self._layer(i, h, positions, *cache[i], masks)
            flat = ops.reshape(h, [batch * width, c.hidden_size])
            final = self._rms_norm(ops.gather(flat, last, axis=0), self.norm)  # [B, hidden]
            graph.output(self._head(final, slots))
        return graph
