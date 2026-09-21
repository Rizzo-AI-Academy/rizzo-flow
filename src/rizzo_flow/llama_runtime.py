"""ctypes binding to libllama: flat batches, explicit KV branches, selected answer logits.

Bindings and struct layouts follow the pinned header of llama.cpp v0.4.1, commit
b29c606e28a01b1bc8c1351026a0fa6e616bf6c4:
https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/include/llama.h
Only the functions this project uses are declared; the structs are transcribed field by field
because ctypes must know the exact layout to receive them by value.
"""

import ctypes
import os
from dataclasses import dataclass, field
from pathlib import Path

# Pinned build. Changing this invalidates the runtime identity of every GGUF cache entry.
LLAMA_COMMIT = "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4"
LLAMA_VERSION = "0.4.1"
DEVICES = ("auto", "gpu", "vulkan", "hip", "cpu")
GPU_LAYERS = 99  # offload every layer; llama.cpp clamps to the number of layers it has

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32


class LlamaModelParams(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("load_mode", ctypes.c_int),
        ("lazy_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]


class LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("flash_attn_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]


class LlamaBatch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


@dataclass(frozen=True)
class BatchSpec:
    """Flat decode plan; llama.cpp never needs padding between suffixes of different length."""

    tokens: list[int]
    positions: list[int]
    seq_ids: list[int]
    logits_at: frozenset[int]  # flat indices whose logits must be materialised
    finals: list[int]  # per suffix, the flat index of its last real token


@dataclass
class Batch:
    """A batch plus the arrays it points at; ctypes does not keep them alive by itself."""

    struct: LlamaBatch
    keepalive: list = field(default_factory=list)


def build_batch_spec(
    prefix_length: int, suffixes: list[list[int]], seq_ids: list[int]
) -> BatchSpec:
    """Concatenate suffixes after a shared prefix, each on its own sequence, with no padding."""
    if not suffixes:
        raise ValueError("No suffixes supplied")
    if len(suffixes) != len(seq_ids):
        raise ValueError("suffixes and seq_ids must have the same length")
    if any(not suffix for suffix in suffixes):
        raise ValueError("Suffixes must not be empty")
    tokens, positions, flat_seq_ids, finals = [], [], [], []
    for suffix, seq_id in zip(suffixes, seq_ids, strict=True):
        for offset, token in enumerate(suffix):
            tokens.append(token)
            positions.append(prefix_length + offset)
            flat_seq_ids.append(seq_id)
        finals.append(len(tokens) - 1)
    return BatchSpec(tokens, positions, flat_seq_ids, frozenset(finals), finals)


def select_slots(pointer, slots: list[int]) -> list[float]:
    """Read only the declared answer-token rows of the returned vocabulary vector."""
    if not pointer:
        raise ValueError("No logits for the requested batch position")
    return [float(pointer[slot]) for slot in slots]


def detect_backends(library_dir: Path) -> list[str]:
    """GPU compute backends shipped next to libllama; ggml loads them at runtime."""
    directory = Path(library_dir)
    found = []
    for name in ("vulkan", "hip"):
        if any(directory.glob(f"libggml-{name}.so*")):
            found.append(name)
    return found


def resolve(device: str, available: list[str]) -> tuple[str, int]:
    """Map a user-facing device name to (compute backend, layers to offload)."""
    if device not in DEVICES:
        raise ValueError(f"Device must be one of: {', '.join(DEVICES)}")
    if device == "cpu":
        return "cpu", 0
    gpu = next((name for name in available if name in ("vulkan", "hip")), None)
    if device == "auto":
        return (gpu, GPU_LAYERS) if gpu else ("cpu", 0)
    if device == "gpu":
        if gpu is None:
            raise ValueError("--device gpu: this build has no GPU backend")
        return gpu, GPU_LAYERS
    if device not in available:
        raise ValueError(f"--device {device}: not in this build; available: {', '.join(available)}")
    return device, GPU_LAYERS


def new_batch(n_tokens: int, n_seq_max: int) -> Batch:
    """Allocate a llama_batch we own: every token carries one sequence id and an optional logits flag."""
    if n_tokens < 1:
        raise ValueError("A batch needs at least one token")
    if n_seq_max < 1:
        raise ValueError("n_seq_max must be positive")
    struct = LlamaBatch()
    tokens = (llama_token * n_tokens)()
    positions = (llama_pos * n_tokens)()
    counts = (ctypes.c_int32 * n_tokens)()
    storage = (llama_seq_id * (n_tokens * n_seq_max))()
    pointers = (ctypes.POINTER(llama_seq_id) * n_tokens)()
    step = ctypes.sizeof(llama_seq_id) * n_seq_max
    for index in range(n_tokens):
        pointers[index] = ctypes.cast(
            ctypes.byref(storage, index * step), ctypes.POINTER(llama_seq_id)
        )
    logits = (ctypes.c_int8 * n_tokens)()
    struct.n_tokens = n_tokens
    struct.token = tokens
    struct.pos = positions
    struct.n_seq_id = counts
    struct.seq_id = pointers
    struct.logits = logits
    return Batch(struct, [tokens, positions, counts, storage, pointers, logits])


def fill_batch(batch: Batch, spec: BatchSpec) -> None:
    if batch.struct.n_tokens != len(spec.tokens):
        raise ValueError("Batch size does not match the decode plan")
    for index in range(len(spec.tokens)):
        batch.struct.token[index] = spec.tokens[index]
        batch.struct.pos[index] = spec.positions[index]
        batch.struct.n_seq_id[index] = 1
        batch.struct.seq_id[index][0] = spec.seq_ids[index]
        batch.struct.logits[index] = 1 if index in spec.logits_at else 0


class _Lib:
    """Thin typed wrapper over the shared library; declared once, reused by every call."""

    def __init__(self, cdll):
        self._cdll = cdll
        self._declare()

    @classmethod
    def open(cls, library_dir: Path) -> "_Lib":
        directory = Path(library_dir)
        if not directory.is_dir():
            raise ValueError(f"llama.cpp library directory not found: {directory}")
        # Load the ggml backends first: libllama.so resolves them lazily.
        for name in ("libggml-base.so", "libggml.so", "libggml-cpu.so", "libggml-vulkan.so"):
            candidate = directory / name
            if candidate.exists():
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        return cls(ctypes.CDLL(str(directory / "libllama.so")))

    def _declare(self):
        lib = self._cdll
        declarations = {
            "llama_backend_init": (None, []),
            "llama_backend_free": (None, []),
            "llama_model_default_params": (LlamaModelParams, []),
            "llama_context_default_params": (LlamaContextParams, []),
            "llama_model_load_from_file": (ctypes.c_void_p, [ctypes.c_char_p, LlamaModelParams]),
            "llama_model_free": (None, [ctypes.c_void_p]),
            "llama_init_from_model": (ctypes.c_void_p, [ctypes.c_void_p, LlamaContextParams]),
            "llama_free": (None, [ctypes.c_void_p]),
            "llama_model_get_vocab": (ctypes.c_void_p, [ctypes.c_void_p]),
            "llama_model_n_ctx_train": (ctypes.c_int32, [ctypes.c_void_p]),
            "llama_model_size": (ctypes.c_uint64, [ctypes.c_void_p]),
            "llama_model_desc": (
                ctypes.c_int32,
                [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t],
            ),
            "llama_model_meta_val_str": (
                ctypes.c_int32,
                [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t],
            ),
            "llama_model_chat_template": (ctypes.c_char_p, [ctypes.c_void_p, ctypes.c_char_p]),
            "llama_vocab_n_tokens": (ctypes.c_int32, [ctypes.c_void_p]),
            "llama_vocab_pad": (llama_token, [ctypes.c_void_p]),
            "llama_vocab_eos": (llama_token, [ctypes.c_void_p]),
            "llama_vocab_is_eog": (ctypes.c_bool, [ctypes.c_void_p, llama_token]),
            "llama_tokenize": (
                ctypes.c_int32,
                [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_int32,
                    ctypes.POINTER(llama_token),
                    ctypes.c_int32,
                    ctypes.c_bool,
                    ctypes.c_bool,
                ],
            ),
            "llama_token_to_piece": (
                ctypes.c_int32,
                [
                    ctypes.c_void_p,
                    llama_token,
                    ctypes.c_char_p,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_bool,
                ],
            ),
            "llama_get_memory": (ctypes.c_void_p, [ctypes.c_void_p]),
            "llama_memory_clear": (None, [ctypes.c_void_p, ctypes.c_bool]),
            "llama_memory_seq_cp": (
                None,
                [ctypes.c_void_p, llama_seq_id, llama_seq_id, llama_pos, llama_pos],
            ),
            "llama_memory_seq_rm": (
                ctypes.c_bool,
                [ctypes.c_void_p, llama_seq_id, llama_pos, llama_pos],
            ),
            "llama_decode": (ctypes.c_int32, [ctypes.c_void_p, LlamaBatch]),
            "llama_get_logits_ith": (
                ctypes.POINTER(ctypes.c_float),
                [ctypes.c_void_p, ctypes.c_int32],
            ),
            "llama_n_ctx": (ctypes.c_uint32, [ctypes.c_void_p]),
            "llama_print_system_info": (ctypes.c_char_p, []),
        }
        for name, (restype, argtypes) in declarations.items():
            function = getattr(lib, name, None)
            if function is None:
                raise ValueError(f"libllama is missing {name}; wrong build or version")
            function.restype = restype
            function.argtypes = argtypes
            # Re-expose it on the wrapper so callers use typed, checked functions.
            setattr(self, name, function)


class LlamaRuntime:
    """Owns a model, one reusable context and the memory module used for KV branches."""

    def __init__(self, lib, *, model=None, context=None, vocab=None, device, compute_backend):
        if lib is None or model is None or context is None or vocab is None:
            raise ValueError("LlamaRuntime: missing library or model handles")
        self._lib = lib
        self._model = model
        self._context = context
        self._vocab = vocab
        self.device = device
        self.compute_backend = compute_backend
        self._memory = None
        self._closed = False

    # --- lifecycle ------------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        library_dir: Path,
        gguf: Path,
        *,
        n_ctx: int = 8192,
        n_batch: int = 2048,
        n_ubatch: int | None = None,
        n_seq_max: int = 5,
        device: str = "auto",
        threads: int | None = None,
    ) -> "LlamaRuntime":
        if n_ctx < 1:
            raise ValueError("n_ctx must be positive")
        gguf = Path(gguf)
        if not gguf.is_file():
            raise ValueError(f"GGUF file not found: {gguf} (run `rizzo download --format gguf`)")
        lib = _Lib.open(Path(library_dir))
        compute_backend, gpu_layers = resolve(device, detect_backends(Path(library_dir)))
        lib.llama_backend_init()
        model_params = lib.llama_model_default_params()
        model_params.n_gpu_layers = gpu_layers
        model = lib.llama_model_load_from_file(str(gguf).encode(), model_params)
        if not model:
            raise ValueError(f"Cannot load GGUF checkpoint: {gguf}")
        context_params = lib.llama_context_default_params()
        context_params.n_ctx = n_ctx
        context_params.n_batch = min(n_batch, n_ctx)
        context_params.n_ubatch = min(n_ubatch or n_batch, context_params.n_batch)
        context_params.n_seq_max = max(n_seq_max, 1)
        workers = threads or os.cpu_count() or 1
        context_params.n_threads = workers
        context_params.n_threads_batch = workers
        context_params.offload_kqv = compute_backend != "cpu"
        context = lib.llama_init_from_model(model, context_params)
        if not context:
            lib.llama_model_free(model)
            raise ValueError("Cannot create a llama.cpp context")
        return cls(
            lib,
            model=model,
            context=context,
            vocab=lib.llama_model_get_vocab(model),
            device="cpu" if compute_backend == "cpu" else "gpu",
            compute_backend=compute_backend,
        )

    def _check_open(self):
        if self._closed:
            raise ValueError("LlamaRuntime is closed")

    def close(self):
        if self._closed:
            return
        self._closed = True
        free_context = getattr(self._lib, "llama_free", None)
        free_model = getattr(self._lib, "llama_model_free", None)
        backend_free = getattr(self._lib, "llama_backend_free", None)
        if free_context:
            free_context(self._context)
        if free_model:
            free_model(self._model)
        if backend_free:
            backend_free()
        self._context = self._model = self._vocab = self._memory = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    # --- inference ------------------------------------------------------------------------

    def memory(self):
        self._check_open()
        if self._memory is None:
            self._memory = self._lib.llama_get_memory(self._context)
        return self._memory

    def decode(self, spec: BatchSpec) -> None:
        self._check_open()
        batch = new_batch(len(spec.tokens), 1)
        fill_batch(batch, spec)
        status = self._lib.llama_decode(self._context, batch.struct)
        if status != 0:
            raise ValueError(f"llama_decode failed with status {status}")

    def logits(self, index: int, slots: list[int]) -> list[float]:
        self._check_open()
        return select_slots(self._lib.llama_get_logits_ith(self._context, index), slots)

    def memory_clear(self) -> None:
        self._lib.llama_memory_clear(self.memory(), True)

    def seq_copy(self, source: int, destination: int) -> None:
        self._lib.llama_memory_seq_cp(self.memory(), source, destination, 0, -1)

    def seq_remove(self, seq_id: int) -> None:
        self._lib.llama_memory_seq_rm(self.memory(), seq_id, -1, -1)

    # --- vocabulary and metadata ----------------------------------------------------------

    def tokenize(
        self, text: str, *, add_special: bool = False, parse_special: bool = True
    ) -> list[int]:
        self._check_open()
        raw = text.encode()
        needed = self._lib.llama_tokenize(
            self._vocab, raw, len(raw), None, 0, add_special, parse_special
        )
        if needed == 0:
            return []
        if needed < 0:
            needed = -needed
        buffer = (llama_token * needed)()
        written = self._lib.llama_tokenize(
            self._vocab, raw, len(raw), buffer, needed, add_special, parse_special
        )
        if written < 0:
            raise ValueError("llama_tokenize needs a larger buffer")
        return [buffer[index] for index in range(written)]

    def token_to_piece(self, token: int) -> str:
        self._check_open()
        buffer = ctypes.create_string_buffer(64)
        written = self._lib.llama_token_to_piece(self._vocab, token, buffer, len(buffer), 0, True)
        if written < 0:
            raise ValueError("llama_token_to_piece needs a larger buffer")
        return buffer.raw[:written].decode("utf-8", errors="replace")

    def is_end_of_generation(self, token: int) -> bool:
        return bool(self._lib.llama_vocab_is_eog(self._vocab, token))

    @property
    def pad_token_id(self):
        return self._lib.llama_vocab_pad(self._vocab)

    @property
    def eos_token_id(self):
        return self._lib.llama_vocab_eos(self._vocab)

    @property
    def vocab_size(self) -> int:
        return self._lib.llama_vocab_n_tokens(self._vocab)

    def chat_template(self) -> str | None:
        template = self._lib.llama_model_chat_template(self._model, None)
        return template.decode("utf-8") if template else None

    def meta(self, key: str) -> str | None:
        buffer = ctypes.create_string_buffer(512)
        written = self._lib.llama_model_meta_val_str(self._model, key.encode(), buffer, len(buffer))
        if written < 0:
            return None
        return buffer.value.decode("utf-8")

    @property
    def architecture(self) -> str | None:
        return self.meta("general.architecture")

    @property
    def file_type(self) -> str | None:
        return self.meta("general.file_type")

    @property
    def description(self) -> str:
        buffer = ctypes.create_string_buffer(256)
        self._lib.llama_model_desc(self._model, buffer, len(buffer))
        return buffer.value.decode("utf-8")

    @property
    def model_bytes(self) -> int:
        return int(self._lib.llama_model_size(self._model))

    @property
    def n_ctx(self) -> int:
        return int(self._lib.llama_n_ctx(self._context))

    @property
    def train_context(self) -> int:
        return int(self._lib.llama_model_n_ctx_train(self._model))

    @property
    def build_info(self) -> str:
        return (self._lib.llama_print_system_info() or b"").decode("utf-8")
