"""llama.cpp runtime: flat-batch logic, slot selection and device resolution.

The lib-facing paths are exercised with a fake library object; the real library is covered by
the integration tests marked `llama`.
"""

import ctypes
import os
from pathlib import Path

import pytest

from rizzo_flow.llama_runtime import (
    BatchSpec,
    LlamaRuntime,
    build_batch_spec,
    detect_backends,
    resolve,
    select_slots,
)

# Real-weight tests are opt-in: `RIZZO_LLAMA_TEST=1 pytest -q -m llama`. They load a 4 GB
# checkpoint, so they must never run as part of the default fast unit suite.
LIBRARY_DIR = Path(os.environ.get("RIZZO_LLAMA_LIB", "/home/snorcini/git/llama.cpp-amd/build/bin"))
GGUF = Path(
    os.environ.get("RIZZO_LLAMA_GGUF", "/mnt/model-cache/rizzo-flow/Spark-X2.5-4B-Q8_0.gguf")
)
llama_integration = pytest.mark.skipif(
    os.environ.get("RIZZO_LLAMA_TEST") != "1"
    or not (LIBRARY_DIR / "libllama.so").exists()
    or not GGUF.is_file(),
    reason="set RIZZO_LLAMA_TEST=1 with libllama and a GGUF to run real-weight tests",
)


# --- flat batch: no padding, per-token positions and sequence ids -------------------------


def test_build_batch_spec_has_no_padding_and_reads_final_positions():
    prefix_length = 10
    suffixes = [[1, 2, 3], [4], [5, 6]]
    spec = build_batch_spec(prefix_length, suffixes, seq_ids=[1, 2, 3])
    assert isinstance(spec, BatchSpec)
    # every token of every suffix is present exactly once, in order, with no filler
    assert spec.tokens == [1, 2, 3, 4, 5, 6]
    assert spec.positions == [10, 11, 12, 10, 10, 11]
    assert spec.seq_ids == [1, 1, 1, 2, 3, 3]
    # logits are requested only at each suffix's last real token
    assert sorted(spec.logits_at) == [2, 3, 5]
    assert spec.finals == [2, 3, 5]


def test_build_batch_spec_rejects_empty_or_ragged_input():
    with pytest.raises(ValueError, match="[Nn]o suffixes"):
        build_batch_spec(0, [], [])
    with pytest.raises(ValueError, match="length"):
        build_batch_spec(0, [[1, 2]], [1, 2])
    with pytest.raises(ValueError, match="empty"):
        build_batch_spec(0, [[1], []], [1, 2])


def test_build_batch_spec_single_token_suffix():
    spec = build_batch_spec(7, [[9]], seq_ids=[4])
    assert (spec.tokens, spec.positions, spec.seq_ids) == ([9], [7], [4])
    assert spec.logits_at == frozenset({0})
    assert spec.finals == [0]


# --- selected logits ----------------------------------------------------------------------


def test_select_slots_reads_only_declared_rows():
    row = (ctypes.c_float * 8)(*[float(i) for i in range(8)])
    assert select_slots(row, [2, 5]) == [2.0, 5.0]
    assert select_slots(row, [7]) == [7.0]


def test_select_slots_rejects_missing_logits():
    with pytest.raises(ValueError, match="logits"):
        select_slots(None, [1])


# --- device resolution --------------------------------------------------------------------


def test_detect_backends_from_library_directory(tmp_path):
    assert detect_backends(tmp_path) == []
    (tmp_path / "libggml-vulkan.so.0.24.0").touch()
    assert detect_backends(tmp_path) == ["vulkan"]
    (tmp_path / "libggml-hip.so.0.24.0").touch()
    assert detect_backends(tmp_path) == ["vulkan", "hip"]


@pytest.mark.parametrize(
    ("device", "available", "expected"),
    [
        ("auto", ["vulkan"], ("vulkan", 99)),
        ("auto", ["hip"], ("hip", 99)),
        ("auto", [], ("cpu", 0)),
        ("gpu", ["vulkan"], ("vulkan", 99)),
        ("gpu", ["vulkan", "hip"], ("vulkan", 99)),
        ("vulkan", ["vulkan"], ("vulkan", 99)),
        ("hip", ["hip"], ("hip", 99)),
        ("cpu", ["vulkan"], ("cpu", 0)),
    ],
)
def test_resolve(device, available, expected):
    assert resolve(device, available) == expected


@pytest.mark.parametrize(
    ("device", "available"),
    [("gpu", []), ("vulkan", []), ("vulkan", ["hip"]), ("hip", ["vulkan"])],
)
def test_resolve_rejects_missing_backend(device, available):
    with pytest.raises(ValueError, match="--device"):
        resolve(device, available)


def test_resolve_rejects_unknown_name():
    with pytest.raises(ValueError, match="one of"):
        resolve("tpu", ["vulkan"])


# --- runtime calls against a fake library -------------------------------------------------


class FakeLib:
    """Records the exact calls LlamaRuntime makes; returns canned pointers and statuses."""

    def __init__(self, logits=None, decode_status=0, logits_pointer=True):
        self.calls = []
        self._logits = logits or [0.0] * 16
        self._decode_status = decode_status
        self._logits_pointer = logits_pointer
        self.seen_batch = None

    def llama_decode(self, context, batch):
        self.calls.append(("decode", batch.n_tokens))
        self.seen_batch = {
            "tokens": [batch.token[i] for i in range(batch.n_tokens)],
            "positions": [batch.pos[i] for i in range(batch.n_tokens)],
            "n_seq_id": [batch.n_seq_id[i] for i in range(batch.n_tokens)],
            "seq_ids": [batch.seq_id[i][0] for i in range(batch.n_tokens)],
            "logits": [bool(batch.logits[i]) for i in range(batch.n_tokens)],
        }
        return self._decode_status

    def llama_get_logits_ith(self, context, index):
        self.calls.append(("logits", index))
        if not self._logits_pointer:
            return None
        return (ctypes.c_float * len(self._logits))(*self._logits)

    def llama_get_memory(self, context):
        return ctypes.c_void_p(0x1234)

    def llama_memory_seq_cp(self, memory, src, dst, p0, p1):
        self.calls.append(("seq_cp", src, dst, p0, p1))

    def llama_memory_seq_rm(self, memory, seq_id, p0, p1):
        self.calls.append(("seq_rm", seq_id, p0, p1))
        return True

    def llama_memory_clear(self, memory, data):
        self.calls.append(("memory_clear", bool(data)))


def make_runtime(fake):
    return LlamaRuntime(fake, model=1, context=2, vocab=3, device="cpu", compute_backend="cpu")


def test_decode_fills_flat_batch_without_padding():
    fake = FakeLib()
    runtime = make_runtime(fake)
    spec = build_batch_spec(5, [[1, 2], [3, 4, 5]], seq_ids=[1, 2])
    runtime.decode(spec)
    seen = fake.seen_batch
    assert seen["tokens"] == [1, 2, 3, 4, 5]
    assert seen["positions"] == [5, 6, 5, 6, 7]
    assert seen["seq_ids"] == [1, 1, 2, 2, 2]
    assert seen["n_seq_id"] == [1, 1, 1, 1, 1]
    assert seen["logits"] == [False, True, False, False, True]


def test_decode_surfaces_failure_status():
    fake = FakeLib(decode_status=1)
    with pytest.raises(ValueError, match="llama_decode"):
        make_runtime(fake).decode(build_batch_spec(0, [[1]], seq_ids=[1]))


def test_logits_selects_slots_and_checks_pointer():
    fake = FakeLib(logits=[float(i) for i in range(8)])
    runtime = make_runtime(fake)
    assert runtime.logits(2, [1, 4]) == [1.0, 4.0]
    assert ("logits", 2) in fake.calls
    broken = FakeLib(logits_pointer=False)
    with pytest.raises(ValueError, match="[Nn]o logits"):
        make_runtime(broken).logits(0, [1])


def test_memory_branch_and_clear_call_the_memory_api():
    fake = FakeLib()
    runtime = make_runtime(fake)
    runtime.memory_clear()
    runtime.seq_copy(0, 3)
    runtime.seq_remove(3)
    assert fake.calls == [
        ("memory_clear", True),
        ("seq_cp", 0, 3, 0, -1),
        ("seq_rm", 3, -1, -1),
    ]


def test_runtime_rejects_bad_construction():
    with pytest.raises(ValueError, match="already closed|not loaded|missing"):
        LlamaRuntime(None, model=1, context=2, vocab=3, device="cpu", compute_backend="cpu")


def test_close_is_idempotent_and_blocks_use():
    fake = FakeLib()
    runtime = make_runtime(fake)
    runtime.close()
    runtime.close()
    with pytest.raises(ValueError, match="closed"):
        runtime.decode(build_batch_spec(0, [[1]], seq_ids=[1]))


# --- real library and checkpoint ----------------------------------------------------------


@pytest.mark.llama
@llama_integration
def test_real_llama_cpp_loads_spark_gguf_and_serves_selected_logits():
    with LlamaRuntime.load(LIBRARY_DIR, GGUF, n_ctx=256, n_seq_max=2, device="auto") as runtime:
        assert runtime.architecture == "spark2_5"
        assert runtime.compute_backend in ("vulkan", "hip", "cpu")
        assert runtime.tokenize("A") == [46]
        assert runtime.token_to_piece(46) == "A"
        prompt = runtime.tokenize("Question: is the sky blue?")
        runtime.memory_clear()
        runtime.decode(build_batch_spec(0, [prompt], [0]))
        runtime.seq_copy(0, 1)
        suffix = runtime.tokenize("Yes")
        spec = build_batch_spec(len(prompt), [suffix], [1])
        runtime.decode(spec)
        row = runtime.logits(spec.finals[0], [runtime.tokenize(letter)[0] for letter in "AB"])
        assert len(row) == 2 and all(isinstance(value, float) for value in row)
        runtime.seq_remove(1)
