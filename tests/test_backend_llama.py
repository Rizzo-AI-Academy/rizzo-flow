"""llama.cpp backend logic against a recording fake session: no library, no weights."""

import pytest

from rizzo_flow import backend_llama, loader
from rizzo_flow.backend_llama import LlamaBackend, LlamaTokenizer
from rizzo_flow.llama_cpp import Device, choose_device
from rizzo_flow.prompts import Compiled


class FakeSession:
    """Records every call; the logit of a slot is `position of the row + slot / 1000`."""

    device = None
    idle_free = None
    pad_token = None
    eos_token = 2

    def __init__(self):
        self.calls = []
        self.cells = {}  # sequence -> positions held
        self.last = None

    def clear(self):
        self.calls.append(("clear",))
        self.cells = {}

    def branch(self, source, target):
        self.calls.append(("branch", source, target))
        self.cells[target] = list(self.cells[source])

    def drop(self, sequence):
        self.calls.append(("drop", sequence))
        del self.cells[sequence]

    def decode(self, tokens, positions, sequences, outputs=()):
        tokens, positions, sequences = list(tokens), list(positions), list(sequences)
        assert len(tokens) == len(positions) == len(sequences) <= backend_llama.N_BATCH
        for position, sequence in zip(positions, sequences, strict=True):
            held = self.cells.setdefault(sequence, [])
            assert position == len(held), "positions must continue each sequence without gaps"
            held.append(position)
        self.calls.append(("decode", tokens, positions, sequences, list(outputs)))
        self.last = (positions, list(outputs))

    def logits(self, index, slots):
        positions, outputs = self.last
        assert index in outputs, "logits were not requested at this row"
        return [positions[index] + slot / 1000 for slot in slots]

    def synchronize(self):
        pass

    def free_bytes(self):
        return None

    def tokenize(self, text, add_special=False):
        return [ord(c) for c in text]


def backend(batch_size=4):
    return LlamaBackend(FakeSession(), None, {"fingerprint": "test"}, batch_size=batch_size)


def job(name, prefix, suffix, slots=(65, 66)):
    return Compiled(name, prefix + suffix, list(slots), "hash")


def test_shared_prefills_once_and_packs_suffixes_without_padding():
    prefix = list(range(100, 110))
    jobs = [
        job("long", prefix, [1, 2, 3, 4]),
        job("short", prefix, [5, 6]),
        job("mid", prefix, [7, 8, 9]),
    ]
    engine = backend(batch_size=2)
    logits, timing = engine.score(prefix, jobs, "shared")
    calls = engine.session.calls
    assert calls[0] == ("clear",)
    assert calls[1] == ("decode", prefix, list(range(10)), [0] * 10, [])
    # Shortest first, two per microbatch, laid end to end on sequences 1 and 2.
    assert calls[2:4] == [("branch", 0, 1), ("branch", 0, 2)]
    assert calls[4] == ("decode", [5, 6, 7, 8, 9], [10, 11, 10, 11, 12], [1, 1, 2, 2, 2], [1, 4])
    assert calls[5:7] == [("drop", 1), ("drop", 2)]
    assert calls[7:] == [
        ("branch", 0, 1),
        ("decode", [1, 2, 3, 4], [10, 11, 12, 13], [1] * 4, [3]),
        ("drop", 1),
    ]
    # Each answer is read at the last position of its own question.
    assert logits == {"short": [11.065, 11.066], "mid": [12.065, 12.066], "long": [13.065, 13.066]}
    assert engine.session.cells == {0: list(range(10))}  # the prefix survives every microbatch
    assert timing.batches == 2 and timing.shared_prefix_tokens == 10
    assert timing.evaluated_tokens_including_padding == 10 + 5 + 4
    assert timing.logical_input_tokens == 14 + 12 + 13
    assert timing.generated_tokens == 0 and timing.peak_device_bytes is None
    assert "peak_device_bytes" not in timing.model_dump()


def test_direct_recomputes_every_question_from_an_empty_cache():
    prefix = [100, 101]
    jobs = [job("a", prefix, [1]), job("b", prefix, [2, 3])]
    engine = backend()
    logits, timing = engine.score(prefix, jobs, "direct")
    assert [c[0] for c in engine.session.calls] == ["clear", "decode", "clear", "decode"]
    assert engine.session.calls[3] == ("decode", [100, 101, 2, 3], [0, 1, 2, 3], [0] * 4, [3])
    assert logits["a"][0] == pytest.approx(2.065) and logits["b"][0] == pytest.approx(3.065)
    assert timing.shared_prefix_tokens == 0 and timing.prefill_seconds == 0.0


def test_a_single_question_takes_one_pass_even_in_shared_mode():
    engine = backend()
    logits, timing = engine.score([100, 101], [job("only", [100, 101], [1, 2])], "shared")
    assert engine.session.calls == [
        ("clear",),
        ("decode", [100, 101, 1, 2], [0, 1, 2, 3], [0] * 4, [3]),
    ]
    assert logits["only"][0] == pytest.approx(3.065)
    assert timing.shared_prefix_tokens == 0 and timing.batches == 1


def test_inputs_longer_than_one_call_are_fed_in_slices(monkeypatch):
    monkeypatch.setattr(backend_llama, "N_BATCH", 4)
    prefix = list(range(10))
    jobs = [job("huge", prefix, list(range(50, 59))), job("tiny", prefix, [1])]
    engine = backend()
    logits, _ = engine.score(prefix, jobs, "shared")
    decodes = [c for c in engine.session.calls if c[0] == "decode"]
    assert [len(c[1]) for c in decodes] == [4, 4, 2, 1, 4, 4, 1]  # prefix, tiny, huge in slices
    assert [c[4] for c in decodes] == [[], [], [], [0], [], [], [0]]  # logits at the very end only
    assert logits["huge"][0] == pytest.approx(18.065)
    # A microbatch never exceeds one call: 3 + 3 tokens do not fit in 4.
    engine = backend()
    engine.score(prefix, [job("x", prefix, [1, 2, 3]), job("y", prefix, [4, 5, 6])], "shared")
    assert [len(c[1]) for c in engine.session.calls if c[0] == "decode"] == [4, 4, 2, 3, 3]


def test_score_rejects_bad_input():
    engine = backend()
    with pytest.raises(ValueError, match="mode"):
        engine.score([1], [job("a", [1], [2])], "batched")
    with pytest.raises(ValueError, match="No decisions"):
        engine.score([1], [], "shared")
    with pytest.raises(ValueError, match="prefix"):
        engine.score([1, 2], [job("a", [1, 3], [4])], "shared")
    with pytest.raises(ValueError, match="prefix"):
        engine.score([1, 2], [job("a", [1, 2], [])], "shared")
    with pytest.raises(ValueError, match="batch_size"):
        backend(batch_size=17)


def test_tokenizer_renders_like_transformers_and_encodes_with_the_gguf():
    template = (
        "{%- if not messages %}{{ raise_exception('No messages provided.') }}{%- endif %}\n"
        "{%- for m in messages %}\n"
        "    {{- '<' + m.role + '>' + m.content }}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}{{ '<bot>' }}"
        "{%- if not enable_thinking %}{{ '</think>' }}{%- endif %}{%- endif %}"
    )
    tokenizer = LlamaTokenizer(FakeSession(), template)
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert text == "<system>S<user>U<bot></think>"  # block lines and indentation leave no trace
    assert tokenizer.encode("AB", add_special_tokens=False) == [65, 66]
    assert tokenizer.pad_token_id is None and tokenizer.eos_token_id == 2
    with pytest.raises(ValueError, match="No messages"):
        tokenizer.apply_chat_template([], tokenize=False)
    with pytest.raises(ValueError, match="encode"):
        tokenizer.apply_chat_template([{"role": "user", "content": "U"}], tokenize=True)


CPU = Device(1, "CPU", "Some CPU", "cpu", "CPU", 64 << 30)
IGPU = Device(2, "Vulkan0", "Intel(R) UHD Graphics", "igpu", "Vulkan", 32 << 30)
RADEON = Device(3, "Vulkan1", "AMD Radeon RX 7900 XTX", "gpu", "Vulkan", 24 << 30)
SMALL = Device(4, "Vulkan2", "AMD Radeon RX 6600", "gpu", "Vulkan", 8 << 30)
GEFORCE = Device(5, "CUDA0", "NVIDIA GeForce RTX 5060 Ti", "gpu", "CUDA", 16 << 30)


def test_device_choice():
    machine = [CPU, IGPU, SMALL, RADEON]
    assert choose_device(machine, "auto") is RADEON  # discrete before integrated, then memory
    assert choose_device(machine, "gpu") is RADEON
    assert choose_device(machine, "cpu") is None
    assert choose_device([CPU, IGPU], "auto") is IGPU
    assert choose_device([CPU], "auto") is None  # no GPU: the CPU, without an error
    assert choose_device(machine, "vulkan") is RADEON
    assert choose_device([CPU, GEFORCE], "cuda") is GEFORCE
    with pytest.raises(ValueError, match="no such GPU"):
        choose_device([CPU], "gpu")  # an explicit request is never downgraded
    with pytest.raises(ValueError, match="rizzo download"):
        choose_device(machine, "cuda")


def test_loader_rejects_options_of_the_other_backend(tmp_path):
    with pytest.raises(ValueError, match="--quant"):
        loader.load_backend("llama", bits=8)
    with pytest.raises(ValueError, match="--bits"):
        loader.load_backend("mlx", quant="q8_0")
    with pytest.raises(ValueError, match="--backend mlx"):
        loader.load_backend("llama", device="mlx")
    with pytest.raises(ValueError, match="llama backend"):
        loader.load_backend("mlx", device="vulkan")
    with pytest.raises(ValueError, match=".gguf"):
        loader.load_backend("llama", model=tmp_path)
    with pytest.raises(ValueError, match="rizzo download"):
        loader.load_backend("llama", model=tmp_path / "missing.gguf")
    with pytest.raises(ValueError, match="one of"):
        loader.load_backend("onnx")
