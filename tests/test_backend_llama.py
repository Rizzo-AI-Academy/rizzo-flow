"""LlamaBackend: runtime identity, provenance and scoring.

Unit paths use a fake runtime; the real GGUF is exercised by the opt-in `llama` integration
test at the bottom.
"""

import hashlib
from pathlib import Path

import pytest

from rizzo_flow.backend_llama import LlamaBackend, library_names, sha256_file
from rizzo_flow.config import GGUF_MODELS
from rizzo_flow.llama_runtime import LLAMA_COMMIT
from rizzo_flow.prompts import PROMPT_VERSION, Compiled, compile_request
from rizzo_flow.schema import Request

GGUF_NAME = Path(GGUF_MODELS["q8_0"].file)
PIN = GGUF_MODELS["q8_0"]


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [("linux", "libllama.so"), ("darwin", "libllama.dylib"), ("win32", "llama.dll")],
)
def test_library_names_follow_the_platform(platform_name, expected):
    """A hardcoded `.so` hid the backend on Windows and macOS, where the file is not a .so."""
    assert library_names(platform_name) == (expected,)


class FakeRuntime:
    """Minimal stand-in exposing only what the identity builder reads."""

    def __init__(
        self, hidden=2560, architecture="spark2_5", file_type_name="Q8_0", backend="vulkan"
    ):
        self.hidden = hidden
        self.architecture = architecture
        self.file_type_name = file_type_name
        self.compute_backend = backend
        self.device = "cpu" if backend == "cpu" else "gpu"
        self.n_seq_max = 8

    def metadata(self):
        return {
            "general.architecture": self.architecture,
            "spark2_5.embedding_length": str(self.hidden),
            "spark2_5.context_length": "1048576",
            "general.file_type": "7",
        }

    def chat_template(self):
        return "TEMPLATE"

    def tokenize(self, text, *, add_special=False, parse_special=True):
        return [ord(character) for character in text]

    pad_token_id = 0
    eos_token_id = 1


def make_backend(**overrides):
    arguments = {
        "gguf": GGUF_NAME,
        "gguf_sha256": "a" * 64,
        "batch_size": 4,
        "prefill_chunk": 512,
    }
    arguments.update(overrides)
    runtime = arguments.pop("runtime", None) or FakeRuntime()
    return LlamaBackend.from_runtime(runtime, **arguments)


def test_identity_records_both_provenance_layers():
    metadata = make_backend().metadata
    assert metadata["backend"] == "llama"
    assert metadata["compute_backend"] == "vulkan"
    assert metadata["device"] == "gpu"
    # model identity comes from the original checkpoint, the artifact from the GGUF pin
    assert metadata["source"] == "XHToken/Spark-X2.5-4B"
    assert metadata["requested_revision"] == "0bcb35678590218655dff3765b9e61c83b35e9c4"
    assert metadata["gguf_source"] == PIN.repo
    assert metadata["gguf_revision"] == PIN.revision
    assert metadata["gguf_file"] == PIN.file
    assert metadata["gguf_sha256"] == "a" * 64
    assert metadata["architecture"] == "spark2_5"
    assert metadata["context_length"] == 1048576
    assert metadata["precision"] == "q8_0"
    assert metadata["llama_cpp_commit"] == LLAMA_COMMIT
    assert metadata["prompt_version"] == PROMPT_VERSION


def test_fingerprint_is_deterministic_and_tracks_the_runtime():
    first = make_backend().metadata["fingerprint"]
    assert len(first) == 64 and first == make_backend().metadata["fingerprint"]
    assert first != make_backend(runtime=FakeRuntime(backend="cpu")).metadata["fingerprint"]
    assert first != make_backend(gguf_sha256="b" * 64).metadata["fingerprint"]
    assert first != make_backend(runtime=FakeRuntime(file_type_name="BF16")).metadata["fingerprint"]


def test_unknown_architecture_or_size_is_rejected():
    with pytest.raises(ValueError, match="Spark2.5"):
        make_backend(runtime=FakeRuntime(architecture="llama"))
    with pytest.raises(ValueError, match="Unrecognized"):
        make_backend(runtime=FakeRuntime(hidden=999))
    with pytest.raises(ValueError, match="embedding_length"):
        make_backend(runtime=FakeRuntime(hidden=None))


def test_batch_limits_are_enforced_like_the_mlx_backend():
    with pytest.raises(ValueError, match="batch_size"):
        make_backend(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        make_backend(prefill_chunk=4096)


def test_runtime_must_allow_one_sequence_per_branch():
    class TooFewSequences(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.n_seq_max = 2

    # llama_memory_seq_cp aborts the process when the destination sequence is out of range,
    # so the backend refuses the combination instead of letting llama.cpp die.
    with pytest.raises(ValueError, match="sequences"):
        make_backend(runtime=TooFewSequences(), batch_size=4)


def test_unpinned_gguf_keeps_the_artifact_fields_empty_but_valid():
    metadata = make_backend(gguf=Path("private-custom.gguf")).metadata
    assert metadata["gguf_source"] is None and metadata["gguf_revision"] is None
    assert metadata["gguf_file"] == "private-custom.gguf"
    assert metadata["source"] == "XHToken/Spark-X2.5-4B"


def test_sha256_file_matches_hashlib(tmp_path):
    payload = b"spark" * 1000
    target = tmp_path / "weights.gguf"
    target.write_bytes(payload)
    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_load_rejects_a_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        LlamaBackend.load(tmp_path / "absent.gguf", library=tmp_path)


@pytest.mark.llama
def test_real_gguf_identity(llama_runtime):
    backend = LlamaBackend.from_runtime(
        llama_runtime, gguf=GGUF_NAME, gguf_sha256="a" * 64, load_seconds=1.0
    )
    metadata = backend.metadata
    assert metadata["architecture"] == "spark2_5"
    assert metadata["source"] == "XHToken/Spark-X2.5-4B"
    assert metadata["precision"] == "q8_0"
    assert metadata["gguf_source"] == PIN.repo
    assert metadata["context_length"] == 1048576
    assert metadata["load_seconds"] == 1.0
    assert backend.tokenizer.encode("A") == [46]


# --- scoring ------------------------------------------------------------------------------


class ScoreRuntime(FakeRuntime):
    """Records the decode plan and returns placeholder logits without touching a GPU."""

    def __init__(self, n_ctx=4096, **overrides):
        super().__init__(**overrides)
        self.n_ctx = n_ctx
        self.calls = []

    def memory_clear(self):
        self.calls.append(("clear",))

    def seq_copy(self, source, destination):
        self.calls.append(("cp", source, destination))

    def seq_remove(self, seq_id):
        self.calls.append(("rm", seq_id))

    def decode(self, spec):
        self.calls.append(
            ("decode", tuple(spec.seq_ids), tuple(spec.positions), tuple(sorted(spec.logits_at)))
        )

    def logits(self, index, slots):
        self.calls.append(("logits", index))
        return [float(index) + offset for offset in range(len(slots))]


def make_score_backend(runtime, batch_size=4, prefill_chunk=4):
    return LlamaBackend(
        runtime, tokenizer=None, metadata={}, batch_size=batch_size, prefill_chunk=prefill_chunk
    )


def compiled(identifier, tokens, slots=(10, 11)):
    return Compiled(identifier, tokens, list(slots), "test")


def test_direct_rebuilds_the_cache_for_every_job():
    runtime = ScoreRuntime()
    backend = make_score_backend(runtime, prefill_chunk=2)
    jobs = [compiled("a", [1, 2, 3]), compiled("b", [4, 5])]
    result, timing = backend.score([], jobs, "direct")
    assert set(result) == {"a", "b"}
    assert [call for call in runtime.calls if call[0] == "clear"] == [("clear",), ("clear",)]
    assert timing["shared_prefix_tokens"] == 0
    assert timing["batches"] == 4  # two jobs: one chunk plus the last token each
    assert timing["generated_tokens"] == 0
    assert timing["logical_input_tokens"] == 5


def test_shared_prefills_once_and_branches_per_group():
    runtime = ScoreRuntime(n_ctx=64)
    backend = make_score_backend(runtime, batch_size=2, prefill_chunk=4)
    prefix = [1, 2, 3]
    jobs = [compiled(str(index), prefix + [10 + index, 20 + index]) for index in range(3)]
    result, timing = backend.score(prefix, jobs, "shared")
    assert set(result) == {"0", "1", "2"}
    assert [call for call in runtime.calls if call[0] == "clear"] == [("clear",)]
    assert [call for call in runtime.calls if call[0] == "cp"] == [
        ("cp", 0, 1),
        ("cp", 0, 2),
        ("cp", 0, 1),
    ]
    assert [call for call in runtime.calls if call[0] == "rm"] == [("rm", 1), ("rm", 2), ("rm", 1)]
    # the shared prefix is decoded once and logits are never requested for it
    decodes = [call for call in runtime.calls if call[0] == "decode"]
    assert decodes[0] == ("decode", (0, 0, 0), (0, 1, 2), ())
    assert all(call[3] for call in decodes[1:])
    assert timing["shared_prefix_tokens"] == 3
    assert timing["batches"] == 3  # one prefix chunk plus two microbatch groups
    assert timing["evaluated_tokens_including_padding"] == 3 + 6


def test_shared_groups_respect_the_context_capacity():
    runtime = ScoreRuntime(n_ctx=10)
    backend = make_score_backend(runtime, batch_size=4, prefill_chunk=8)
    prefix = [1, 2]
    jobs = [compiled(str(index), prefix + [7, 8, 9, 10, 11, 12]) for index in range(3)]
    backend.score(prefix, jobs, "shared")
    groups = [call for call in runtime.calls if call[0] == "cp"]
    # 6-token suffixes cannot share a group under a 10-token context: one branch at a time
    assert groups == [("cp", 0, 1), ("cp", 0, 1), ("cp", 0, 1)]


def test_score_validates_inputs():
    backend = make_score_backend(ScoreRuntime())
    with pytest.raises(ValueError, match="Unknown execution mode"):
        backend.score([], [compiled("a", [1, 2])], "batch")
    with pytest.raises(ValueError, match="No decisions"):
        backend.score([], [], "shared")
    with pytest.raises(ValueError, match="Invalid shared prefix"):
        backend.score([9, 9], [compiled("a", [1, 2, 3])], "shared")
    with pytest.raises(ValueError, match="Invalid shared prefix"):
        backend.score([1], [compiled("a", [1])], "shared")


@pytest.mark.llama
def test_shared_and_direct_agree_on_the_ticket(llama_runtime):
    backend = LlamaBackend.from_runtime(llama_runtime, gguf=GGUF_NAME, gguf_sha256="a" * 64)
    request = Request.model_validate_json(Path("examples/ticket.json").read_text(encoding="utf-8"))
    prefix, jobs = compile_request(backend.tokenizer, request, 8192)
    shared, shared_timing = backend.score(prefix, jobs, "shared")
    direct, direct_timing = backend.score(prefix, jobs, "direct")
    for job in jobs:
        winner = max(range(len(shared[job.id])), key=shared[job.id].__getitem__)
        other = max(range(len(direct[job.id])), key=direct[job.id].__getitem__)
        assert winner == other, f"{job.id}: shared={shared[job.id]} direct={direct[job.id]}"
    assert shared_timing["shared_prefix_tokens"] == len(prefix)
    assert direct_timing["shared_prefix_tokens"] == 0
    assert shared_timing["batches"] <= direct_timing["batches"]
