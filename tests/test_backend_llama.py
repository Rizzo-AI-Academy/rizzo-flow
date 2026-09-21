"""LlamaBackend: runtime identity, provenance and scoring.

Unit paths use a fake runtime; the real GGUF is exercised by the opt-in `llama` integration
test at the bottom.
"""

import hashlib
from pathlib import Path

import pytest

from rizzo_flow.backend_llama import LlamaBackend, sha256_file
from rizzo_flow.config import GGUF_MODELS
from rizzo_flow.llama_runtime import LLAMA_COMMIT
from rizzo_flow.prompts import PROMPT_VERSION

GGUF_NAME = Path(GGUF_MODELS["q8_0"].file)
PIN = GGUF_MODELS["q8_0"]


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
