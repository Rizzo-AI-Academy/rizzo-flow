"""Stand-ins that satisfy the protocols in `rizzo_flow.protocols` without any weights.

They live in their own module rather than inside a test file, so importing them does not
mean importing somebody else's tests.
"""

from typing import Any

from rizzo_flow.config import MODEL_ID
from rizzo_flow.metadata import LlamaMetadata, MlxMetadata
from rizzo_flow.prompts import PROMPT_VERSION, Compiled
from rizzo_flow.responses import Timing


class CharacterTokenizer:
    """Test tokenizer; deliberately distinct from real Spark tokenizer integration tests."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs["enable_thinking"] is False
        return "\n".join(m["content"] for m in messages) + "\nASSISTANT:"


class FakeBackend:
    tokenizer = CharacterTokenizer()

    def __init__(self) -> None:
        self.metadata = LlamaMetadata(
            source=MODEL_ID,
            requested_revision="test",
            gguf_source=None,
            gguf_revision=None,
            source_files={"fake.gguf": "0" * 64},
            precision="q8_0",
            device="cpu",
            backend="cpu",
            runtime="llama.cpp",
            llama_cpp_release="test",
            llama_cpp_commit="test",
            prompt_version=PROMPT_VERSION,
        )

    def score(
        self, prefix: list[int], jobs: list[Compiled], mode: str
    ) -> tuple[dict[str, list[float]], Timing]:
        logits = {j.id: [0, 10] + [0] * (len(j.slots) - 2) for j in jobs}
        return logits, Timing(
            inference_seconds=0.0,
            prefill_seconds=0.0,
            shared_prefix_tokens=len(prefix),
            evaluated_tokens_including_padding=sum(len(j.tokens) for j in jobs),
            logical_input_tokens=sum(len(j.tokens) for j in jobs),
            batches=1,
            generated_tokens=0,
        )


def throwaway_mlx_metadata() -> MlxMetadata:
    """Identity for tests that build an MLX backend but never report on it."""
    return MlxMetadata(
        source=MODEL_ID,
        requested_revision="test",
        runtime_revision="test",
        source_files={"fake.safetensors": "0" * 64},
        precision="bf16",
        quantization_group_size=None,
        device="cpu",
        backend="cpu",
        mlx="test",
        mlx_lm="test",
        prompt_version=PROMPT_VERSION,
    )
