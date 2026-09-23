"""What this package's moving parts must be able to do, stated structurally.

These are `typing.Protocol` classes: a backend, a tokenizer or a llama.cpp session
satisfies one by having the right members, not by inheriting anything. That puts the
contract of `Engine` in code instead of in the prose of a module docstring, and it holds
the real implementations and the test fakes to the same shape without a base class
between them.

Deliberately not `@runtime_checkable`: `isinstance()` against a Protocol only checks that
the names exist, never that the signatures match, which would buy a false sense of safety.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .llama_cpp import Device
    from .metadata import Metadata
    from .prompts import Compiled
    from .responses import Timing


class Tokenizer(Protocol):
    """The two calls `prompts.compile_request` makes, plus the padding token MLX needs."""

    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def apply_chat_template(self, messages: list[dict[str, str]], **variables: Any) -> str: ...


class ScoringBackend(Protocol):
    """One resident model that reads answer-letter logits. Never generates text."""

    tokenizer: Tokenizer
    metadata: Metadata

    def score(
        self, prefix: list[int], jobs: list[Compiled], mode: str = "shared"
    ) -> tuple[dict[str, list[float]], Timing]: ...


class LlamaSession(Protocol):
    """The llama.cpp context as `LlamaBackend` uses it once the model is loaded."""

    device: Device | None
    idle_free: int | None
    pad_token: int | None
    eos_token: int | None

    def tokenize(self, text: str, add_special: bool = False) -> list[int]: ...

    def decode(
        self,
        tokens: list[int],
        positions: Any,
        sequences: list[int],
        outputs: Any = (),
    ) -> None: ...

    def logits(self, index: int, slots: list[int]) -> list[float]: ...

    def clear(self) -> None: ...

    def branch(self, source: int, target: int) -> None: ...

    def drop(self, sequence: int) -> None: ...

    def synchronize(self) -> None: ...

    def free_bytes(self) -> int | None: ...


class ProgressCallback(Protocol):
    """Reports bytes of one download; `total` is 0 when the server does not say."""

    def __call__(self, name: str, done: int, total: int) -> None: ...
