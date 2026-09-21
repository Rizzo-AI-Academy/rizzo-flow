"""Chat template and tokenizer adapter for the llama.cpp backend.

The prompt must stay byte-identical to the one the MLX backend compiled for
`spark-decisions-v3`, so the template is rendered with the same Jinja settings transformers
uses for `apply_chat_template` (sandboxed environment, trim/lstrip blocks, loopcontrols):
https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py
The template itself always comes from the loaded GGUF (`llama_model_chat_template`), so the
GGUF remains the single source of truth; no template is hardcoded here.
"""


def _environment():
    """Jinja environment equivalent to the one HF uses, with the Spark-specific helper."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    environment.globals["raise_exception"] = raise_exception
    return environment


class LlamaTokenizer:
    """Presents the same surface `prompts.compile_request` expects from an MLX tokenizer."""

    def __init__(self, runtime, template: str):
        if not getattr(runtime, "tokenize", None):
            raise ValueError("LlamaTokenizer needs a runtime with tokenize()")
        if not template:
            raise ValueError("LlamaTokenizer needs a non-empty chat template")
        self._runtime = runtime
        self._template = template
        self._compiled = _environment().from_string(template)

    @classmethod
    def from_runtime(cls, runtime) -> "LlamaTokenizer":
        template = runtime.chat_template()
        if not template:
            raise ValueError("The GGUF does not contain a chat template")
        return cls(runtime, template)

    # --- the contract used by prompts.py --------------------------------------------------

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        **kwargs,
    ) -> str:
        if tokenize:
            raise NotImplementedError("LlamaTokenizer renders text only; use tokenize=False")
        if kwargs:
            raise ValueError(f"Unsupported template argument(s): {', '.join(sorted(kwargs))}")
        return self._compiled.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._runtime.tokenize(text, add_special=add_special_tokens, parse_special=True)

    @property
    def pad_token_id(self):
        return self._runtime.pad_token_id

    @property
    def eos_token_id(self):
        return self._runtime.eos_token_id

    def decode(self, tokens: list[int]) -> str:
        """Inverse of `encode`, for diagnostics and tests only."""
        return "".join(self._runtime.token_to_piece(token) for token in tokens)
