"""GGUF chat template rendering and the tokenizer contract `Engine` relies on.

The template is the Spark2.5 one shipped with XHToken/Spark-X2.5-4B; in production it comes
from the GGUF, but keeping a copy here lets the rendering be tested without weights.
"""

import string
from pathlib import Path

import pytest

from rizzo_flow.llama_tokenizer import LlamaTokenizer
from rizzo_flow.prompts import compile_request
from rizzo_flow.schema import Request

TEMPLATE = (Path(__file__).parent / "data" / "spark2_5-chat-template.jinja").read_text(
    encoding="utf-8"
)
SYSTEM = "You are a precise decision function."
USER = "<evidence>\nhe is an engineer\n</evidence>\n\nQuestion: seniority?\n\nA. Junior"
MESSAGES = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
# The template chain is deterministic: system block, user turn, then the generation prompt.
# With enable_thinking=false it closes the think block immediately (see CLAUDE.md).
PREFIX = (
    "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.\n\n"
    + SYSTEM
    + "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>"
    + USER
    + "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>"
)


class FakeVocab:
    """Character-level stand-in for the llama vocabulary; keeps the tests weight-free."""

    pad_token_id = 0
    eos_token_id = 1

    def tokenize(self, text, *, add_special=False, parse_special=True):
        return [ord(character) for character in text]


def make_tokenizer():
    return LlamaTokenizer(FakeVocab(), TEMPLATE)


def test_apply_chat_template_matches_the_spark_layout_without_thinking():
    rendered = make_tokenizer().apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert rendered == PREFIX + "</think>"


def test_apply_chat_template_thinking_opens_the_block():
    rendered = make_tokenizer().apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    assert rendered == PREFIX + "<think>"


def test_apply_chat_template_rejects_tokenize_true():
    with pytest.raises(NotImplementedError, match="tokenize"):
        make_tokenizer().apply_chat_template(MESSAGES, tokenize=True)


def test_apply_chat_template_rejects_unknown_arguments():
    with pytest.raises(ValueError, match="Unsupported template argument"):
        make_tokenizer().apply_chat_template(MESSAGES, tools=[{"type": "function"}])


def test_encode_forwards_the_special_token_flag():
    tokenizer = make_tokenizer()
    assert tokenizer.encode("AB", add_special_tokens=False) == [65, 66]
    assert tokenizer.encode("AB") == [65, 66]
    assert tokenizer.pad_token_id == 0 and tokenizer.eos_token_id == 1


def test_from_runtime_requires_a_template():
    class WithoutTemplate(FakeVocab):
        def chat_template(self):
            return None

    with pytest.raises(ValueError, match="chat template"):
        LlamaTokenizer.from_runtime(WithoutTemplate())

    class WithTemplate(FakeVocab):
        def chat_template(self):
            return TEMPLATE

    tokenizer = LlamaTokenizer.from_runtime(WithTemplate())
    assert tokenizer.encode("A") == [65]


def test_compile_request_works_through_the_llama_tokenizer():
    tokenizer = make_tokenizer()
    request = Request.model_validate(
        {
            "state": {"ticket": "Cannot log in"},
            "questions": {
                "supported": {
                    "type": "boolean",
                    "instructions": "Does the user need login help?",
                    "policy": {"allow_abstain": False},
                },
                "route": {
                    "type": "choice",
                    "instructions": "Choose a queue",
                    "policy": {"allow_abstain": False},
                    "options": [
                        {"id": "billing", "description": "Payment problem"},
                        {"id": "access", "description": "Login problem"},
                    ],
                },
            },
        }
    )
    prefix, jobs = compile_request(tokenizer, request, 8192)
    assert prefix
    assert all(job.tokens[: len(prefix)] == prefix for job in jobs)
    assert [len(job.slots) for job in jobs] == [2, 2]
    # Slots are the single tokens for A and B, exactly as the prompt labels them.
    assert jobs[1].slots == [65, 66]


@pytest.mark.llama
def test_real_gguf_template_renders_the_same_prompt(llama_runtime):
    """The template inside the GGUF and the vendored copy must render identically."""
    from_gguf = LlamaTokenizer.from_runtime(llama_runtime)
    from_fixture = LlamaTokenizer(llama_runtime, TEMPLATE)
    arguments = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    rendered = from_gguf.apply_chat_template(MESSAGES, **arguments)
    assert rendered == from_fixture.apply_chat_template(MESSAGES, **arguments)
    assert rendered == PREFIX + "</think>"
    assert all(len(from_gguf.encode(letter)) == 1 for letter in string.ascii_uppercase)
