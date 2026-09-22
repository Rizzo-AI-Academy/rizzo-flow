"""Real Spark computation/cache tests with small random weights, no model download."""

import importlib.util
from typing import Any

import pytest
from fakes import throwaway_mlx_metadata

pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(
        importlib.util.find_spec("spark_mlx_llm") is None, reason="Install the mlx extra"
    ),
]


def tiny_model(bits: int | None = None) -> Any:
    import mlx.core as mx
    from spark_mlx_llm.model import Model, ModelArgs

    from rizzo_flow.backend import quantize_model

    mx.random.seed(42)
    args = ModelArgs(
        model_type="spark2_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=128,
        sliding_window=16,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1},
            "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
        },
    )
    model = Model(args)
    if bits:
        quantize_model(model, bits)
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("bits", [None, 4, 8])
def test_selected_projection_equals_full_vocabulary(bits: int | None) -> None:
    import mlx.core as mx

    from rizzo_flow.backend import selected_logits

    model = tiny_model(bits)
    tokens = mx.array([[4, 5, 6]])
    hidden = model.model(tokens)[:, -1, :]
    expected = model(tokens)[:, -1, [10, 11, 12]]
    actual = selected_logits(model, hidden, [10, 11, 12])
    assert mx.max(mx.abs(expected - actual)).item() < 1e-4


@pytest.mark.parametrize("prefix_length", [5, 16, 49])
def test_shared_padding_rotating_cache_and_repeat_are_equivalent(prefix_length: int) -> None:
    from rizzo_flow.backend import SparkBackend
    from rizzo_flow.prompts import Compiled

    class Tokenizer:
        pad_token_id = 0

    backend = SparkBackend(
        tiny_model(), Tokenizer(), throwaway_mlx_metadata(), batch_size=3, prefill_chunk=7
    )
    prefix = [4, 5, 6, 7, 8] * (prefix_length // 5 + 1)
    prefix = prefix[:prefix_length]
    jobs = [
        Compiled(str(i), prefix + [9, 10, 11] * n, [20, 21, 22], "test")
        for i, n in enumerate([1, 7, 3])
    ]
    direct, _ = backend.score(prefix, jobs, "direct")
    shared, timing = backend.score(prefix, jobs, "shared")
    repeated, _ = backend.score(prefix, list(reversed(jobs)), "shared")
    for key in direct:
        assert shared[key] == pytest.approx(direct[key], abs=2e-4)
        assert repeated[key] == pytest.approx(shared[key], abs=2e-4)
    assert timing.generated_tokens == 0 and timing.peak_mlx_bytes is not None


def test_branch_does_not_mutate_retained_prefix() -> None:
    import mlx.core as mx

    from rizzo_flow.backend import SparkBackend, branch_cache

    model = tiny_model()
    backend = SparkBackend(model, None, throwaway_mlx_metadata(), prefill_chunk=8)
    prefix = backend._prefill([4, 5, 6] * 20)
    before = [[a.tolist() for a in c.state] for c in prefix]
    offsets = [c.offset for c in prefix]
    branch = branch_cache(prefix, 2)
    output = model(mx.array([[7, 8], [9, 10]]), cache=branch)
    mx.eval(output)
    assert [c.offset for c in prefix] == offsets
    assert [[a.tolist() for a in c.state] for c in prefix] == before
