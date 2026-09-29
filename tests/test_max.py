"""The MAX graphs against the MLX reference: the same small random Spark, no model download.

MLX's `spark_mlx_llm` is the implementation the checkpoint was validated with, so its logits are
the reference. Both run in float32 on the CPU (`uv sync --extra max --extra cpu` on Linux).
The first run compiles the graphs, which takes about a minute on the CPU.
"""

import importlib.util

import pytest

pytestmark = [
    pytest.mark.max,
    pytest.mark.skipif(
        importlib.util.find_spec("max") is None
        or importlib.util.find_spec("spark_mlx_llm") is None,
        reason="Install the max extra and an MLX extra",
    ),
]

WINDOW = 16
SLOTS = [20, 21, 22, 90]


@pytest.fixture(scope="module")
def reference():
    import mlx.core as mx
    from spark_mlx_llm.model import Model, ModelArgs

    mx.random.seed(42)
    config = {
        "model_type": "spark2_5",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 128,
        "sliding_window": WINDOW,
        "rms_norm_eps": 1e-6,
        "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1},
            "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
        },
    }
    model = Model(ModelArgs(**config))
    model.eval()
    mx.eval(model.parameters())
    return config, model


@pytest.fixture(scope="module")
def backend(reference):
    import numpy as np
    from max.driver import CPU
    from max.dtype import DType
    from mlx.utils import tree_flatten

    from rizzo_flow.backend_max import MaxBackend, SparkRunner
    from rizzo_flow.max_spark import SparkConfig

    config, model = reference
    weights = {
        name: np.array(value, dtype=np.float32) for name, value in tree_flatten(model.parameters())
    }

    class Tokenizer:
        pad_token_id = 0

    # Blocks of 8 so that the small model still crosses chunk and bucket boundaries.
    runner = SparkRunner(
        SparkConfig.from_hf(config),
        weights,
        CPU(),
        DType.float32,
        batch_size=3,
        chunk=8,
        block=8,
        min_view=8,
    )
    return MaxBackend(runner, Tokenizer(), {}, batch_size=3, prefill_chunk=8)


def mlx_logits(model, tokens):
    import mlx.core as mx

    logits = model(mx.array([tokens]))[0, -1, SLOTS]
    return logits.astype(mx.float32).tolist()


@pytest.mark.parametrize("length", [1, 5, WINDOW, 3 * WINDOW + 5])
def test_direct_matches_mlx(reference, backend, length):
    from rizzo_flow.prompts import Compiled

    tokens = [(7 * i + 3) % 128 for i in range(length)]
    result, timing = backend.score([], [Compiled("q", tokens, SLOTS, "test")], "direct")
    assert result["q"] == pytest.approx(mlx_logits(reference[1], tokens), abs=2e-4)
    assert timing["generated_tokens"] == 0


@pytest.mark.parametrize("prefix_length", [5, WINDOW, 49])
def test_shared_padded_branches_match_mlx(reference, backend, prefix_length):
    from rizzo_flow.prompts import Compiled

    prefix = ([4, 5, 6, 7, 8] * 20)[:prefix_length]
    jobs = [
        Compiled(str(i), prefix + [9, 10, 11] * n, SLOTS, "test") for i, n in enumerate([1, 7, 3])
    ]
    shared, timing = backend.score(prefix, jobs, "shared")
    repeated, _ = backend.score(prefix, list(reversed(jobs)), "shared")
    for job in jobs:
        expected = mlx_logits(reference[1], job.tokens)
        assert shared[job.id] == pytest.approx(expected, abs=2e-4)
        assert repeated[job.id] == pytest.approx(shared[job.id], abs=1e-5)
    assert timing["shared_prefix_tokens"] == prefix_length
    assert timing["batches"] == 1


def test_loader_rejects_options_of_other_backends(tmp_path):
    from rizzo_flow import loader

    with pytest.raises(ValueError, match="--quant"):
        loader.load_backend("max", quant="q8_0")
    with pytest.raises(ValueError, match="--quant"):
        loader.load_backend("max", bits=8)
    with pytest.raises(ValueError, match="auto, gpu or cpu"):
        loader.load_backend("max", device="vulkan")
    with pytest.raises(ValueError, match="--kv-type"):
        loader.load_backend("max", kv_type="q8_0")
    with pytest.raises(ValueError, match="--max-resident"):
        loader.load_backend("llama", max_resident=4)
    with pytest.raises(ValueError, match="--weights"):
        loader.load_backend("max", model=tmp_path, weights="flow")
    with pytest.raises(ValueError, match="Model not found"):
        loader.load_backend("max", model=tmp_path / "missing")


def test_runner_rejects_weights_that_do_not_match_the_config(reference):
    import numpy as np
    from max.driver import CPU
    from max.dtype import DType
    from mlx.utils import tree_flatten

    from rizzo_flow.backend_max import SparkRunner
    from rizzo_flow.max_spark import SparkConfig

    config, model = reference
    weights = {n: np.array(v, dtype=np.float32) for n, v in tree_flatten(model.parameters())}
    spark = SparkConfig.from_hf(config)
    with pytest.raises(ValueError, match="the graphs expect"):
        SparkRunner(spark, weights, CPU(), DType.bfloat16)
    weights["model.norm.weight"] = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError, match="the config implies"):
        SparkRunner(spark, weights, CPU(), DType.float32)
    del weights["model.norm.weight"]
    with pytest.raises(ValueError, match="lacks 1 tensors"):
        SparkRunner(spark, weights, CPU(), DType.float32)


def test_long_suffixes_are_split_under_the_score_token_cap(reference, backend):
    """Microbatches shrink, and a suffix alone over the cap takes the chunked prefill; the
    logits do not change."""
    from rizzo_flow.prompts import Compiled

    prefix = [4, 5, 6, 7, 8] * 4
    jobs = [
        Compiled(str(i), prefix + [9, 10, 11] * n, SLOTS, "test") for i, n in enumerate([1, 7, 3])
    ]
    backend.runner.score_tokens, cap = 24, backend.runner.score_tokens
    try:
        groups, alone = backend._groups(sorted(jobs, key=lambda j: len(j.tokens)), len(prefix))
        assert [j.id for j in alone] == ["1"] and [[j.id for j in g] for g in groups] == [
            ["0"],
            ["2"],
        ]
        result, _ = backend.score(prefix, jobs, "shared")
    finally:
        backend.runner.score_tokens = cap
    for job in jobs:
        assert result[job.id] == pytest.approx(mlx_logits(reference[1], job.tokens), abs=2e-4)
