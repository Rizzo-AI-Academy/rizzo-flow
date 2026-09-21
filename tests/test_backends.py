"""Backend selection and option validation; no model, no library and no GPU required."""

from pathlib import Path

import pytest

from rizzo_flow import backends, config


def test_select_backend_prefers_a_gguf_then_mlx_then_llama():
    assert backends.select_backend("auto", Path("custom.gguf"), available=[]) == "llama"
    assert backends.select_backend("auto", None, available=["mlx"]) == "mlx"
    assert backends.select_backend("auto", None, available=["llama"]) == "llama"
    assert backends.select_backend("auto", None, available=["llama", "mlx"]) == "mlx"
    assert backends.select_backend("llama", None, available=[]) == "llama"
    assert backends.select_backend("mlx", None, available=[]) == "mlx"


def test_select_backend_rejects_unknown_and_unavailable():
    with pytest.raises(ValueError, match="one of"):
        backends.select_backend("tpu", None, available=["mlx"])
    with pytest.raises(ValueError, match="No backend"):
        backends.select_backend("auto", None, available=[])


def test_device_names_are_validated_per_backend():
    backends.ensure_options("llama", bits=None, device="vulkan")
    backends.ensure_options("llama", bits=None, device="cpu")
    backends.ensure_options("mlx", bits=8, device="cuda")
    with pytest.raises(ValueError, match="does not apply"):
        backends.ensure_options("mlx", bits=None, device="vulkan")
    with pytest.raises(ValueError, match="does not apply"):
        backends.ensure_options("llama", bits=None, device="mlx")


def test_llama_rejects_bits_before_loading_anything():
    # `--bits` is in-memory MLX quantization; llama.cpp wants an already quantized artifact.
    with pytest.raises(ValueError, match="bits"):
        backends.load_backend("llama", gguf=Path("absent.gguf"), bits=8)


def test_quant_is_rejected_for_the_mlx_backend():
    with pytest.raises(ValueError, match="quant"):
        backends.load_backend("mlx", quant="bf16")


def test_download_rejects_an_unknown_quant():
    with pytest.raises(ValueError, match="Unknown quant"):
        config.download_gguf("q9_9")


def test_pinned_quants_expose_path_and_digest():
    for quant, spec in config.GGUF_MODELS.items():
        assert len(spec.sha256) == 64
        assert config.gguf_path(quant).name == spec.file
        assert config.find_gguf_pin(spec.file) is spec


def test_model_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(config.MODEL_DIR_ENV, str(tmp_path))
    assert config.gguf_path("q8_0").parent == tmp_path


def test_describe_reports_both_stacks_without_importing_mlx():
    report = backends.describe()
    assert set(report) >= {"available", "llama", "mlx"}
    assert report["llama"]["devices"] == list(backends.LLAMA_DEVICES)
    assert isinstance(report["llama"]["present"], bool)
    assert report["mlx"]["available"] in (True, False)
