"""Backend selection and option validation; no model, no library and no GPU required.

Routing is tested with injected hardware and capability maps, so the NVIDIA/AMD/Apple
decisions are covered on a machine that only has one of them.
"""

from pathlib import Path

import pytest

from rizzo_flow import backend_llama, backends, config
from rizzo_flow.backends import Routing, announce, route

# Capability maps shaped like the ones the real probes return.
MLX_APPLE = {"apple": "mlx", "cpu": "cpu"}
MLX_CUDA = {"nvidia": "cuda", "cpu": "cpu"}
MLX_CPU_ONLY = {"cpu": "cpu"}  # the package imports but has no usable GPU
LLAMA_VULKAN = {"nvidia": "vulkan", "amd": "vulkan", "intel": "vulkan", "cpu": "cpu"}
LLAMA_CUDA = {"nvidia": "cuda", "cpu": "cpu"}
LLAMA_CPU_ONLY = {"cpu": "cpu"}


def pick(device="auto", *, hardware=("amd", "cpu"), mlx_caps=None, llama_caps=None, **kwargs):
    return route(
        device,
        hardware_present=list(hardware),
        mlx_caps=MLX_CPU_ONLY if mlx_caps is None else mlx_caps,
        llama_caps=LLAMA_VULKAN if llama_caps is None else llama_caps,
        **kwargs,
    )


# --- availability probes ------------------------------------------------------------------


def test_llama_present_reads_the_directory_it_is_given(tmp_path):
    (tmp_path / backend_llama.library_names()[0]).touch()
    assert backends.llama_present(tmp_path) is True
    assert backends.llama_present(tmp_path / "absent") is False


# --- auto routing -------------------------------------------------------------------------


def test_auto_uses_mlx_on_apple_silicon():
    assert pick(hardware=["apple"], mlx_caps=MLX_APPLE).backend == "mlx"
    assert pick(hardware=["apple"], mlx_caps=MLX_APPLE).device == "mlx"


def test_auto_prefers_a_usable_mlx_gpu_on_nvidia():
    # Documented default: MLX-CUDA first when it is installed *and* has an accelerator.
    chosen = pick(hardware=["nvidia"], mlx_caps=MLX_CUDA, llama_caps=LLAMA_CUDA)
    assert (chosen.backend, chosen.device) == ("mlx", "cuda")


def test_auto_uses_llama_cuda_when_mlx_has_no_gpu():
    # An NVIDIA llama.cpp build ships libggml-cuda.so; missing it sent the box to CPU.
    chosen = pick(hardware=["nvidia"], mlx_caps={}, llama_caps=LLAMA_CUDA)
    assert (chosen.backend, chosen.device) == ("llama", "cuda")


def test_auto_ignores_an_importable_but_cpu_only_mlx_on_amd():
    # The headline bug: `mlx-cpu` imports, is picked first, and takes minutes per decision
    # while a working Vulkan backend sits one branch away.
    chosen = pick(hardware=["amd"], mlx_caps=MLX_CPU_ONLY, llama_caps=LLAMA_VULKAN)
    assert (chosen.backend, chosen.device) == ("llama", "vulkan")


def test_auto_reports_a_cpu_downgrade_instead_of_hiding_it():
    chosen = pick(hardware=["cpu"], mlx_caps=MLX_CPU_ONLY, llama_caps=LLAMA_VULKAN)
    assert (chosen.backend, chosen.device) == ("llama", "cpu")
    assert chosen.downgraded is True
    assert "cpu" in chosen.reason.lower()


def test_auto_leaves_an_explicit_cpu_alone():
    chosen = pick("cpu")
    assert (chosen.backend, chosen.device) == ("llama", "cpu")
    assert chosen.downgraded is False


def test_auto_prefers_apple_over_a_second_accelerator():
    chosen = pick(hardware=["apple", "amd"], mlx_caps=MLX_APPLE, llama_caps=LLAMA_VULKAN)
    assert (chosen.backend, chosen.device) == ("mlx", "mlx")


# --- explicit preferences -----------------------------------------------------------------


def test_gpu_pin_errors_instead_of_downgrading():
    with pytest.raises(ValueError, match="No backend"):
        pick("gpu", hardware=["cpu"], mlx_caps=MLX_CPU_ONLY)


def test_vendor_pin_errors_when_that_vendor_is_absent():
    with pytest.raises(ValueError, match="nvidia"):
        pick("nvidia", hardware=["amd"], mlx_caps={})


def test_backend_pin_restricts_the_stack():
    chosen = pick(hardware=["nvidia"], mlx_caps=MLX_CUDA, llama_caps=LLAMA_CUDA, backend="llama")
    assert (chosen.backend, chosen.device) == ("llama", "cuda")


def test_backend_and_vendor_pin_reject_an_impossible_pair():
    # The vendor is here, but the pinned stack has no accelerator for it.
    with pytest.raises(ValueError, match="mlx"):
        pick("nvidia", hardware=["nvidia"], mlx_caps=MLX_CPU_ONLY, backend="mlx")


def test_a_gguf_settles_on_llama_even_where_mlx_would_win():
    # A GGUF is a llama.cpp artifact; on Apple there is no NVIDIA/AMD accelerator for it,
    # so the pinned stack falls back to CPU and says so rather than picking MLX.
    chosen = pick(hardware=["apple"], mlx_caps=MLX_APPLE, gguf=Path("custom.gguf"))
    assert (chosen.backend, chosen.device) == ("llama", "cpu")
    assert chosen.downgraded is True


def test_legacy_stack_names_still_pin_the_backend():
    assert pick("vulkan").backend == "llama"
    assert pick("hip", llama_caps={"amd": "hip", "cpu": "cpu"}).device == "hip"
    assert pick("mlx", hardware=["apple"], mlx_caps=MLX_APPLE).backend == "mlx"


def test_cuda_is_accepted_as_an_alias_for_nvidia():
    chosen = pick("cuda", hardware=["nvidia"], mlx_caps=MLX_CUDA, llama_caps=LLAMA_CUDA)
    assert chosen.device == "cuda"


def test_an_explicit_backend_with_nothing_installed_defers_to_the_loader():
    # The loader knows the library path and prints a far better error than the router can.
    chosen = route("auto", backend="llama", hardware_present=["cpu"], mlx_caps={}, llama_caps={})
    assert (chosen.backend, chosen.device) == ("llama", "auto")


def test_no_stack_at_all_raises():
    with pytest.raises(ValueError, match="No backend"):
        pick(hardware=["amd"], mlx_caps={}, llama_caps={})


def test_unknown_device_and_backend_are_rejected():
    with pytest.raises(ValueError, match="Device must be one of"):
        pick("tpu")
    with pytest.raises(ValueError, match="Backend must be one of"):
        pick(backend="opencl")


def test_announce_warns_only_on_a_downgrade():
    with pytest.warns(RuntimeWarning, match="CPU"):
        announce(Routing("llama", "cpu", "no usable GPU; using llama on CPU", downgraded=True))
    announce(Routing("llama", "vulkan", "amd via llama", downgraded=False))  # silent


# --- option validation and artifacts ------------------------------------------------------


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


def test_gguf_path_refuses_a_quant_pinned_for_another_size():
    # Only a 4b GGUF is pinned: asking for 1.7b must fail loudly, not load 4B weights.
    with pytest.raises(ValueError, match="1.7b"):
        config.gguf_path("q8_0", "1.7b")
    assert config.gguf_path("q8_0", "4b").name == config.GGUF_MODELS["q8_0"].file


def test_pinned_quants_expose_path_and_digest():
    for quant, spec in config.GGUF_MODELS.items():
        assert len(spec.sha256) == 64
        assert config.gguf_path(quant).name == spec.file
        assert config.find_gguf_pin(spec.file) is spec


def test_model_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(config.MODEL_DIR_ENV, str(tmp_path))
    assert config.gguf_path("q8_0").parent == tmp_path


def test_describe_reports_both_stacks_and_the_auto_choice():
    report = backends.describe()
    assert set(report) >= {"available", "llama", "mlx", "hardware", "auto"}
    assert report["hardware"][-1] == "cpu"
    assert report["llama"]["devices"] == list(backends.LLAMA_DEVICES)
    assert isinstance(report["llama"]["present"], bool)
    assert report["mlx"]["available"] in (True, False)
    assert "backend" in report["auto"] or "error" in report["auto"]
