from dataclasses import dataclass
from pathlib import Path

from .protocols import ProgressCallback


@dataclass(frozen=True)
class ModelSpec:
    size: str
    repo: str
    revision: str
    hidden_size: int  # identifies a checkpoint directory regardless of its path

    @property
    def path(self) -> Path:
        return Path("models") / self.repo.split("/")[1]


# Same Spark2.5 architecture, tokenizer and 1M-token context; pinned original weights.
MODELS = {
    spec.size: spec
    for spec in (
        ModelSpec("4b", "XHToken/Spark-X2.5-4B", "0bcb35678590218655dff3765b9e61c83b35e9c4", 2560),
        ModelSpec(
            "1.7b", "XHToken/Spark-X2.5-1.7B", "14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8", 2048
        ),
    )
}
DEFAULT_SIZE = "4b"
MODEL_ID = MODELS[DEFAULT_SIZE].repo
MODEL_REVISION = MODELS[DEFAULT_SIZE].revision
RUNTIME_REVISION = "de2b4379fa1e2f2e1f99d84c83f0e008f651d86c"


@dataclass(frozen=True)
class GgufSpec:
    """One file of the GGUF conversions published by the model's authors, pinned by hash."""

    size: str
    quant: str
    repo: str
    revision: str
    file: str
    sha256: str

    @property
    def path(self) -> Path:
        return Path("models") / self.repo.split("/")[1] / self.file

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.file}"


_GGUF_4B = ("XHToken/Spark-X2.5-4B-GGUF", "9826e0be84e6e6e8b9668abc91421109a1df1e2d")
_GGUF_17B = ("XHToken/Spark-X2.5-1.7B-GGUF", "1f7fa33b1245c14730da39e125714ad3a327901b")
GGUF = {
    (spec.size, spec.quant): spec
    for spec in (
        GgufSpec(
            "4b",
            "q8_0",
            *_GGUF_4B,
            "Spark-X2.5-4B-Q8_0.gguf",
            "5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2",
        ),
        GgufSpec(
            "4b",
            "q4_k_m",
            *_GGUF_4B,
            "Spark-X2.5-4B-Q4_K_M.gguf",
            "adfcfa19a4ed6a5985da8bf565fe15f8e1a7e131d79bae2d19d48d1c40109428",
        ),
        GgufSpec(
            "4b",
            "bf16",
            *_GGUF_4B,
            "Spark-X2.5-4B.gguf",
            "8cecf405a41a4a10f833530910c2e13fde9fb39c325c8afc3c5d10e4181e1a14",
        ),
        GgufSpec(
            "1.7b",
            "q8_0",
            *_GGUF_17B,
            "Spark-X2.5-1.7B-Q8_0.gguf",
            "cd77c03185a834bb1162a4b7713520be5838058bfc54873645beff470bb24442",
        ),
        GgufSpec(
            "1.7b",
            "q4_k_m",
            *_GGUF_17B,
            "Spark-X2.5-1.7B-Q4_K_M.gguf",
            "902bde2522394954ac17821b3e5fd0df02defbc6944f122253f2580acf0503f4",
        ),
        GgufSpec(
            "1.7b",
            "bf16",
            *_GGUF_17B,
            "Spark-X2.5-1.7B.gguf",
            "67d5f2f06e6d898efcf0dc40cab8528bc82b871c8dafb0936784183d2c10cdd9",
        ),
    )
}
QUANTS = ("q8_0", "q4_k_m", "bf16")
DEFAULT_QUANT = "q8_0"
DEFAULT_MODEL_PATH = MODELS[DEFAULT_SIZE].path


def identify(hidden_size: int | None) -> ModelSpec:
    """Match a checkpoint's hidden size to a supported, pinned model."""
    for spec in MODELS.values():
        if hidden_size == spec.hidden_size:
            return spec
    raise ValueError(
        f"Unrecognized Spark2.5 checkpoint (hidden_size={hidden_size}); "
        f"supported sizes: {', '.join(MODELS)}"
    )


def download_gguf(
    size: str = DEFAULT_SIZE,
    quant: str = DEFAULT_QUANT,
    destination: Path | str | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Fetch one pinned GGUF file, verified against its sha256."""
    from .llama_release import fetch

    spec = GGUF[(size, quant)]
    return fetch(spec.url, Path(destination) if destination else spec.path, spec.sha256, progress)


def download_model(destination: Path | str | None = None, size: str = DEFAULT_SIZE) -> str:
    from huggingface_hub import snapshot_download

    spec = MODELS[size]
    return snapshot_download(
        spec.repo,
        revision=spec.revision,
        local_dir=destination or spec.path,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.safetensors",
            "tokenizer.model",
            "LICENSE",
            "README.md",
        ],
    )
