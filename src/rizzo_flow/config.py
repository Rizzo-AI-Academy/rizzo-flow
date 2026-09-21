import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path


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
DEFAULT_MODEL_PATH = MODELS[DEFAULT_SIZE].path


def identify(config: dict) -> ModelSpec:
    """Match a checkpoint's config.json to a supported, pinned model."""
    for spec in MODELS.values():
        if config.get("hidden_size") == spec.hidden_size:
            return spec
    raise ValueError(
        f"Unrecognized Spark2.5 checkpoint (hidden_size={config.get('hidden_size')}); "
        f"supported sizes: {', '.join(MODELS)}"
    )


# --- llama.cpp backend: pinned GGUF artifacts ---------------------------------------------
# The GGUF is a third-party conversion of the pinned original weights, so both provenance
# layers are recorded: the model identity comes from `MODELS`, the artifact from here.
MODEL_DIR_ENV = "RIZZO_MODEL_DIR"


@dataclass(frozen=True)
class GgufSpec:
    size: str
    repo: str
    revision: str
    file: str
    sha256: str
    quant: str


GGUF_MODELS = {
    spec.quant: spec
    for spec in (
        GgufSpec(
            "4b",
            "stornic56/Spark-X2.5-4B-GGUF",
            "7ce72e5cba148e5e4bcd0ff0e59c8268f9820619",
            "Spark-X2.5-4B-Q8_0.gguf",
            "092a263df8c891cdddd98b14b9ed71e44bb84643049fbfe656fb682b71d316c6",
            "q8_0",
        ),
        GgufSpec(
            "4b",
            "stornic56/Spark-X2.5-4B-GGUF",
            "7ce72e5cba148e5e4bcd0ff0e59c8268f9820619",
            "Spark-X2.5-4B-bf16.gguf",
            "2ff41881527d095dbc02fe0c9b8e6ecd221dfe28b242d1f03dc5592d1b39fbb2",
            "bf16",
        ),
    )
}


def model_dir() -> Path:
    """Where GGUF checkpoints live; `RIZZO_MODEL_DIR` overrides the in-repo `models/`."""
    return Path(os.environ.get(MODEL_DIR_ENV, "models"))


def gguf_path(quant: str, size: str = DEFAULT_SIZE) -> Path:
    """Path of the pinned GGUF for this size and quant; refuses a pin of another size."""
    if quant not in GGUF_MODELS:
        raise ValueError(f"Unknown quant {quant}; available: {', '.join(GGUF_MODELS)}")
    spec = GGUF_MODELS[quant]
    if spec.size != size:
        raise ValueError(
            f"No GGUF pinned for size {size}: the {quant} artifact is {spec.size}. "
            "Pin one in config.GGUF_MODELS or pass --gguf."
        )
    return model_dir() / spec.file


def find_gguf_pin(filename: str) -> GgufSpec | None:
    """Match a local GGUF file name back to its pinned artifact, for provenance."""
    return next((spec for spec in GGUF_MODELS.values() if spec.file == filename), None)


def sha256_file(path: Path) -> str:
    """Streaming digest: checkpoints are gigabytes, so they are never read into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_gguf(quant: str, destination=None, size: str = DEFAULT_SIZE) -> Path:
    """Fetch the pinned GGUF with the standard library, verifying the pinned sha256."""
    if quant not in GGUF_MODELS:
        raise ValueError(f"Unknown quant {quant}; available: {', '.join(GGUF_MODELS)}")
    spec = GGUF_MODELS[quant]
    target = Path(destination) if destination else gguf_path(quant, size)
    if target.is_file() and sha256_file(target) == spec.sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{spec.file}"
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "rizzo-flow"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    digest = sha256_file(partial)
    if digest != spec.sha256:
        partial.unlink()
        raise ValueError(f"{spec.file}: sha256 mismatch (expected {spec.sha256}, got {digest})")
    partial.replace(target)
    return target


def download_model(destination=None, size=DEFAULT_SIZE):
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
