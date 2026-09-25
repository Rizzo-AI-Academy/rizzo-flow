import os
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


@dataclass(frozen=True)
class GgufSpec:
    """One pinned GGUF file: `flow` is our LoRA fine-tune merged into the base weights,
    `base` the conversion published by the model's authors."""

    size: str
    quant: str
    variant: str
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
# Fine-tuned on the prompts of spark-decisions-v3 (docs/training.md). Q4_K_M: llama-quantize
# b11081 from the BF16 file, no importance matrix.
_FLOW_4B = ("rizzoaiacademy/rizzo-flow", "55633c8cbd2b826bd3eefdeb05310450996649df")
_FLOW_17B = ("rizzoaiacademy/rizzo-flow-1.7b", "532e1586cc60787375ec7d86a5a302deb9a1adec")
GGUF = {
    (spec.size, spec.quant, spec.variant): spec
    for spec in (
        GgufSpec(
            "4b",
            "q8_0",
            "flow",
            *_FLOW_4B,
            "spark-x2.5-4b-rizzo-flow-lora-q8_0.gguf",
            "dbec3c89d33984772324e65a8ed56b48e958e301b856cb870a7c1b385d01691a",
        ),
        GgufSpec(
            "4b",
            "q4_k_m",
            "flow",
            *_FLOW_4B,
            "spark-x2.5-4b-rizzo-flow-lora-q4_k_m.gguf",
            "79de5cb8dbfd1a1f5cb3037252251594352841fe5e3dc1ae8cead053010fcd54",
        ),
        GgufSpec(
            "4b",
            "bf16",
            "flow",
            *_FLOW_4B,
            "spark-x2.5-4b-rizzo-flow-lora-bf16.gguf",
            "8e0f9a73644d50fc319ef4205f64878f17a88ba8d4b875ffd3bb5e61597da865",
        ),
        GgufSpec(
            "1.7b",
            "q8_0",
            "flow",
            *_FLOW_17B,
            "spark-x2.5-1.7b-rizzo-flow-lora-q8_0.gguf",
            "685403da9c62e0745cb77817ee2d66418e3ce85002ef403cf2680481e22bf16d",
        ),
        GgufSpec(
            "1.7b",
            "q4_k_m",
            "flow",
            *_FLOW_17B,
            "spark-x2.5-1.7b-rizzo-flow-lora-q4_k_m.gguf",
            "c2dab675c94226d548afb2d5027507a380c879adad2ef7a4e65289ff9e01123b",
        ),
        GgufSpec(
            "1.7b",
            "bf16",
            "flow",
            *_FLOW_17B,
            "spark-x2.5-1.7b-rizzo-flow-lora-bf16.gguf",
            "21bf261a4c15460d03172b522e1507152d21c31e167ca44ffe083f2f1893c756",
        ),
        GgufSpec(
            "4b",
            "q8_0",
            "base",
            *_GGUF_4B,
            "Spark-X2.5-4B-Q8_0.gguf",
            "5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2",
        ),
        GgufSpec(
            "4b",
            "q4_k_m",
            "base",
            *_GGUF_4B,
            "Spark-X2.5-4B-Q4_K_M.gguf",
            "adfcfa19a4ed6a5985da8bf565fe15f8e1a7e131d79bae2d19d48d1c40109428",
        ),
        GgufSpec(
            "4b",
            "bf16",
            "base",
            *_GGUF_4B,
            "Spark-X2.5-4B.gguf",
            "8cecf405a41a4a10f833530910c2e13fde9fb39c325c8afc3c5d10e4181e1a14",
        ),
        GgufSpec(
            "1.7b",
            "q8_0",
            "base",
            *_GGUF_17B,
            "Spark-X2.5-1.7B-Q8_0.gguf",
            "cd77c03185a834bb1162a4b7713520be5838058bfc54873645beff470bb24442",
        ),
        GgufSpec(
            "1.7b",
            "q4_k_m",
            "base",
            *_GGUF_17B,
            "Spark-X2.5-1.7B-Q4_K_M.gguf",
            "902bde2522394954ac17821b3e5fd0df02defbc6944f122253f2580acf0503f4",
        ),
        GgufSpec(
            "1.7b",
            "bf16",
            "base",
            *_GGUF_17B,
            "Spark-X2.5-1.7B.gguf",
            "67d5f2f06e6d898efcf0dc40cab8528bc82b871c8dafb0936784183d2c10cdd9",
        ),
    )
}


@dataclass(frozen=True)
class FlowCheckpoint:
    """The fine-tune merged into the BF16 weights, as a Hugging Face checkpoint (MLX backend,
    transformers). Same repository as the GGUF files; the weight files are pinned by hash."""

    size: str
    repo: str
    revision: str
    weights: dict  # safetensors file name -> sha256

    @property
    def path(self) -> Path:
        return Path("models") / self.repo.split("/")[1]


# Everything a checkpoint directory needs, and nothing else from the repository (GGUF, adapter).
CHECKPOINT_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "configuration_spark.py",
    "modeling_spark.py",
    "LICENSE",
]
FLOW_CHECKPOINTS = {
    spec.size: spec
    for spec in (
        FlowCheckpoint(
            "4b",
            *_FLOW_4B,
            {
                "model-00001-of-00005.safetensors": "828eda44cd2dbc9e14f09fa38a53c5424cdb2bc53059eec8d2f79c82d10c0354",
                "model-00002-of-00005.safetensors": "e9b4f2ddaa2e0131fa7031e1a445edd1158e6ca011bdbd345887453e8389ae10",
                "model-00003-of-00005.safetensors": "0e83bd091df991896c757b6d5e84d2770c950226eb94322b322c83fb87be53db",
                "model-00004-of-00005.safetensors": "c0d9e2912665c94d424a9c5c3b7938f945e3dbad61a4f0fbf08d5b46b0a27457",
                "model-00005-of-00005.safetensors": "e717c60c5b077a13546ce2688cd52048e917198dd36c97c59c628cf6b5ecbe5d",
            },
        ),
        FlowCheckpoint(
            "1.7b",
            *_FLOW_17B,
            {
                "model-00001-of-00002.safetensors": "f651c72c4acca88b9e32c266956235310642e7a90ce7a0b511c9234e7c54d96c",
                "model-00002-of-00002.safetensors": "f6408f4bb3188f7c9d2965013b9c751f8744c76e97c7502cbbd8857c64d0365e",
            },
        ),
    )
}


QUANTS = ("q8_0", "q4_k_m", "bf16")
DEFAULT_QUANT = "q8_0"
VARIANTS = ("flow", "base")
DEFAULT_VARIANT = "flow"
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


def gguf_spec(size=DEFAULT_SIZE, quant=None, variant=None) -> GgufSpec:
    quant, variant = quant or DEFAULT_QUANT, variant or DEFAULT_VARIANT
    spec = GGUF.get((size, quant, variant))
    if spec is None:
        have = sorted(q for s, q, v in GGUF if s == size and v == variant)
        raise ValueError(
            f"No {variant} GGUF for {size} {quant}; {variant} has: {', '.join(have)}"
            + (" (use --weights base for the original q4_k_m)" if variant == "flow" else "")
        )
    return spec


def hf_token() -> str | None:
    """Token for private Hugging Face repositories: HF_TOKEN, a .env file in the working
    directory, or the token saved by `hf auth login`."""
    if token := os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return token.strip()
    dotenv = Path(".env")
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN" and value.strip().strip("\"'"):
                return value.strip().strip("\"'")
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    saved = home / "token"
    return saved.read_text(encoding="utf-8").strip() or None if saved.is_file() else None


def download_gguf(size=DEFAULT_SIZE, quant=None, destination=None, progress=None, variant=None):
    """Fetch one pinned GGUF file, verified against its sha256."""
    from .llama_release import fetch

    spec = gguf_spec(size, quant, variant)
    target = Path(destination) if destination else spec.path
    return fetch(spec.url, target, spec.sha256, progress, token=hf_token())


def checkpoint_path(size=DEFAULT_SIZE, variant=None) -> Path:
    """Where `rizzo download --backend mlx` puts a checkpoint directory."""
    variant = variant or DEFAULT_VARIANT
    return (FLOW_CHECKPOINTS if variant == "flow" else MODELS)[size].path


def download_model(destination=None, size=DEFAULT_SIZE, variant=None):
    from huggingface_hub import snapshot_download

    if (variant or DEFAULT_VARIANT) == "flow":
        spec = FLOW_CHECKPOINTS[size]
        # Root files only: the repository also holds GGUF files and the bare adapter.
        return snapshot_download(
            spec.repo,
            revision=spec.revision,
            local_dir=destination or spec.path,
            allow_patterns=CHECKPOINT_FILES + list(spec.weights),
            token=hf_token(),
        )
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
