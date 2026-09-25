"""Shared pieces of the LoRA/QLoRA pipeline (run with .venv-train, not the project venv).

Prompts are compiled by the project's own `prompts.compile_request`, so a training example is
byte for byte the prompt the served model sees; the target is a distribution over the answer
letters, read at the last prompt position exactly like the llama.cpp/MLX backends do.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# RIZZO_TRAIN_SIZE=1.7b points every script at the small checkpoint (same architecture).
MODEL_DIR = (
    ROOT
    / "models"
    / {"4b": "Spark-X2.5-4B", "1.7b": "Spark-X2.5-1.7B"}[os.environ.get("RIZZO_TRAIN_SIZE", "4b")]
)
DATA_DIR = ROOT / ".research" / "train-data"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Progress:
    """`step N/M`, elapsed, rate and remaining-time estimate, printed every `every` steps."""

    def __init__(self, total, label, every=10, done=0):
        """`done`: steps already completed before this process (a resumed run)."""
        self.total, self.label, self.every, self.done = total, label, max(1, every), done
        self.start = time.perf_counter()

    def __call__(self, step, extra=""):
        if step % self.every and step != self.total:
            return
        elapsed = time.perf_counter() - self.start
        rate = (step - self.done) / elapsed if elapsed else 0.0
        remaining = (self.total - step) / rate if rate else float("nan")
        log(
            f"{self.label} step {step}/{self.total} ({100 * step / self.total:.1f}%) "
            f"elapsed {fmt(elapsed)} ETA {fmt(remaining)} {extra}".rstrip()
        )


def fmt(seconds):
    if math.isnan(seconds):
        return "?"
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)


def compile_native(tok, native_request: dict, ctx=4096):
    """Native request dict → [(question key, tokens, slots)] via the project's compiler."""
    from rizzo_flow.prompts import compile_request
    from rizzo_flow.schema import Request

    request = Request.model_validate(native_request)
    _, compiled = compile_request(tok, request, ctx)
    return [(c.id, c.tokens, c.slots) for c in compiled]


def load_model(quant="bf16", dtype=None):
    import torch
    from transformers import AutoModelForCausalLM

    # On Windows the driver spills past VRAM into system RAM instead of failing, and a run then
    # crawls for minutes (seen with --no-checkpointing). Capping the allocator turns that into an
    # immediate out-of-memory error.
    torch.cuda.set_per_process_memory_fraction(0.92)
    kwargs = {"trust_remote_code": True, "dtype": torch.bfloat16, "device_map": {"": 0}}
    if quant == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **kwargs)
    if quant == "nf4":
        patch_quantized_cast(model)
    return model


def patch_quantized_cast(model):
    """The remote decoder layer casts activations to `mlp.gate_proj.weight.dtype`, which is uint8
    once bitsandbytes packs the weights in NF4. Same forward, cast to the compute dtype instead."""
    import torch

    layer_class = type(model.model.layers[0])

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states).to(torch.bfloat16)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).to(torch.bfloat16)
        return residual + self.mlp(hidden_states)

    layer_class.forward = forward


def letter_logits(model, tokens, slots):
    """Logits of the answer letters after the full prompt (batch of one, no padding)."""
    import torch

    ids = torch.tensor([tokens], device="cuda")
    out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
    return out.logits[0, -1, slots].float()


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
