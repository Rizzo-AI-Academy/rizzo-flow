"""Validate the llama.cpp backend against the Spark GGUF and preserve raw evidence.

Local proofs for the prompt criterion (SPEC.md #2a-#2d):
- the GGUF chat template renders byte-identically to the vendored HF fixture;
- every answer letter A-Z is a single token in the GGUF vocabulary;
- our ctypes tokenizer agrees with the `llama-tokenize` binary of the same build;
- `prompt_sha256` and `input_tokens` for the ticket and the smoke set are recorded, so a future
  cross-backend comparison is possible.

  RIZZO_LLAMA_LIB=/path/to/llama.cpp/build/bin \
    .venv/bin/python scripts/validate_llama.py --gguf /path/Spark-X2.5-4B-Q8_0.gguf \
      --output results/llama-q8-vulkan-validation
"""

import argparse
import re
import string
import subprocess
import time
from pathlib import Path

from rizzo_flow.backend_llama import library_dir
from rizzo_flow.backends import load_backend
from rizzo_flow.cli import read_jsonl, write_json
from rizzo_flow.decisions import decode
from rizzo_flow.engine import Engine
from rizzo_flow.evaluation import evaluate
from rizzo_flow.llama_tokenizer import LlamaTokenizer
from rizzo_flow.prompts import compile_request
from rizzo_flow.schema import Request

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "data" / "spark2_5-chat-template.jinja"
CLI_TOKENIZE = re.compile(r"\[([\d,\s]*)\]")
MESSAGES = [
    {"role": "system", "content": "You are a precise decision function."},
    {"role": "user", "content": "<evidence>\nx\n</evidence>\n\nQuestion: q?"},
]
TEMPLATE_ARGUMENTS = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}


def template_proof(backend) -> dict:
    """(a) The template carried by the GGUF must render like the HF one we vendored."""
    if not FIXTURE.is_file():
        return {"ok": None, "reason": f"missing fixture {FIXTURE}"}
    fixture = LlamaTokenizer(backend.runtime, FIXTURE.read_text(encoding="utf-8"))
    from_gguf = backend.tokenizer.apply_chat_template(MESSAGES, **TEMPLATE_ARGUMENTS)
    from_fixture = fixture.apply_chat_template(MESSAGES, **TEMPLATE_ARGUMENTS)
    return {
        "ok": from_gguf == from_fixture,
        "fixture": str(FIXTURE),
        "ends_with_think_close": from_gguf.endswith("</think>"),
    }


def letters_proof(backend) -> dict:
    """(b) Every declared answer letter must be exactly one token."""
    encoded = {letter: backend.tokenizer.encode(letter) for letter in string.ascii_uppercase}
    single = {letter: ids[0] for letter, ids in encoded.items() if len(ids) == 1}
    return {
        "ok": len(single) == len(string.ascii_uppercase),
        "single_tokens": len(single),
        "first": single.get("A"),
        "last": single.get("Z"),
    }


def cli_tokenize(binary: Path, gguf: Path, text: str) -> list[int] | None:
    result = subprocess.run(
        [str(binary), "-m", str(gguf), "--ids", "-p", text],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    match = CLI_TOKENIZE.search(result.stdout)
    if not match:
        return None
    return [int(value) for value in match.group(1).split(",") if value.strip()]


def tokenizer_proof(backend, gguf: Path) -> dict:
    """(c) Our binding must tokenize exactly like the reference binary of the same build."""
    binary = library_dir() / "llama-tokenize"
    if not binary.exists():
        return {"ok": None, "reason": f"missing {binary}"}
    cases = []
    for text in ("A", "Question:", "He is an engineer."):
        ours = backend.tokenizer.encode(text)
        theirs = cli_tokenize(binary, gguf, text)
        cases.append(
            {"text": text, "ours": ours, "cli": theirs, "ok": theirs is not None and ours == theirs}
        )
    return {"ok": all(case["ok"] for case in cases), "binary": str(binary), "cases": cases}


def prompt_digest(answers: dict) -> dict:
    return {
        key: {"prompt_sha256": answer["prompt_sha256"], "input_tokens": answer["input_tokens"]}
        for key, answer in answers.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke", default="benchmarks/smoke.jsonl")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    backend = load_backend(
        "llama", gguf=args.gguf, device=args.device, ctx=args.ctx, batch_size=args.batch_size
    )
    engine = Engine(backend, ctx=args.ctx)
    request = Request.model_validate_json(Path("examples/ticket.json").read_text(encoding="utf-8"))
    prefix, jobs = compile_request(backend.tokenizer, request, args.ctx)
    logits, timing = backend.score(prefix, jobs, "shared")
    ticket = {
        "prefix_tokens": len(prefix),
        "timing": timing,
        "answers": {
            job.id: {
                **decode(request.questions[job.id], logits[job.id]),
                "prompt_sha256": job.prompt_sha256,
                "input_tokens": len(job.tokens),
            }
            for job in jobs
        },
    }
    smoke = evaluate(engine, read_jsonl(args.smoke))
    report = {
        "backend": backend.metadata,
        "device": args.device,
        "ctx": args.ctx,
        "batch_size": args.batch_size,
        "local_proofs": {
            "template_matches_fixture": template_proof(backend),
            "letters_are_single_tokens": letters_proof(backend),
            "tokenizer_matches_cli": tokenizer_proof(backend, args.gguf.resolve()),
        },
        "ticket": ticket,
        "smoke": {
            "summary": smoke["summary"],
            "prompt_digests": [
                {"id": row["id"], "answers": prompt_digest(row["response"]["answers"])}
                for row in smoke["rows"]
            ],
        },
        "validation_seconds": time.perf_counter() - started,
    }
    write_json(report, out / "validation.json")
    for name, proof in report["local_proofs"].items():
        print(f"{name}: {proof['ok']}", flush=True)
    print("smoke decisions_per_second", round(smoke["summary"]["decisions_per_second"], 2))
    print("written", out / "validation.json")


if __name__ == "__main__":
    main()
