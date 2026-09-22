import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_QUANT, DEFAULT_SIZE, MODELS, QUANTS, download_gguf, download_model
from .llama_release import ACCELERATORS
from .loader import BACKENDS, DEVICES


def write_json(value: object, destination: Path | str | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Results are create-only; never silently overwrite benchmark evidence.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")


def read_jsonl(path: Path | str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def progress(name: str, done: int, total: int) -> None:
    """One carriage-returned line per file; silent when stderr is not a terminal."""
    if sys.stderr.isatty():
        share = f"{100 * done / total:5.1f}%" if total else f"{done >> 20} MiB"
        sys.stderr.write(f"\r{name}: {share}")
        if total and done >= total:
            sys.stderr.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rizzo Flow — local Spark typed decisions")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download", help="Download the pinned llama.cpp runtime for this machine and the weights"
    )
    download.add_argument("--size", choices=tuple(MODELS), default=DEFAULT_SIZE)
    download.add_argument("--backend", choices=BACKENDS, default="llama")
    download.add_argument("--quant", choices=QUANTS, default=DEFAULT_QUANT, help="GGUF file")
    download.add_argument(
        "--runtime",
        choices=ACCELERATORS,
        default="auto",
        help="llama.cpp build: auto = Metal on a Mac, CUDA with an NVIDIA driver, else Vulkan "
        "(AMD, Intel and NVIDIA GPUs)",
    )
    download.add_argument("--only", choices=("runtime", "weights"))
    download.add_argument("--destination", type=Path, help="Weights; default: under models/")
    schema = commands.add_parser("schema", help="Print the JSON Schema for requests")
    schema.add_argument("--output")
    schema.add_argument("--response", action="store_true", help="Print the output schema")
    commands.add_parser("devices", help="Show which compute backends this install can use")
    fit = commands.add_parser("calibrate", help="Fit temperatures on separate labeled logit rows")
    fit.add_argument("input", type=Path)
    fit.add_argument("--fingerprint", required=True)
    fit.add_argument("--output", required=True)
    for name in ("decide", "serve", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--size", choices=tuple(MODELS), default=DEFAULT_SIZE)
        p.add_argument("--backend", choices=BACKENDS, default="llama")
        p.add_argument("--model", type=Path, help="GGUF file (MLX: checkpoint directory)")
        p.add_argument("--quant", choices=QUANTS, help=f"Pinned GGUF file; default {DEFAULT_QUANT}")
        p.add_argument("--bits", type=int, choices=(4, 8), help="MLX backend only")
        p.add_argument(
            "--device",
            choices=DEVICES,
            default="auto",
            help="auto = best GPU of the installed runtime, else CPU; a family name requires it",
        )
        p.add_argument("--threads", type=int, help="CPU threads (llama backend)")
        p.add_argument("--batch-size", type=int, default=4)
        # --max-tokens is the former name, kept as an alias.
        p.add_argument(
            "--ctx",
            "--max-tokens",
            dest="ctx",
            type=int,
            default=8192,
            help="Context limit in tokens per question (state + question); longer inputs "
            "are rejected",
        )
        p.add_argument("--calibration", type=Path)
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8017)
        else:
            p.add_argument("input", type=Path)
            p.add_argument("--output")
            if name == "evaluate":
                p.add_argument("--repeats", type=int, default=1)
                p.add_argument("--compare-modes", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "download":
            if args.backend == "mlx":
                print(download_model(args.destination, args.size))
                return
            if args.only != "weights":
                from .llama_release import install

                print(install(args.runtime, progress))
            if args.only != "runtime":
                print(download_gguf(args.size, args.quant, args.destination, progress))
            return
        if args.command == "devices":
            from .loader import describe

            write_json(describe(), None)
            return
        if args.command == "schema":
            from .responses import Response
            from .schema import Request

            write_json((Response if args.response else Request).model_json_schema(), args.output)
            return
        if args.command == "calibrate":
            from .calibration import fit_temperature

            write_json(
                fit_temperature(read_jsonl(args.input), args.fingerprint).model_dump(), args.output
            )
            return
        from .calibration import Calibration
        from .engine import Engine

        # Validate the request before loading gigabytes of weights.
        if args.command == "decide":
            from .schema import Request

            request = Request.model_validate_json(args.input.read_text(encoding="utf-8"))
        from .loader import load_backend

        backend = load_backend(
            args.backend,
            size=args.size,
            model=args.model,
            quant=args.quant,
            bits=args.bits,
            device=args.device,
            ctx=args.ctx,
            batch_size=args.batch_size,
            threads=args.threads,
        )
        calibration = Calibration.from_file(args.calibration) if args.calibration else None
        engine = Engine(backend, ctx=args.ctx, calibration=calibration)
        if args.command == "decide":
            write_json(engine.decide(request).model_dump(), args.output)
        elif args.command == "evaluate":
            from .evaluation import evaluate

            report = evaluate(engine, read_jsonl(args.input), args.repeats, args.compare_modes)
            write_json(report.model_dump(), args.output)
        elif args.command == "serve":
            import uvicorn

            from .api import create_app

            uvicorn.run(create_app(engine), host=args.host, port=args.port)
    except (ValueError, OSError, ImportError) as error:
        print(f"rizzo: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
