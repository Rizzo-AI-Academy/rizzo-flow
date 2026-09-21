import argparse
import json
import sys
from pathlib import Path

from .backends import BACKENDS, DEFAULT_QUANT, DEVICE_CHOICES
from .config import DEFAULT_SIZE, GGUF_MODELS, MODELS, download_gguf, download_model


def write_json(value, destination):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Results are create-only; never silently overwrite benchmark evidence.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser(description="Rizzo Flow — local Spark typed decisions")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the pinned original Spark checkpoint")
    download.add_argument("--size", choices=tuple(MODELS), default=DEFAULT_SIZE)
    download.add_argument("--destination", type=Path, help="Default: models/<checkpoint name>")
    download.add_argument(
        "--format",
        choices=("mlx", "gguf"),
        default="mlx",
        help="mlx = original safetensors for the MLX backend; gguf = pinned llama.cpp artifact",
    )
    download.add_argument("--quant", choices=tuple(GGUF_MODELS), default=DEFAULT_QUANT)
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
        p.add_argument("--model", type=Path, help="Checkpoint directory; overrides --size")
        p.add_argument("--bits", type=int, choices=(4, 8))
        p.add_argument(
            "--backend",
            choices=BACKENDS,
            default="auto",
            help="auto = best stack for this machine's GPU; mlx/llama pin one",
        )
        p.add_argument(
            "--gguf",
            type=Path,
            help="GGUF checkpoint; implies the llama backend and settles the choice",
        )
        p.add_argument(
            "--quant",
            choices=tuple(GGUF_MODELS),
            default=DEFAULT_QUANT,
            help="Which pinned GGUF to load (llama backend only)",
        )
        p.add_argument(
            "--device",
            choices=DEVICE_CHOICES,
            default="auto",
            help=(
                "auto = best accelerator present; gpu/apple/nvidia/amd/intel pin hardware; "
                "mlx/vulkan/hip pin a stack; cpu forces CPU"
            ),
        )
        p.add_argument("--batch-size", type=int, default=4)
        # --max-tokens is the former name, kept as an alias.
        p.add_argument(
            "--ctx",
            "--max-tokens",
            dest="ctx",
            type=int,
            default=8192,
            help="Context limit in tokens per question (state + question); longer inputs are rejected",
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
            if args.format == "gguf":
                print(download_gguf(args.quant, args.destination, args.size))
            else:
                print(download_model(args.destination, args.size))
            return
        if args.command == "devices":
            from .backends import describe as describe_backends

            write_json(describe_backends(), None)
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
        from .backends import load_backend
        from .calibration import Calibration
        from .engine import Engine

        # Validate the request before loading gigabytes of weights.
        if args.command == "decide":
            from .schema import Request

            request = Request.model_validate_json(args.input.read_text(encoding="utf-8"))
        backend = load_backend(
            args.backend,
            size=args.size,
            model=args.model,
            gguf=args.gguf,
            quant=args.quant,
            bits=args.bits,
            device=args.device,
            ctx=args.ctx,
            batch_size=args.batch_size,
        )
        calibration = Calibration.from_file(args.calibration) if args.calibration else None
        engine = Engine(backend, ctx=args.ctx, calibration=calibration)
        if args.command == "decide":
            write_json(engine.decide(request), args.output)
        elif args.command == "evaluate":
            from .evaluation import evaluate

            write_json(
                evaluate(engine, read_jsonl(args.input), args.repeats, args.compare_modes),
                args.output,
            )
        elif args.command == "serve":
            import uvicorn

            from .api import create_app

            uvicorn.run(create_app(engine), host=args.host, port=args.port)
    except (ValueError, OSError, ImportError) as error:
        print(f"rizzo: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
