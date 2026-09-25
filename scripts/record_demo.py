"""Re-record the answers and timings shown by the landing page demo (docs/index.html).

Every question of the inline `DEMO` object is replayed through the Jev-compatible path
(`compat.to_native` / `from_native`), in process, one question per request, on a warm model.
The stored time is the median of `--repeats` calls. Nothing is picked or fixed by hand: whatever
the model answers is written back. Run from the repository root:

  python scripts/record_demo.py            # llama.cpp, 4B, q8_0
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from rizzo_flow import compat
from rizzo_flow.engine import Engine
from rizzo_flow.loader import BACKENDS, load_backend

PAGE = Path("docs/index.html")
LINE = re.compile(r"^(\s*var DEMO = )(\{.*\})(;\s*)$", re.MULTILINE)


def wire(kind, record):
    if kind == "bool":
        return {"type": "noul", "instructions": record["q"]}
    if kind == "class":
        return {
            "type": "choice",
            "instructions": record["q"],
            "criteria": dict.fromkeys(record["probs"]),
        }
    return {"type": "score", "instructions": record["q"], "criteria": record["levels"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="llama")
    parser.add_argument("--quant")
    parser.add_argument("--weights", choices=("flow", "base"), help="pinned GGUF; default flow")
    parser.add_argument("--bits", type=int, choices=(4, 8))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    with PAGE.open(encoding="utf-8", newline="") as stream:  # keep the line endings as they are
        page = stream.read()
    found = LINE.search(page)
    demo = json.loads(found.group(2))
    backend = load_backend(
        args.backend, quant=args.quant, weights=args.weights, bits=args.bits, device=args.device
    )
    engine = Engine(backend)
    served = compat.model_name(backend.metadata)
    times = []
    for preset, languages in demo["data"].items():
        for language, entry in languages.items():
            for kind in ("bool", "class", "score"):
                record = entry[kind]
                request = compat.SystemOneRequest.model_validate(
                    {
                        "state": entry["state"],
                        "model": served,
                        "questions": {"q": wire(kind, record)},
                    }
                )
                native, options = compat.to_native(request)
                seconds = []
                for _ in range(args.repeats + 1):  # the first call warms up and is dropped
                    mark = time.perf_counter()
                    response = engine.decide(native)
                    seconds.append(time.perf_counter() - mark)
                answer = compat.from_native(request, response, options, served)["answers"]["q"]
                record["ms"] = round(statistics.median(seconds[1:]) * 1000)
                times.append(record["ms"])
                if kind == "bool":
                    record["p"] = round(answer["noul"], 4)
                elif kind == "class":
                    record["choice"] = answer["choice"]
                    record["probs"] = {k: round(v, 4) for k, v in answer["probabilities"].items()}
                else:
                    record["score"] = round(answer["score"], 3)
                    record["probs"] = [round(v, 4) for v in answer["probabilities"].values()]
                print(
                    preset,
                    language,
                    kind,
                    record["ms"],
                    "ms",
                    json.dumps(record, ensure_ascii=False),
                )
    demo["model"] = served
    text = json.dumps(demo, ensure_ascii=False, separators=(",", ":"))
    PAGE.write_text(
        page[: found.start(2)] + text + page[found.end(2) :], encoding="utf-8", newline=""
    )
    print(f"{len(times)} decisions, median {statistics.median(times)} ms, model {served}")


if __name__ == "__main__":
    main()
