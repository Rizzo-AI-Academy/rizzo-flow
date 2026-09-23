"""Second-level numbers for one `semif_compare.py` run: held-out halves and paired differences.

  python scripts/semif_report.py results/semif-compare/RUN --semif /path/SemIf \\
      --against mlx=results/semif-compare/rizzo-q8-v3-cuda \\
      --against semif=/path/SemIf/results/mlx/2026-09-17-q8-fixed

Held-out = the half of the source groups the v3 prompt was *not* chosen on (`prompt_lab.split`).
Paired differences use SemIf's own source-group bootstrap. Writes RUN/analysis.json, create-only.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

from semif_compare import chosen, dev_groups, read, write


def held_out(rows: list[dict], perturbed: list[dict]) -> tuple[list[dict], list[dict]]:
    """The half of the source groups the prompt was never chosen on."""
    dev = dev_groups(rows)
    return (
        [row for row in rows if row["group_id"] not in dev],
        [row for row in perturbed if row["provenance"]["source_group_id"] not in dev],
    )


def predictions(folder: Path, name: str) -> list[dict]:
    plain, packed = folder / f"{name}.jsonl", folder / f"{name}.jsonl.gz"
    if plain.is_file():
        return read(plain)
    with gzip.open(packed, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--semif", type=Path, required=True, help="SemIf repository checkout")
    parser.add_argument("--against", action="append", default=[], metavar="NAME=FOLDER")
    args = parser.parse_args()
    sys.path[:0] = [str(args.semif / "benchmarks")]
    import evaluate

    data = args.semif / "benchmarks" / "data"
    gold = {name: read(data / f"{name}.jsonl") for name in ("authored144", "perturbations108")}
    held = dict(zip(gold, held_out(gold["authored144"], gold["perturbations108"]), strict=True))
    ours = {name: predictions(args.run, name) for name in gold}

    def subset(rows: list[dict], part: list[dict]) -> list[dict]:
        wanted = {row["id"] for row in part}
        return [row for row in rows if row["id"] in wanted]

    def score(rows: list[dict], part: list[dict]) -> float:
        return evaluate.evaluate(part, subset(rows, part))["mean_family_balanced_accuracy"]

    report = {
        "run": args.run.name,
        "held_out": {name: score(ours[name], held[name]) for name in gold},
        "against": {},
    }
    for entry in args.against:
        label, folder = entry.split("=", 1)
        theirs = {name: predictions(Path(folder), name) for name in gold}
        other = {row["id"]: row for name in gold for row in theirs[name]}
        mine = [row for name in gold for row in ours[name]]
        report["against"][label] = {
            "folder": folder,
            "held_out": {name: score(theirs[name], held[name]) for name in gold},
            "paired_difference_ours_minus_theirs": {
                name: evaluate.evaluate(gold[name], ours[name], theirs[name])["paired_comparison"]
                for name in gold
            },
            "different_argmax": sum(chosen(row) != chosen(other[row["id"]]) for row in mine),
            "rows": len(mine),
        }
    write(args.run / "analysis.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
