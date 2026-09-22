# Parity suite — this port vs. the original Rizzo Flow

The value of this fork is **behavioural parity**: a client written for Rizzo Flow must
work unchanged, and the ported logic must produce the same answers. This directory
contains the automated check, in two layers.

```
parity/run_parity.sh [--end-to-end]
```

| | Layer A — decision core | Layer B — end-to-end |
|---|---|---|
| what | same question + **same logits** → same answer | full request through both engines |
| reference | `rizzo_flow.decisions.decode` | `rizzo_flow` engine (Spark-X2.5, MLX) |
| our side | `decisions::decode` (Rust) | `rizzo-flow-rs decide` (llama.cpp) |
| model | **none** (model-free, deterministic) | two different models — see caveat |
| comparison | **exact**: every field must match, numbers within 1e-9 | **structure only**: keys, types, nesting |
| verdict | any difference is a porting bug | only *structural* differences count |

Layer A is the strong one: it removes the model from the equation, so any difference is
a defect in the port. Layer B cannot compare answers — the reference runs
Spark-X2.5 on MLX, this port runs a GGUF on llama.cpp — so it compares the *contract*.

## Layer A — golden vectors

`core_vectors.json` holds 23 cases covering: boolean / choice / score / numeric,
abstention winning, out-of-range (`__below_range__` / `__above_range__`),
`policy.allow_abstain`, `policy.min_top_probability`,
`policy.max_unavailable_probability`, ties (first maximum wins), temperatures
(0.5 / 1.0 / 2.0) and **three cases that must fail** (logit/candidate mismatch,
temperature 0, temperature negative) — the error messages are compared exactly, so the
failure behaviour is part of the parity claim, not just the happy path. Candidate order is documented in the file itself, because both
implementations must agree on it: `boolean=[false,true]` · `choice=options in order` ·
`score=levels` · `numeric=anchors, __below_range__, __above_range__` · `__insufficient__`
last when abstention is allowed.

Latest result (`results/report_core.json`, 22/09/2026):

```
cases: 23   matched: 23   with differences: 0
fields compared: 350 numeric, 168 scalar
max numeric delta: 4.44e-16      (double-precision summation order)
blocking differences: 0
```

## Layer B — response shape

Same request file (`rizzo_test_6.json`) through both engines. The reference is run with
`bits=8` — the documented Q8 configuration; bf16 is ~10 GB and does not fit the node's
8 GB.

Two notes on running the reference:

* the MLX-CUDA reference **aborts during interpreter teardown** (`terminate called without
  an active exception`) *after* writing a complete response — the script tolerates the
  exit status and validates the artefact instead;
* the reference needs the GPU to itself: stop any `rizzo-flow-rs serve` first.

Latest result (`results/report_shape.json`, 22/09/2026):

```
sections: model, mode, answers, calibration, timing
fields compared: 100 numeric, 64 scalar
blocking differences: 0
documented differences: 19        (declared, backend-specific)
informational differences: 100    (values: the models are different by design)
```

`mode` and `calibration` matched with **no** differences at all.

## Declared differences (the only structural ones)

Everything else — `mode`, the whole `answers` structure, `calibration`, and all timing
fields except the three below — is identical.

| path | Python (reference) | this port | why |
|---|---|---|---|
| `model.mlx`, `model.mlx_lm` | MLX / MLX-LM versions | absent | not an MLX runtime |
| `model.requested_revision`, `model.runtime_revision`, `model.source_files`, `model.quantization_group_size` | pinned HF revision, per-file SHA-256, MLX quant group | absent | GGUF carries its own metadata |
| `model.n_gpu_layers`, `model.name`, `model.path` | absent | llama.cpp offload + file identity | llama.cpp runtime |
| `model.source`, `model.device`, `model.precision`, `model.prompt_version`, `model.backend`, `model.fingerprint`, `model.load_seconds` | present | present | same field, different value (different backend) |
| `timing.evaluated_tokens_including_padding` | padded batch positions | absent | an MLX batching artefact; this engine does not pad |
| `timing.peak_mlx_bytes` | MLX peak memory | absent | MLX memory accounting |
| `timing.load_seconds` | reported under `model` | reported under both | convenience |

These are the "declared differences" of SPEC §1-bis point 4: the *identity* reports the
backend actually used, and the *timing* carries backend-specific counters. A client that
reads `answers`, `mode`, `calibration` or the shared timing fields is unaffected.

## Reproducing

```bash
# on the node (needs the Python reference checkout with its venv)
RIZZO_REFERENCE=/mnt/disco1/rizzo-flow \
RIZZO_MODEL=/home/alforiva/inference/models/Qwen3.5-4B-Instruct-SingleTurn.Q4_K_M.gguf \
  parity/run_parity.sh --end-to-end
```

Layer A alone needs no model and runs anywhere the reference checkout is importable.

Reports are written to `$PARITY_OUT` (default `/tmp/parity`); the two committed copies in
`results/` are the ones quoted above.
