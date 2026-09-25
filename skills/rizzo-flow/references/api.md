# Rizzo Flow API reference

Server: `rizzo serve` → `http://127.0.0.1:8017` (flags `--host`, `--port`). One model resident;
concurrent requests are queued and run one at a time (parallelism is *inside* a request).

| Route | Purpose |
| --- | --- |
| `GET /health` | `{"status": "ready", "model": {…metadata, fingerprint…}}` |
| `POST /v1/decisions` | native API (this file, first half) |
| `POST /v1/systemone` | TypeSafe/Jev-compatible API (second half) |
| `GET /v1/models` | model list in TypeSafe format |
| `GET /playground` | interactive builder, probability bars, cURL export |
| `GET /snake` | demo: Snake played by one `choice` per move |

## Native: `POST /v1/decisions`

All objects are **strict**: unknown fields → 422, types are not coerced (`"true"` is not `true`).

### Request

```jsonc
{
  "state": "text" | {…} | […],        // required, non-empty, ≤ 256 KB serialized
  "questions": {                       // 1–64 entries; key = your id (1–128 chars)
    "<id>": <Question>
  },
  "mode": "shared"                     // default; "direct" = no prefix reuse (debug/reference only)
}
```

Common fields of every question:

```jsonc
{
  "type": "boolean" | "choice" | "score" | "numeric",
  "instructions": "…",                 // 1–8000 chars, the question itself
  "policy": {                          // optional
    "allow_abstain": true,             // adds the __insufficient__ candidate
    "max_unavailable_probability": 0.5,// P(abstain/out-of-range) ≥ this → not ok
    "min_top_probability": 0.0         // top p below this → status "uncertain"
  }
}
```

Type-specific fields:

| Type | Fields | Limits |
| --- | --- | --- |
| `boolean` | `true_description`, `false_description` (optional, default "Yes. The evidence supports…") | — |
| `choice` | `options: [{"id": "^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$", "description": "…"}]` | 2–26, unique ids; 25 with abstention |
| `score` | `levels: ["lowest…", …, "highest…"]` | 2–26 distinct; 25 with abstention |
| `numeric` | `unit: "…"`, `anchors: [{"value": number, "description": "…"}]` strictly increasing | 2–24 without abstention, 23 with |

Each candidate — including the special `__insufficient__`, `__below_range__`, `__above_range__` —
is one answer letter A–Z, hence the 26-slot limit.

### Response

```jsonc
{
  "model": {"source": "XHToken/Spark-X2.5-4B", "precision": "q8_0", "weights": "flow",
            "runtime": "llama.cpp", "backend": "cuda", "prompt_version": "spark-decisions-v3",
            "fingerprint": "…", …},
  "mode": "shared",
  "calibration": null,                 // or the loaded temperature fit
  "timing": {"total_seconds": …, "prefill_seconds": …, "shared_prefix_tokens": …, …},
  "answers": {
    "<id>": {
      "type": "choice",
      "status": "ok",                  // ok | insufficient_evidence | out_of_range | uncertain
      "choice": "access",              // primary value; null whenever status != ok
      "probabilities": {"access": 0.97, "billing": 0.02, "sales": 0.005, "__insufficient__": 0.005},
      "option_logits": {…},
      "legend": {"access": "Login, password…", …},   // what each id meant in the prompt
      "uncertainty": {"top_probability": 0.97, "entropy_nats": 0.15,
                      "concentration": 0.89, "unavailable_probability": 0.005},
      "probability_status": "uncalibrated_conditional_option_scores",
      "temperature": 1.0,
      "prompt_sha256": "…", "input_tokens": 212
    }
  }
}
```

Primary value per type (all `null` unless `status == "ok"`):

| Type | Primary field | Extra fields |
| --- | --- | --- |
| `boolean` | `value: bool` | `probability_true_given_available` (P(true) renormalized without abstention) |
| `choice` | `choice: id` | — |
| `score` | `score: float` = mean level index (0…n−1) | `normalized_score` (0–1), `statistics_given_available` {mean, stddev, median, anchor_quantiles p10/p90}, `values`, `support` |
| `numeric` | `value: float` = probability-weighted anchor value | `unit`, `statistics_given_available`, `values`, `support`, `range_probabilities` {below, above} |

Status logic: the winner is an abstain/range candidate, or their total probability ≥
`max_unavailable_probability` → `insufficient_evidence` (or `out_of_range` if below+above beats
insufficient); else top probability < `min_top_probability` → `uncertain`; else `ok`.

Numeric values never leave `[first anchor, last anchor]`; quantiles are those of the discrete
distribution over anchors, not statistical intervals.

## Compatible: `POST /v1/systemone`

Same shape as the public TypeSafe API (<https://docs.typesafe.ai/api>). Optional
`Authorization: Bearer <RIZZO_API_KEY>` (enforced only if the server has that variable).

### Request

```jsonc
{
  "state": "text" | {…} | […],
  "model": "rizzo-latest",             // or the local id, or any "jev-*" alias
  "questions": {
    "is_urgent":  {"type": "noul", "instructions": "Does this convey urgency?",
                   "criteria": {"true": "…optional…", "false": "…optional…"}},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages", "sales": null}},
    "frustration":{"type": "score", "instructions": "How frustrated is the customer?",
                   "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}
```

- `instructions` and criteria values may be strings, objects or arrays (objects are serialized as
  canonical JSON).
- `choice`: 2–26 free-form keys; the value is a description or `null` (the key alone is shown).
- `score`: 2–10 ordered levels.
- No abstention and no numeric type in this format. Unknown top-level fields are ignored; unknown
  fields inside a question → 422.

### Response

```jsonc
{
  "model": "rizzo-flow-4b-q8_0",       // always the local model, never a Jev id
  "answers": {
    "is_urgent":   {"type": "noul", "noul": 0.99998},                  // P(yes)
    "department":  {"type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.99, "technical": 0.01, "sales": 0.0},
                    "confidence": 0.98},
    "frustration": {"type": "score", "score": 1.0,
                    "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                    "probabilities": {"0": 0.0, "1": 0.99, "2": 0.01}, "confidence": 0.99}
  },
  "usage": {"input_tokens": 291, "output_tokens": 0},
  "x_rizzo": {"timing": {…}, "probability_status": […], "fingerprint": "…"}
}
```

`confidence = (n·p_max − 1)/(n − 1)`: the shape of the distribution, not a probability of being right.

## Errors

| Code | When |
| --- | --- |
| 422 | schema violation, too many options, state over 256 KB, prompt over `--ctx` tokens. Body `detail` says which. Nothing is truncated. |
| 401 | `/v1/systemone`, `/v1/models`: API key configured and bearer missing/wrong |
| 400 | `/v1/systemone`: `model` is not `rizzo-latest`, the served id or `jev-*` |

## CLI equivalents

```bash
uv run rizzo decide request.json [--output out.json]   # one request, loads the model itself
uv run rizzo schema [--response]                       # JSON Schema of request/response
uv run rizzo evaluate cases.jsonl                      # accuracy/NLL/Brier/ECE on labelled cases
uv run rizzo calibrate rows.jsonl --fingerprint <fp> --output fit.json
uv run rizzo serve --calibration fit.json              # serve with temperature scaling
```

Server/model options (`serve`, `decide`, `evaluate`): `--size 4b|1.7b`, `--weights flow|base`,
`--quant q8_0|q4_k_m|bf16`, `--device auto|gpu|cpu|cuda|vulkan|metal|rocm|sycl`, `--ctx N`
(tokens per state + question, default 8192; KV cache ~144 KiB/token reserved at start),
`--kv-type q8_0|q4_0` (halves/quarters the cache for long contexts), `--batch-size` (default 4).
