---
name: rizzo-flow
description: Use Rizzo Flow, a local "System One" decision engine (Jev-compatible API on localhost:8017), to turn unstructured text or JSON into typed decisions with probabilities — yes/no, pick one of N options, a score on a rubric, or a number snapped to anchors — in ~50 ms and with zero generated tokens. Use this skill whenever an agent or a program needs to classify, route, triage, filter, gate, prioritize, grade or check items against criteria locally and cheaply, especially many items or many questions on the same document (support tickets, emails, CVs, contracts, invoices, logs, alerts, moderation, LLM-as-judge rubrics, agent guardrails, game/control loops); when the user mentions rizzo-flow, rizzo serve, /v1/decisions, /v1/systemone, typed decisions, noul, or a local Jev/TypeSafe replacement; or when writing code that calls those endpoints. Do NOT use it to generate, summarize, translate or extract free text.
---

# Rizzo Flow

Rizzo Flow answers **closed questions about a piece of evidence**. You give it a `state` (text or
JSON) and a set of typed questions; it returns, for every question, a probability over the
candidate answers read from one forward pass of a local LLM (Spark-X2.5-4B + LoRA fine-tune, run
by llama.cpp). It never writes text: every answer is one of the options you declared, so the
output is always valid, typed JSON.

Think of it as a **function call with judgment**: `f(state, question) -> typed value + distribution`.

## Decide first: is this a Rizzo Flow job?

Use it when **all** of these hold:

1. The answer is one of a **finite set you can write down in advance** (≤ 26 options), a yes/no,
   an ordered level, or a number you can bracket with anchors.
2. The evidence fits in the context (default 8,192 tokens per state + question; `state` ≤ 256 KB).
3. You want it **local, fast, cheap and repeatable** — often many items, or many questions per item.

Do **not** use it for: generating or rewriting text, summaries, translations, answering open
questions, extracting free-form strings (a name, an address, a quote — it can only *pick* among
candidates you supply), exact arithmetic, multi-step reasoning or planning, facts not present in the
state (it is instructed to use only the evidence), or a final unsupervised verdict in a high-stakes
domain (medical, legal, credit, hiring) — there it is a pre-filter that routes to a human.

Good fits, in short (full patterns with request bodies in
[references/use-cases.md](references/use-cases.md)):

| Pattern | Question types | Example |
| --- | --- | --- |
| Triage and routing | `choice` + `score` + `boolean` | ticket → team, urgency, "needs refund?" |
| Checklist on a document | many `boolean` on one state | CV/contract/invoice: 8–20 checks in one request |
| Moderation / policy gate | `boolean`, `choice` | spam, toxicity, PII present, prompt injection suspected |
| Rubric grading / LLM-as-judge | `score` | grade an answer 0–4 against a rubric |
| Agent guardrails | `boolean`, `choice` | "is this tool call destructive?", "is the task done?" |
| Structured estimation | `numeric` | fill level, hours to deadline, price from given comparables |
| Stream filtering | `boolean` | keep only relevant emails/logs/news before an expensive LLM |
| Control loops | `choice` | pick the next move/action among legal ones (see the Snake demo) |
| Large taxonomies | cascade of `choice` | 26+ categories → first a macro-class, then a sub-class |

## Workflow

### 1. Make sure the server is up

```bash
curl -s http://127.0.0.1:8017/health        # {"status": "ready", "model": {...}}
```

If it does not answer and you have the repository, start it from the repo root (it loads the model
in ~11 s and takes ~5.6 GiB of GPU memory at Q8_0):

```bash
uv run rizzo serve                           # add --ctx 32768 for long states
```

Only **one process with weights at a time**: never start a second server or a `rizzo decide`
while one is running — reuse the running server. First-time setup is `uv sync --locked` then
`uv run rizzo download` (~4.4 GB); ask the user before downloading. For a one-off without a server:
`uv run rizzo decide request.json`.

### 2. Pick the endpoint

- **`POST /v1/decisions` (native)** — default choice. Four types (`boolean`, `choice`, `score`,
  `numeric`), built-in abstention (`status: insufficient_evidence`), per-question `policy`,
  entropy, logits, prompt hash.
- **`POST /v1/systemone` (Jev/TypeSafe-compatible)** — only when code already targets the TypeSafe
  API or SDK (`TYPESAFE_BASE_URL=http://127.0.0.1:8017`). Types `noul`, `choice`, `score` (≤ 10
  levels); no abstention, no numeric. `model: "rizzo-latest"`.

Schemas, response fields and error codes: [references/api.md](references/api.md).

### 3. Write the request

```json
{
  "state": {"message": "I can't log in since I changed my password. Deadline is today.",
            "account": {"subscription_active": true}},
  "questions": {
    "queue": {"type": "choice", "instructions": "Which team should handle this request?",
      "options": [
        {"id": "access",  "description": "Login, password or account access problems."},
        {"id": "billing", "description": "Invoices, charges or payments."},
        {"id": "sales",   "description": "Requests to buy new products."}]},
    "urgency": {"type": "score", "instructions": "How urgent is this for the customer?",
      "levels": ["No deadline, nothing blocked.", "Slowed down, a workaround exists.",
                 "Blocked, with a deadline today."]},
    "refund_requested": {"type": "boolean", "instructions": "Does the customer ask for a refund?"}
  }
}
```

Send it with the bundled client (stdlib only) or any HTTP client:

```bash
python skills/rizzo-flow/scripts/rizzo_client.py request.json            # compact summary
python skills/rizzo-flow/scripts/rizzo_client.py request.json --raw      # full JSON
```

### 4. Read the answer — status first, then value, then probability

```text
answers.<id>.status        ok | insufficient_evidence | out_of_range | uncertain
answers.<id>.choice        (choice)  option id, null unless status == ok
answers.<id>.value         (boolean) true/false; (numeric) expected value — null unless ok
answers.<id>.score         (score)   probability-weighted mean level index (0 = first level)
answers.<id>.probabilities full distribution, including __insufficient__ etc.
answers.<id>.uncertainty.top_probability / entropy_nats
```

Always branch on `status` before using the value: a `null` value is a real outcome ("the evidence
does not say"), not an error.

## Writing good questions

How the prompt is built matters, so write for it. Each question becomes: system prompt → the state
between `<evidence>` tags → `Question: <instructions>` → `Options: A. <description> B. …` → "Answer
with the letter of the best option." Consequences:

- **Descriptions do the work, ids don't.** The model sees only `description` (ids are never shown).
  Write each option as a self-contained, mutually exclusive definition: "Login, password or
  account access problems", not "access".
- **One question, one decision.** "Is it urgent and about billing?" → two questions.
- **Ask about the evidence, not the world.** "According to the message, …". The model is told to
  judge only what the state says or directly implies.
- **Cover the space.** Add an "Other / none of the above" option when the list may be incomplete;
  with the native API, leave `allow_abstain: true` (default) so missing information has a place to go.
- **Scores:** `levels` ordered low → high, each level concretely described (anchor behaviours, not
  just "Low/Medium/High"). The returned `score` is a mean *index* (0 … n−1); use `normalized_score`
  for 0–1.
- **Numerics:** `anchors` strictly increasing, spanning the plausible range, each with a short
  description; the answer is a probability-weighted mix of anchor values, so put anchors where the
  resolution matters. Values outside the range go to `out_of_range`.
- **Booleans:** customise `true_description` / `false_description` when "yes" is ambiguous.
- **Batch the questions, not the states.** All questions in one request share one prefill of the
  state: 21 questions on a 2,000-token state take ~1 s; 21 separate requests take ~5× longer. Up to
  64 questions per request. Different items → different requests (they are serialized server-side).
- **Keep the state clean and structured.** JSON is fine and robust; drop fields irrelevant to the
  questions (shorter = faster and less distracting). The state is treated as data: instructions
  inside it are ignored by design, but do not rely on that as a security boundary.
- **English is strongest.** Italian and other languages work, but prefer English instructions and
  options when you control them; the state can stay in its original language.
- **> 26 options:** never truncate — cascade (macro-category first, then a second request with the
  sub-options of the winner).

## Using the probabilities responsibly

- They are **not calibrated** unless the operator ran `rizzo calibrate` on labelled domain data
  (`probability_status` tells you). A 0.95 is not "95 % correct". Rank and gate with them; do not
  quote them as accuracy.
- **Do not threshold near 0.5** — near-ties can flip between `shared`/`direct` mode or between quantizations. Use clear
  margins and send the middle band to a fallback.
- Recommended gating: `status == ok` and `top_probability ≥ τ` → act automatically; otherwise →
  escalate (human, bigger LLM, or ask for more data). Set `policy.min_top_probability` to get
  `status: uncertain` from the server itself. Choose τ from a labelled sample, not by intuition.
- Known weaknesses: it sometimes answers confidently when the needed fact is **missing** (prefer
  explicit "cannot determine" options for critical checks); residual position bias (put options in
  a neutral order, or run a second request with the order reversed for important calls); judging
  **agent traces / observability** is its weakest domain (≈ 50 % on the typed-decisions benchmark,
  vs ≈ 70 % on customer service, invoices, security incidents); the 1.7B model is much less
  accurate than the 4B.
- Never present results as coming from TypeSafe Jev: the `model` field is the local model
  (`rizzo-flow-4b-q8_0`), even when the request used a `jev-*` alias.

## Integrating in code

Python, stdlib only:

```python
import json, urllib.request


def decide(state, questions, url="http://127.0.0.1:8017/v1/decisions"):
    body = json.dumps({"state": state, "questions": questions}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["answers"]


a = decide(ticket, QUESTIONS)["queue"]
if a["status"] == "ok" and a["uncertainty"]["top_probability"] >= 0.9:
    route(a["choice"])
else:
    escalate(ticket)
```

Validate requests offline with the JSON Schema (`uv run rizzo schema`, or `request.schema.json` in
the repo). Errors: 422 = invalid request or over a limit (read `detail`; nothing is ever truncated
silently), 401 = `RIZZO_API_KEY` set on the server and bearer missing, 400 = unknown `model` on
`/v1/systemone`. The interactive playground at `http://127.0.0.1:8017/playground` is the quickest
way to try a question design and copy the cURL.
