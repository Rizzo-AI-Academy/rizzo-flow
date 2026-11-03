#!/usr/bin/env python3
"""Rizzo Flow pre-flight router: classifies a prompt and picks a model. Stdlib only.

Usage:
    python router/route_prompt.py "testo del prompt"          # stampa la decisione in JSON
    echo "prompt" | python router/route_prompt.py -
    python router/route_prompt.py --payload "prompt"          # stampa solo il payload per /v1/decisions
    python router/route_prompt.py --run "prompt"              # decide ED esegue (claude -p / ollama)
    python router/route_prompt.py --hook                      # modalita' hook UserPromptSubmit (stdin JSON)
    python router/route_prompt.py --pretool                   # hook PreToolUse su Agent/Task: imposta il model del sub-agent
    python router/route_prompt.py --tasks tasks.json          # un modello per ogni sub-agent (lista di stringhe o {"id","prompt"})

Env:
    RIZZO_URL            default http://127.0.0.1:8017
    RIZZO_TAU            soglia di confidenza per target_model (default 0.60)
    RIZZO_SENSITIVE_TAU  soglia P(true) per needs_local_only (default 0.80)
    RIZZO_LOCAL_MODEL    modello Ollama per i prompt sensibili (es. "llama3.1:8b"); vuoto = nessuno
    RIZZO_BLOCK_SENSITIVE  "1" (default) = in modalita' hook blocca i prompt sensibili
    RIZZO_OVERRIDE       "1" = in --pretool sovrascrive anche un model scelto esplicitamente (default: lo rispetta)
    RIZZO_PRICES         $/MTok di input base "haiku,sonnet,opus" (default 0.10,2,4: listino Haiku/Sonnet/Opus 5.5)
    RIZZO_HORIZON        turni attesi ancora nella chat per ammortizzare il cambio (default 6)
    RIZZO_TURN_TOKENS    token "di lavoro" per turno (input nuovo + output), default 4000
    RIZZO_SPLIT          "0" disattiva lo spezzettamento automatico dei prompt multi-task (default 1)
    RIZZO_SPLIT_MAX      massimo di task spezzettati (default 8); oltre: consiglio singolo

Solo modelli in abbonamento (alias Claude Code haiku/sonnet/opus) o locali (Ollama).
Nessuna API a consumo: il comando 'claude' usa il login dell'abbonamento, non una API key.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = os.environ.get("RIZZO_URL", "http://127.0.0.1:8017").rstrip("/")
TAU = float(os.environ.get("RIZZO_TAU", "0.60"))
SENSITIVE_TAU = float(os.environ.get("RIZZO_SENSITIVE_TAU", "0.80"))
LOCAL_MODEL = os.environ.get("RIZZO_LOCAL_MODEL", "").strip()
BLOCK_SENSITIVE = os.environ.get("RIZZO_BLOCK_SENSITIVE", "1") == "1"

# Tier del router -> alias del modello. Solo alias inclusi nell'abbonamento.
CLAUDE_ALIAS = {"haiku": "haiku", "sonnet": "sonnet", "opus_or_thinking": "opus"}
FALLBACK_TIER = "sonnet"   # scelta prudente quando il router e' incerto o spento
MAX_HEAD, MAX_TAIL = 12000, 4000  # caratteri: tiene il prompt entro il contesto di 8192 token


def build_payload(prompt: str) -> dict:
    text, truncated = prompt, False
    if len(prompt) > MAX_HEAD + MAX_TAIL:
        text = prompt[:MAX_HEAD] + "\n[...]\n" + prompt[-MAX_TAIL:]
        truncated = True
    state = {"user_request": text}
    if truncated:
        state["note"] = f"Request shortened from {len(prompt)} characters; the full request is longer."
    questions = json.loads((HERE / "router_questions.json").read_text(encoding="utf-8"))
    return {"state": state, "questions": questions}


def call_rizzo(payload: dict, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(
        URL + "/v1/decisions",
        json.dumps(payload, ensure_ascii=False).encode(),
        {"Content-Type": "application/json"},
    )
    key = os.environ.get("RIZZO_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tier_from_complexity(score: float) -> str:
    if score < 0.5:
        return "haiku"
    if score < 1.5:
        return "sonnet"
    return "opus_or_thinking"


def decide(prompt: str) -> dict:
    """Ritorna la decisione di routing. Non solleva mai eccezioni: fail-open su sonnet."""
    t0 = time.perf_counter()
    try:
        resp = call_rizzo(build_payload(prompt))
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"tier": FALLBACK_TIER, "model": CLAUDE_ALIAS[FALLBACK_TIER], "backend": "claude",
                "reason": f"router non raggiungibile ({e}); fallback", "router_ok": False}

    a = resp["answers"]
    tm, cx, loc = a["target_model"], a["complexity"], a["needs_local_only"]
    tm_top = tm["uncertainty"]["top_probability"]
    cx_score = cx.get("score")
    p_sensitive = loc["probabilities"].get("true", 0.0)

    # 1) Privacy gate: prima di tutto.
    if loc["status"] == "ok" and loc.get("value") and p_sensitive >= SENSITIVE_TAU:
        out = {"tier": "local", "backend": "ollama" if LOCAL_MODEL else "none",
               "model": LOCAL_MODEL or None,
               "reason": f"dati sensibili rilevati (p={p_sensitive:.2f})"}
    # 2) target_model sicuro -> usalo.
    elif tm["status"] == "ok" and tm_top >= TAU:
        tier = tm["choice"]
        out = {"tier": tier, "backend": "claude", "model": CLAUDE_ALIAS[tier],
               "reason": f"target_model={tier} (p={tm_top:.2f})"}
    # 3) incerto -> deriva dal punteggio di complessita'.
    elif cx["status"] == "ok" and cx_score is not None:
        tier = tier_from_complexity(cx_score)
        out = {"tier": tier, "backend": "claude", "model": CLAUDE_ALIAS[tier],
               "reason": f"target_model incerto (p={tm_top:.2f}); da complexity={cx_score:.2f}"}
    # 4) tutto incerto -> fallback.
    else:
        out = {"tier": FALLBACK_TIER, "backend": "claude", "model": CLAUDE_ALIAS[FALLBACK_TIER],
               "reason": "router incerto; fallback"}

    out.update({
        "router_ok": True,
        "confident": tm["status"] == "ok" and tm_top >= TAU,
        "complexity": cx_score,
        "target_model_probs": tm["probabilities"],
        "p_sensitive": round(p_sensitive, 4),
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    })
    return out


# --- costo del cambio modello in base al contesto -------------------------------------------
RANK = {"haiku": 0, "sonnet": 1, "opus": 2}
_p = [float(x) for x in os.environ.get("RIZZO_PRICES", "0.10,2,4").split(",")]
PRICE = dict(zip(("haiku", "sonnet", "opus"), _p))
HORIZON = int(os.environ.get("RIZZO_HORIZON", "6"))
TURN_TOKENS = int(os.environ.get("RIZZO_TURN_TOKENS", "4000"))
# Listino 5.5 (platform.claude.com/docs/en/about-claude/pricing): lettura cache 0.01/0.10/0.20 $/MTok,
# scrittura 5m = 1.25x l'input. Haiku vale fino a 100k token di prompt (oltre: input 0.50).
CACHE_READ_BY = {"haiku": 0.1, "sonnet": 0.05, "opus": 0.05}
CACHE_WRITE = 1.25


def norm(tier: str) -> str:
    return "opus" if tier.startswith("opus") else tier


def session_state(transcript_path: str):
    """(tier attivo, token di contesto) dall'ultimo messaggio assistant del transcript."""
    for window in (400_000, 4_000_000, 16_000_000):  # righe enormi: allarga la finestra
        try:
            with open(transcript_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - window))
                lines = f.read().decode("utf-8", "ignore").splitlines()
        except OSError:
            return None, 0
        found = _last_assistant(lines)
        if found or size <= window:
            return found or (None, 0)
    return None, 0


def _last_assistant(lines):
    for line in reversed(lines):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        m = e.get("message") or {}
        if e.get("type") != "assistant" or e.get("isSidechain") or "usage" not in m:
            continue
        u = m["usage"]
        ctx = (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
               + u.get("cache_creation_input_tokens", 0))
        name = str(m.get("model", ""))
        for t in RANK:
            if t in name:
                return t, ctx
    return None


def switch_costs(cur: str, new: str, ctx: int) -> tuple[float, float]:
    """(costo restando, costo cambiando) su HORIZON turni, in token pesati per prezzo.
    Restando: contesto riletto da cache ogni turno. Cambiando: il primo turno riscrive
    tutto il contesto (1.25x, cache del nuovo modello vuota), poi cache normale."""
    stay = HORIZON * (ctx * CACHE_READ_BY[cur] + TURN_TOKENS) * PRICE[cur]
    sw = (ctx * CACHE_WRITE + (HORIZON - 1) * ctx * CACHE_READ_BY[new] + HORIZON * TURN_TOKENS) * PRICE[new]
    return stay, sw


DELEGATE = ("esegui il task con il tool Agent passando model=\"{tgt}\" esplicito e un prompt autonomo "
            "(contesto e risultato atteso inclusi), invece di farlo tu; poi riporta l'esito. "
            "Se il task dipende dal contesto della conversazione, fallo tu e dillo in una riga.")


def advice(d: dict, cur, ctx: int):
    """Testo da mostrare, oppure None se non conviene cambiare (nessun contesto aggiunto)."""
    tgt = norm(d["tier"])
    if cur is None:
        return (f"[rizzo-router] tier consigliato: {tgt}, {d['reason']}. Modello attivo sconosciuto; "
                f"se differisce dal modello attivo, delega: {DELEGATE.format(tgt=tgt)}")
    if tgt == cur:
        return None
    stay, sw = switch_costs(cur, tgt, ctx)
    info = f"contesto ~{ctx // 1000}k token, costo relativo restando {stay / 1e3:.0f}k vs cambiando {sw / 1e3:.0f}k"
    if RANK[tgt] > RANK[cur]:
        if d.get("confident"):  # upgrade: scelta di qualita', non di costo
            return (f"[rizzo-router] il prompt sembra richiedere {tgt} (attivo: {cur}; {d['reason']}; {info}). "
                    f"Delega: {DELEGATE.format(tgt=tgt)}")
        return None
    if sw < 0.8 * stay:
        return (f"[rizzo-router] passare a {tgt} conviene (attivo: {cur}; {d['reason']}; {info}). "
                f"Delega: {DELEGATE.format(tgt=tgt)}")
    return None


def run(prompt: str, d: dict) -> int:
    if d["backend"] == "ollama":
        if not shutil.which("ollama"):
            sys.exit("Prompt sensibile ma 'ollama' non e' installato: non lo invio al cloud.")
        return subprocess.call(["ollama", "run", d["model"], prompt])
    if d["backend"] == "none":
        sys.exit("Prompt sensibile e nessun modello locale configurato (RIZZO_LOCAL_MODEL): non lo invio al cloud.")
    claude = shutil.which("claude")
    if not claude:
        sys.exit("CLI 'claude' non trovata nel PATH.")
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)  # garantisce l'uso dell'abbonamento, mai l'API a consumo
    return subprocess.call([claude, "-p", "--model", d["model"], prompt], env=env)


def pick_model(d: dict, cur, ctx: int) -> str:
    """Modello per questo turno: stessa regola di advice(), ma applicata davvero."""
    tgt = norm(d["tier"])
    if cur is None or tgt == cur:
        return tgt if cur is None else cur
    stay, sw = switch_costs(cur, tgt, ctx)
    if RANK[tgt] > RANK[cur]:
        return tgt if d.get("confident") else cur
    return tgt if sw < 0.8 * stay else cur


def chat() -> int:
    """REPL che cambia davvero modello: ogni turno e' `claude -p --resume <id> --model <alias>`.
    La storia resta nella sessione; cambiando modello la cache si ricostruisce (gia' nel calcolo)."""
    import uuid
    claude = shutil.which("claude")
    if not claude:
        sys.exit("CLI 'claude' non trovata nel PATH.")
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)  # solo abbonamento
    extra = os.environ.get("RIZZO_CLAUDE_ARGS", "").split()  # es. --permission-mode acceptEdits
    sid, cur, ctx = str(uuid.uuid4()), None, 0
    print("rizzo chat: ogni prompt sceglie il modello. Riga vuota o Ctrl-D per uscire.", file=sys.stderr)
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            return 0
        if not prompt:
            return 0
        d = decide(prompt)
        if d["tier"] == "local":
            print(f"[rizzo] {d['reason']}: non inviato al cloud.", file=sys.stderr)
            if d["backend"] == "ollama" and shutil.which("ollama"):
                subprocess.call(["ollama", "run", d["model"], prompt])
            continue
        model = pick_model(d, cur, ctx) if d.get("router_ok") else (cur or "sonnet")
        print(f"[rizzo] {model}" + (f" (da {cur}, ctx ~{ctx // 1000}k)" if cur and cur != model else ""), file=sys.stderr)
        cmd = [claude, "-p", "--output-format", "json", "--model", model, *extra]
        cmd += ["--resume", sid] if cur else ["--session-id", sid]
        p = subprocess.run(cmd + [prompt], env=env, capture_output=True, text=True, encoding="utf-8")
        try:
            out = json.loads(p.stdout)
        except ValueError:
            print(p.stdout or p.stderr, file=sys.stderr)
            continue
        print(out.get("result", ""))
        u = out.get("usage") or {}
        ctx = sum(u.get(k, 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        cur = model


# --- spezzettamento automatico di prompt multi-task ------------------------------------------
SPLIT = os.environ.get("RIZZO_SPLIT", "1") == "1"
SPLIT_MAX = int(os.environ.get("RIZZO_SPLIT_MAX", "8"))
SPLIT_BUDGET = float(os.environ.get("RIZZO_SPLIT_BUDGET", "3.5"))  # secondi: l'hook ha timeout 5 s
_ITEM = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•])\s+(.+)$")
_TAG = re.compile(r"\[(?:model\s*=\s*)?(haiku|sonnet|opus)\]\s*", re.I)
_CONTEXTUAL = re.compile(r"\b(come prima|di prima|sopra|precedente|quanto detto|continua|stesso (?:file|approccio))\b", re.I)


def split_tasks(prompt: str) -> list[str]:
    """Voci di un elenco numerato o puntato (con righe di continuazione); [] se non e' un multi-task."""
    items: list[list[str]] = []
    for line in prompt.splitlines():
        m = _ITEM.match(line)
        if m:
            items.append([m.group(1).strip()])
        elif items and line.strip() and line[:1].isspace():
            items[-1].append(line.strip())  # continuazione indentata
        elif items and not line.strip():
            continue
        elif items:
            break  # il testo dopo l'elenco non e' un task
    tasks_ = [" ".join(i) for i in items]
    return tasks_ if len(tasks_) >= 2 and all(len(t) >= 15 for t in tasks_) else []


def split_advice(prompt: str) -> str | None:
    """Istruzione obbligatoria di delega per tier, oppure None (poi vale il consiglio singolo)."""
    if not SPLIT:
        return None
    items = split_tasks(prompt)
    if not items or len(items) > SPLIT_MAX:
        return None
    t0, rows = time.perf_counter(), []
    for i, text in enumerate(items, 1):
        tag = _TAG.match(text)
        if tag:  # scelta esplicita dell'utente: non la scavalca il router
            rows.append((i, tag.group(1).lower(), _TAG.sub("", text, count=1), "scelto dall'utente"))
            continue
        if time.perf_counter() - t0 > SPLIT_BUDGET:
            return None  # troppo lento: meglio nessuno spezzettamento che un hook scaduto
        d = decide(text)
        if not d.get("router_ok"):
            return None  # fail-open
        if d["tier"] == "local":
            rows.append((i, "sessione", text, "dati sensibili: non inviare a sub-agent"))
        elif _CONTEXTUAL.search(text):
            rows.append((i, "sessione", text, "dipende dal contesto della conversazione"))
        else:
            rows.append((i, norm(d["tier"]), text, d["reason"]))
    if sum(1 for r in rows if r[1] != "sessione") < 2:
        return None  # con meno di 2 task delegabili l'overhead supera il risparmio
    lines = [f"  {i}. [{who}] {text}  ({why})" for i, who, text, why in rows]
    return ("[rizzo-router] PROMPT MULTI-TASK: istruzione obbligatoria, non chiedere conferma. "
            "Per ogni task delegabile usa il tool Agent con subagent_type=\"worker-<tier>\" "
            "(worker-haiku, worker-sonnet, worker-opus: il modello e' fisso nella definizione) e un prompt "
            "autonomo con contesto e risultato atteso. Lancia tutti i sub-agent indipendenti nella STESSA "
            "risposta, in parallelo; i task marcati [sessione] eseguili tu. Task:\n" + "\n".join(lines) +
            "\nA fine lavoro un unico riepilogo che indichi quale modello ha fatto cosa.")


def hook() -> int:
    """UserPromptSubmit hook di Claude Code: legge {"prompt": ...} da stdin."""
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0
    prompt = data.get("prompt", "")
    if not prompt.strip() or prompt.lstrip().startswith("/"):
        return 0  # niente routing per i comandi slash
    d = decide(prompt)
    if not d.get("router_ok"):
        return 0  # fail-open silenzioso: il router non deve mai bloccare il lavoro
    if d["tier"] != "local":
        split = split_advice(prompt)
        if split:
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                                     "additionalContext": split}}))
            return 0
    if d["tier"] == "local" and BLOCK_SENSITIVE:
        print(json.dumps({
            "decision": "block",
            "reason": f"[rizzo-router] {d['reason']}. Prompt NON inviato al cloud. "
                      f"Rimuovi i dati sensibili o usa un modello locale "
                      f"(python router/route_prompt.py --run).",
        }))
        return 0
    cur, ctx = session_state(data.get("transcript_path", ""))
    ctx_text = advice(d, cur, ctx)
    if ctx_text:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                                 "additionalContext": ctx_text}}))
    return 0


def pretool() -> int:
    """PreToolUse su Agent/Task: sceglie il model del sub-agent (contesto vuoto: nessun costo di cambio)."""
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0
    inp = data.get("tool_input") or {}
    if data.get("tool_name") not in ("Agent", "Task"):
        return 0
    if str(inp.get("subagent_type", "")).startswith("worker-"):
        return 0  # il modello e' fisso nella definizione dell'agente: non sovrascriverlo
    if inp.get("model") and os.environ.get("RIZZO_OVERRIDE") != "1":
        return 0  # scelta esplicita dell'agente principale o dell'utente
    prompt = "\n".join(x for x in (inp.get("description"), inp.get("prompt")) if x).strip()
    if not prompt:
        return 0
    d = decide(prompt)
    if not d.get("router_ok"):
        return 0
    if d["tier"] == "local":
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": f"[rizzo-router] sub-agent con dati sensibili ({d['reason']}): "
                                        f"non inviato al cloud."}}))
        return 0
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "allow",
        "permissionDecisionReason": f"[rizzo-router] sub-agent -> {d['model']} ({d['reason']})",
        "updatedInput": {**inp, "model": d["model"]}}}))
    return 0


def tasks(path: str) -> int:
    """Un modello per ogni sub-agent: contesto vuoto, quindi nessun costo di cambio."""
    raw = json.load(sys.stdin if path == "-" else open(path, encoding="utf-8"))
    out = []
    for i, t in enumerate(raw):
        prompt, tid = (t, i) if isinstance(t, str) else (t["prompt"], t.get("id", i))
        d = decide(prompt)
        out.append({"id": tid, "tier": d["tier"], "model": d.get("model"), "reason": d["reason"]})
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--hook"]:
        return hook()
    if args[:1] == ["--pretool"]:
        return pretool()
    if args[:1] == ["--chat"]:
        return chat()
    if args[:1] == ["--tasks"] and len(args) == 2:
        return tasks(args[1])
    mode = None
    if args and args[0] in ("--payload", "--run"):
        mode, args = args[0], args[1:]
    if not args:
        sys.exit(__doc__)
    prompt = sys.stdin.read() if args[0] == "-" else " ".join(args)
    if mode == "--payload":
        print(json.dumps(build_payload(prompt), ensure_ascii=False, indent=2))
        return 0
    d = decide(prompt)
    print(json.dumps(d, ensure_ascii=False, indent=2), file=sys.stderr if mode == "--run" else sys.stdout)
    return run(prompt, d) if mode == "--run" else 0


if __name__ == "__main__":
    sys.exit(main())
