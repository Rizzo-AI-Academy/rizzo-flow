<!-- Aggiornato il 3 novembre 2026 -->

# Rizzo Router (pre-flight model routing)

Ogni prompt passa prima da Rizzo Flow (`POST /v1/decisions`, locale, ~50 ms) con 3 domande:
`complexity` (score 0-2), `target_model` (haiku | sonnet | opus_or_thinking), `needs_local_only` (boolean).

## Regola di decisione (route_prompt.py)

1. `needs_local_only` = true con P >= 0.80 -> modello locale Ollama (`RIZZO_LOCAL_MODEL`), mai cloud.
   Nell'hook il prompt viene bloccato.
2. `target_model` ok con P >= 0.60 -> quel tier.
3. Altrimenti tier da `complexity`: < 0.5 haiku, < 1.5 sonnet, altrimenti opus.
4. Router spento o in errore -> sonnet (fail-open, non blocca mai il lavoro).

Tier -> alias Claude Code dell'abbonamento: haiku, sonnet, opus. Nessuna API a consumo:
`--run` rimuove `ANTHROPIC_API_KEY` dall'ambiente prima di lanciare `claude -p`.

## Comandi (PowerShell, dalla root del repo)

```powershell
uv run --no-sync rizzo serve                                  # terminale 1
curl.exe http://127.0.0.1:8017/health                         # terminale 2
python router/route_prompt.py "rinomina le variabili in camelCase"       # solo decisione
python router/route_prompt.py --payload "..."                 # payload JSON per /v1/decisions
python router/route_prompt.py --run "progetta un sistema di cache distribuita"   # decide ed esegue
$env:RIZZO_LOCAL_MODEL="llama3.1:8b"                          # opzionale, per i prompt sensibili
```

Soglie da tarare su un campione etichettato: `RIZZO_TAU`, `RIZZO_SENSITIVE_TAU`.

## Costo del cambio modello e sub-agent

L'hook legge `transcript_path` e ricava modello attivo e token di contesto dall'ultimo messaggio
assistant. La cache del prompt e' per modello: cambiando, il contesto va riscritto (1.25x) invece di
essere riletto da cache (0.1x per Haiku, 0.05x per Sonnet e Opus). Su `RIZZO_HORIZON` turni (default 6)
confronta costo restando e cambiando, pesato con `RIZZO_PRICES` (default 0.10,2,4 $/MTok di input:
listino Haiku/Sonnet/Opus 5.5 da platform.claude.com/docs/en/about-claude/pricing; l'output, che costa
5x l'input, e la soglia di 100k token di Haiku non sono modellati):

- stesso tier: nessun messaggio (zero contesto aggiunto);
- downgrade: suggerito solo se cambiando costa < 80% del restare;
- upgrade: suggerito solo se `target_model` e' sicuro (qualita', non costo).

Sub-agent: partono con contesto vuoto, quindi nessun costo di cambio. Un modello per ciascuno:

```powershell
python router/route_prompt.py --tasks tasks.json   # lista di stringhe o {"id","prompt"}
```
Rizzo Flow non intercetta lo spawn: l'agente principale deve chiamare `--tasks` e passare il `model` ad `Agent`.

## Cambio modello reale: `--chat`

Un hook non puo' cambiare il modello della sessione, e `set_session_model` e' rifiutato sulla
sessione stessa. `--chat` e' un REPL che ad ogni prompt sceglie il modello (stessa regola di costo
dell'hook, ma applicata) ed esegue `claude -p --resume <id> --model <alias> --output-format json`.
La storia resta nella stessa sessione; il contesto si ricava da `usage`.

```powershell
python router/route_prompt.py --chat
$env:RIZZO_CLAUDE_ARGS="--permission-mode acceptEdits"   # -p non puo' chiedere permessi
```
Limiti: niente TUI interattiva (modalita' headless), nessuno streaming; prompt sensibili -> Ollama o scartati.

## Sub-agent automatici (hook PreToolUse)

`install_global.py` registra anche `route_prompt.py --pretool` su `Agent|Task`. Ad ogni spawn Rizzo
classifica `description`+`prompt` e imposta `model` (haiku/sonnet/opus) via `updatedInput`.
Se il `model` e' gia' esplicito lo rispetta (`RIZZO_OVERRIDE=1` per sovrascrivere); prompt sensibili
-> spawn negato; router spento -> nessuna modifica. Gli hook si caricano all'avvio della sessione:
riavviare Claude Code dopo l'installazione.
