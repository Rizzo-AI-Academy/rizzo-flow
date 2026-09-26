# rizzo-flow-rs

**Rizzo Flow, in Rust** — port del pattern *Jev / System One* con backend **llama.cpp (GGUF)**:
state non strutturato → decisioni tipizzate con probabilità (`boolean` · `choice` · `score` ·
`numeric`), **zero token generati**, **multi-modello**, API e CLI **compatibili con Rizzo Flow**.

> **Fork candidato**: questo port è pensato come **fork di [Rizzo Flow](https://github.com/Rizzo-AI-Academy/rizzo-flow)**
> (© Rizzo AI Academy, Apache-2.0) e sarà proposto a **Simone Rizzo** come alternativa Rust al
> runtime Python. Ispirazione a monte: **Jev** (© TypeSafe). La scelta di adottarlo è loro.

- **Spec completa**: [SPEC.md](SPEC.md) · **Report di verifica**: [REPORT.md](REPORT.md)
- **Suite di parità**: [parity/README.md](parity/README.md) · **Attribuzione**: [NOTICE](NOTICE)

## Stato (22/09/2026)

| area | stato |
|---|---|
| core decisionale (`schema`, `prompts`, `decisions`) | ✅ portato — 10 test originali, parità esatta su 20 casi golden |
| backend `llama.cpp` / CUDA | ✅ verde sul nodo (RTX 2070), modelli ibridi inclusi |
| CLI drop-in | ✅ `devices` `schema` `download` `calibrate` `decide` `evaluate` `serve` |
| server | ✅ `/health` `/v1/decisions` `/v1/systemone` `/v1/models` + auth bearer |
| contratto TypeSafe (`compat`) | ✅ portato (`noul`/`choice`/`score`, `usage`, `x_rizzo`) |
| calibrazione | ✅ `fit_temperature` + `calibrate` (parità numerica col Python; percorso end-to-end provato — [calibration/pilot](calibration/pilot/README.md)) |
| benchmark riproducibile | ✅ `evaluate` (coverage, NLL/Brier/ECE, latenze, confronto modalità) |
| test | **44 verdi** (locali e sul nodo con CUDA) |
| playground/snake/logo (UI dev) | ⬜ non portati (rispondono 501, dichiarato) |
| etichette multi-token (>26 opzioni), `/metrics` | ⬜ non ancora |

## Numeri misurati (nodo2070, RTX 2070 8 GB)

Con `Qwen3.5-4B-Instruct-SingleTurn` Q4_K_M (2,7 GB), richieste mono-domanda, caldo:

| | valore |
|---|---|
| decisione | **0,268 s** mediana · 0,28 s p95 |
| throughput | 3,7 decisioni/s |
| carico modello | ~1,3 s |
| test canone 6 domande | **5/6** (stessa divergenza di Spark su s3), ~1,1-1,4 s totali |
| VRAM residente | ~3,4 GB |

Riferimento: Rizzo Flow (Spark-X2.5-4B, MLX-CUDA, Q8) sullo stesso test → 5/6 in **~27 s**
inferenza. Caveat e dettagli in [REPORT.md](REPORT.md).

## Verifiche

```
parity/run_parity.sh --end-to-end     # parità col riferimento Python (due livelli)
cargo test                            # 44 test
```

- **Livello A** (core, senza modello): 20/20 casi identici, 515 campi confrontati,
  delta massimo **4,4e-16**, zero differenze.
- **Livello B** (end-to-end, modelli diversi): **zero differenze strutturali** bloccanti;
  19 differenze *dichiarate* (identità backend + contatori MLX), 100 informative (valori).

## Come si accende

```bash
# build (CPU)
cargo build --release --features llama
# build con CUDA
cargo build --release --features cuda

# una decisione
rizzo-flow-rs decide request.json --model model.gguf --ctx 4096 --ngl 99

# server (auth opzionale via RIZZO_API_KEY)
rizzo-flow-rs serve --model model.gguf --host 127.0.0.1 --port 8017

# schemi del contratto, calibrazione, benchmark
rizzo-flow-rs schema [--response]
rizzo-flow-rs calibrate rows.jsonl --fingerprint <fp> --output calibration.json
rizzo-flow-rs evaluate fixtures.jsonl --model model.gguf --repeats 2 --compare-modes
```

## Differenza chiave rispetto al progetto originale

Non più «llama.cpp invece di MLX» — il progetto originale ha adottato llama.cpp come
runtime predefinito (dopo la PR #1 di un contributore). La differenza di questo port è
**Rust invece di Python**: un binario singolo, nessuno stack Python/uv, nessuna gestione
di runtime prebuilt, e l'apparato di verifica (`parity/`, `calibration/pilot/`, REPORT.md).
Funziona con **qualsiasi GGUF** supportato dal llama.cpp incorporato.

**Spark-X2.5 gira, ed è verificato**: era il modello del progetto che questo port non
riusciva a caricare. Due blocchi, entrambi risolti e misurati ([REPORT.md](REPORT.md) §3.11):
la **dipendenza** (binding vendorizzato su llama.cpp **b11081**, con tre adattamenti
documentati nel wrapper C++) e il **formato del prompt** (`llama_chat_apply_template` rende
solo i template che riconosce; quello di Spark usa costrutti solo-HuggingFace, quindi il port
ora lo rende nativamente in `src/chat_format.rs`). Risultato: **`prompt_sha256` identico
6/6** e **risposte identiche 6/6** al riferimento, con delta massimo sulle probabilità
4,06e-02 (sono due conversioni dello stesso modello: GGUF Q8_0 contro MLX bits=8).

La ricetta è riproducibile: `tools/vendor_llama_cpp.sh`. La parità a parità di modello è
verificata da `parity/run_same_model.sh` (Livello C).

Resta una cautela onesta: la soluzione richiede un binding **patchato** — la crate ufficiale,
così com'è consegnata qui, non carica ancora `spark2_5`.

Le differenze di contratto sono **solo** quelle dichiarate (identità del modello e
contatori di timing specifici del backend) — vedi `parity/README.md`.

## Licenza

Apache-2.0. Attribuzione completa in [NOTICE](NOTICE). I pesi dei modelli non sono
distribuiti qui: si scaricano dal loro publisher, sotto la loro licenza.
