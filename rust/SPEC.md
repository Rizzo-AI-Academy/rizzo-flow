# SPEC — **rizzo-flow-rs** (nome di lavoro): port Rust di Rizzo Flow — fork candidato per Rizzo AI Academy

> Ispirato al pattern di **Jev** (TypeSafe, "System One") e alla sua implementazione open
> **Rizzo Flow** (Rizzo AI Academy, Apache-2.0) — di cui portiamo *l'idea e l'interfaccia*,
> non il codice. **Riscrittura da zero in Rust**, multi-modello (GGUF), 100% locale.
> **Proposta**: questo port sarà proposto a **Simone Rizzo** come alternativa Rust a Python,
> da valutare come **fork** del suo progetto (tramite Alfonso) → vedi §1-bis.
> Data: 21/09/2026 · Autrice: Sempre (per Alfonso) · Aggiornato: **22/09/2026**
> Stato: **F1+F2+F2.1+F3 in gran parte portati.** Core (`schema`+`decisions`, 10 test
> originali 10/10) · backend llama.cpp/CUDA verde · **serve** (`/health`,
> `/v1/decisions`, `/v1/models`, `/v1/systemone` + auth bearer) · **CLI drop-in**
> (`devices` `schema` `download` `calibrate` `decide` `evaluate` `serve`) ·
> **41 test verdi** (locali e sul nodo con CUDA) · parità numerica del fit di
> calibrazione verificata contro il Python.

## 1 · Cos'è, e perché

Un **server di decisioni tipizzate** in Rust: riceve uno *state* (testo/JSON) e domande
tipizzate, e restituisce **decisioni con probabilità** — `boolean` · `choice` · `score` ·
`numeric` — **senza generare un singolo token**. Zero testo, zero parsing, zero allucinazioni
possibili per costruzione: si leggono i **logit delle lettere di risposta** (A, B, C…) in un
forward pass.

**Perché lo facciamo noi (in Rust):**
1. **Rust-first** (regola di casa): il layer decisionale è logica pura — perfetto per Rust.
2. **Multi-modello**: agganciato a **llama.cpp/GGUF**, funziona con *qualunque* modello —
   **inclusi i nostri** (lfm-faro, candela v12/v13). Non si è vincolati a Spark.
3. **Performance**: llama.cpp ha kernel CUDA maturi per la Turing del nodo → obiettivo
   **<1s a decisione calda** (contro i ~27s di MLX-CUDA acerbo misurati oggi).
4. **Indipendenza**: niente API esterne, niente rate limit, dati che non escono.
5. **Piattaforma per il riadattamento "a nostro piacere"**: calibrazione + fine-tuning di un
   nostro modello sui nostri task (tipizzazione, confini, triage) — convergenza con candela.

**Riferimenti misurati (21-22/09/2026, nodo2070 — RTX 2070 8 GB):**
- Rizzo Flow (Spark-X2.5-4B, MLX-CUDA, Q8): nostro test 6 domande → **5/6** (1 astensione
  onesta), ~27s; su M4 Pro ~0,25s/decisione.
- JEV 1.13 (hosted, gratis per ora): **6/6** in 0,72s; API chiusa.
- Rizzo Flow **non era mai stato acceso su Linux/CUDA**: prima accensione = nostra (21/09).
- **rizzo-flow-rs (questo port)** con `Qwen3.5-4B-Instruct-SingleTurn` Q4_K_M (2,7 GB):
  test 6 domande → **5/6** con la stessa divergenza di Spark su s3 (vedi caveat sotto),
  `status_accuracy` 0,833, **0,27 s/decisione calda** (mediana, richieste mono-domanda),
  **3,7 decisioni/s**, carico modello ~1,3 s. Confronto: il probe `Qwen2.5-7B-Instruct` q4
  ne fa 4/6 (sbaglia s4). Dettaglio in `KB-20260922-01`.
  - **Caveat onesto**: su s3 («Eliminando il canone, circa 1,8 miliardi passerebbero al
    bilancio dello Stato») Spark risponde `__insufficient__`, il nostro modello
    `DERIVAZIONE` con top 0,94 — è una *convergenza di giudizio* con l'altro modello
    generale provato, non un errore del port: la frase è una stima condizionale.

## 1-bis · Il fork e la proposta a Simone Rizzo (vincolo di Alfonso, 21/09)

Il port è pensato **come fork di Rizzo Flow** e sarà **proposto a Simone Rizzo** (tramite Alfonso)
come **alternativa Rust al runtime Python**. Impegni che ne derivano:

1. **Parità di contratto** — stessi schemi JSON di rizzo-flow (`request.schema.json` /
   `response.schema.json`), stessi endpoint (`/v1/decisions`, `/v1/systemone`, `/v1/models`,
   `/health`), stessi nomi di campo e semantica (status, policy, astensione, incertezza,
   statistiche). Un client scritto per rizzo-flow **funziona senza modifiche**.
2. **CLI drop-in** — binario `rizzo` con gli stessi sottocomandi (`devices`, `download`,
   `decide`, `serve`, `schema`, `evaluate`); opzioni compatibili dove ha senso
   (`--bits`, `--device`, `--ctx`, `--batch-size`).
3. **Comportamento portato fedelmente** — `decisions.py`/`prompts.py`/`compat.py` riscritti
   **con i loro test** + una **suite di parità** (§9). Numeri verificati: la suite originale
   ha **26 test** in 5 file — `test_decisions` 10, `test_compat` 5, `test_service` 5,
   `test_mlx` 3, `test_runtime` 3. Di questi **20 stanno in moduli portabili** (decisioni,
   compat, servizio) e **6 sono specifici di MLX/runtime** (caricamento del backend MLX e
   risoluzione dei device), non portabili per costruzione: è la differenza di backend
   dichiarata al punto 4. Questo port copre le aree portabili con **41 test** (10 portati
   1:1 da `test_decisions` + 6 compat + 7 servizio + 18 unitari su calibrazione, compat,
   valutazione, I/O e checkpoint pinnati).
4. **Differenza dichiarata** — il backend: **llama.cpp/GGUF** al posto di MLX → *multi-modello*
   (qualsiasi GGUF), binario singolo, kernel CUDA maturi. Spark-X2.5 entra quando avrà un GGUF
   (oggi architettura custom, solo MLX).
5. **Etichetta di fork e crediti** — licenza **Apache-2.0** (la stessa), `NOTICE` con attribuzione
   completa (Rizzo Flow © Rizzo AI Academy; Jev © TypeSafe come ispirazione a monte), README che
   dichiara cosa è portato, cosa manca, e i numeri misurati. **Mai presentarsi come progetto
   ufficiale** di Rizzo AI Academy: è un fork *proposto* — la scelta è loro.
6. **La proposta** — quando F1/F2 sono verdi: fork pubblico (o PR) + una nota breve per Simone:
   cosa c'è, come si accende, i numeri, cosa manca. Decisione finale a lui.

## 2 · Architettura

```
POST /v1/decisions ──► schema (serde, extra=forbid) ──► prompts::compile
   (o /v1/systemone)      validazione + errori 422       │
                                                         ▼
   risposta tipizzata ◄── decisions::decode ◄── backend::score (llama.cpp)
   (serde, ri-validata)     softmax/tipi/astensione    prefill una volta (KV)
                                                        + suffix per domanda (KV reuse)
                                                        + logit delle SOLE lettere
```

- **Crate unico** `decisore` (bin + lib), moduli:
  `schema` · `prompts` · `engine` · `backend` · `decisions` · `responses` · `compat` · `api` · `cli` · `calibration`.
- **Backend**: `llama-cpp-2` (binding Rust di llama.cpp) — GGUF, CUDA sul nodo.
  MVP: contesto singolo, prefisso condiviso in cache; per ogni domanda: truncate KV al
  prefisso → decode del suffisso → logit all'ultima posizione → selezione delle righe-lettera.
- **Un modello residente**, `Engine` serializzato (come rizzo-flow); parallelismo *dentro* la richiesta.
- **Modello di partenza per i test**: i nostri GGUF (lfm-faro / candela v13) + Spark se/quando
  avrà GGUF (oggi: architettura custom, solo MLX).

## 3 · API (doppia, come rizzo-flow)

- **Nativa** `POST /v1/decisions`: `boolean | choice | score | numeric`;
  `policy` = `{allow_abstain, min_top_probability, max_unavailable_probability}`;
  speciali `__insufficient__` · `__below_range__` · `__above_range__`;
  `status` = `ok | insufficient_evidence | out_of_range | uncertain`;
  incertezza = `{top_probability, entropy_nats, concentration, unavailable_probability}`;
  `temperature`; logit per opzione; legenda; statistiche (mean/std/median/p10/p90) per score/numeric.
- **Compat Jev** `POST /v1/systemone` + `GET /v1/models`: `noul`→boolean, `choice`, `score`
  (max 10 livelli), `confidence = (n·p_max − 1)/(n − 1)`, `usage.output_tokens = 0`,
  accetta `*-latest` e id `jev-*` ma risponde **sempre col nostro id**.
- `GET /health` (stato + fingerprint), `GET /playground` (pagina minima), auth bearer opzionale.

## 4 · Cuore algoritmico (porting fedele di `decisions.py` / `prompts.py`)

- **Puro e deterministico, senza I/O**: `candidates()` · `softmax(temperature)` · `summarize()` ·
  `decode()` — portati **con i loro test**: i **10 test di `test_decisions` portati 1:1**
  (`tests/ported_decisions.rs`), più la verifica di parità esatta su 20 casi golden
  (`parity/core_vectors.json`) contro il riferimento Python.
- **Lettere single-token**: ogni candidato = una lettera A–Z, verificata col tokenizer GGUF
  (`encode(lettera) == 1 token` e `encode(prompt+lettera) == tokens+[id]`). Limite **26**
  (estensione a etichette multi-token in F3, via chain rule).
- **Prefisso condiviso**: lo state è identico per tutte le domande → prefill una volta; il
  confine del prefisso si trova **token per token, mai per lunghezza**.
- **Prompt**: base = variante `a-text-all` (system corto decisionale + `<evidence>` + MCQ testo +
  "Answer with the letter…"); `PROMPT_VERSION` dentro il fingerprint (cambiare prompt invalida
  le calibrazioni — voluto).

## 5 · Performance (obiettivi onesti)

| | target F1 | stretch F2 | **misurato 22/09** (2070, Qwen3.5-4B Q4_K_M) |
|---|---|---|---|
| decisione calda, stato corto (shared) | **< 1 s** (2070, Q4/Q8) | < 300 ms | **0,268 s** mediana (mono-domanda) · 0,28 s p95 |
| prefill | una volta per richiesta | idem + chunking | 0,66 s (6 domande, prefisso 242 token) |
| VRAM | ≤ 6 GB (Q4/Q8, 8 GB totali) | BF16 dove c'è spazio | ~3,4 GB residenti (pesi 2,6 GB + KV) |
| multi-domanda (8×) | < 4 s | < 1,5 s | 6 domande in ~1,1-1,4 s |

**Nota sul `mode`** (misurato 22/09): su richieste **mono-domanda** `direct` è *più veloce*
di `shared` (0,199 s vs 0,268 s mediana) — non c'è prefisso da condividere e il percorso
`shared` paga due decode (prefisso + suffisso) invece di una. `shared` conviene quando la
stessa richiesta porta **più domande** (prefill una volta, N suffissi). Il report
`evaluate --compare-modes` misura il delta: max 0,0088 sulle probabilità, 0 argmax cambiati.

Regola di casa: ogni numero pubblicato con macchina, precisione e caveat.

## 6 · Calibrazione e riadattamento ("massima resa a nostro piacere")

- **F2 — calibrazione**: temperature scaling per tipo, legato al fingerprint
  (porting di `calibration.py`); le probabilità di Spark sono spesso 0.9999 → senza calibrazione
  le soglie non si usano. Sul **nostro** modello, calibriamo sui **nostri** dati.
- **F4 — riadattamento**: addestrare/calibrare un **nostro** modello (catena QLoRA candela) sul
  formato decisionale: tracce MCQ single-letter generate dai nostri container + task di casa
  (tipizzazione `[NORMA]/[FATTO]/[DERIVAZIONE]/[APERTO]`, confini, triage). Obiettivo: **un
  decisore italiano, calibrato, sui nostri task, 100% casa** — e il confronto onesto
  JEV vs Rizzo-flow vs Decisore sugli stessi set.

## 7 · Fasi

- **F1 (MVP)** — ✅ `schema`+`prompts`+`decisions`+`backend`(llama.cpp)+CLI `decide`.
  Test: le 6 domande canone + confronto coi 5/6 di rizzo-flow; tempi misurati sul nodo.
- **F2** — ✅ server (`tiny_http`, non axum: dipendenza minima) con `/health`,
  `/v1/decisions`, `/v1/systemone`, `/v1/models` + auth bearer opzionale ·
  ✅ calibrazione (`fit_temperature` + `calibrate`) · ✅ multi-modello (qualunque GGUF) ·
  ⬜ playground minimo (oggi: 501 dichiarato) — la UI dev del Python non è portata.
- **F3** — parziale: ✅ `mode shared/direct` come strategia reale · ✅ `evaluate`
  riproducibile (coverage, NLL/Brier/ECE, confronto modalità) · ✅ `schema`/`download` ·
  ⬜ batching multi-riga, etichette >26 (chain rule), `/metrics`, confini d'uso.
- **F4** — modello nostro addestrato/calibrato + confronto pubblico.
- **F5 (proposta)** — fork pubblico + nota per Simone (tramite Alfonso); decisione loro.

## 8 · Cosa NON facciamo

- **Niente copia-incolla**: riscrittura da zero (Apache-2.0 compatibile; crediti completi —
  Jev di TypeSafe, Rizzo Flow di Rizzo AI Academy).
- **Mai presentarsi come progetto ufficiale** di Rizzo AI Academy: è un fork *proposto*,
  la scelta è loro.
- **Niente modifiche al contratto** senza mantenerne la parità con rizzo-flow: la compatibilità
  è il valore della proposta.
- **Niente affermazioni senza dati**: probabilità non calibrate dichiarate tali; nessuna
  "superiorità" senza confronto appaiato.
- **Niente MLX**: il nostro backend è llama.cpp/GGUF (più portabile e ottimizzato per il nodo).
- **Niente nomi/ID di Jev nelle risposte**: sempre il nostro id locale.

## 9 · Criteri di accettazione F1

- [x] `rizzo-flow-rs decide rizzo_test_6.json` → risposte equivalenti o migliori di rizzo-flow (5/6), **in ~1,4 s a freddo**
- [x] `cargo test` verde (porting dei test chiave di `decisions`/`schema`) — **41 test**
- [x] Build con CUDA sul nodo; funziona con **più GGUF** (LFM2.5 ibrido, Qwen2.5, Qwen3.5)
- [x] Prefisso in cache: 6 domande < 1,5 s; nessuna regressione sui tipi (score/numeric inclusi)
- **Parità di contratto**: stesse richieste JSON → stessi campi e stessa semantica
      (differenze dichiarate: `timing`/`model` riportano il backend llama.cpp; schemi
      vendorizzati dal progetto originale). Parità numerica verificata su `canonical`+`sha256`
      e sul fit di calibrazione (temperature e NLL identici al Python).
- [x] **Suite di parità automatica** (`parity/run_parity.sh`): Livello A esatto sul core
      (20 casi, 515 campi, delta max 4,4e-16, 0 differenze) + Livello B di forma end-to-end
      (0 differenze strutturali bloccanti, 19 dichiarate). Report grezzi versionati in
      `parity/results/`. **Verbale completo: [REPORT.md](REPORT.md)**.

## 10 · Cosa resta prima della proposta a Simone

- [x] **Suite di parità automatica** Python↔Rust — `parity/run_parity.sh`, due livelli,
      report versionati.
- [x] **NOTICE + README** con numeri misurati e cosa è portato / cosa manca.
- [x] **Report di verifica strutturato** — [REPORT.md](REPORT.md): metodo, evidenze,
      differenze dichiarate, limiti, riproducibilità.
- [ ] Dataset etichettato **di dominio** per la calibrazione: il meccanismo e il percorso
      end-to-end sono verificati (`calibration/pilot/`), ma il pilota a 24 righe produce un
      fit degenere (T al bordo, overfitting) → nessuna temperatura usabile in produzione.
- [ ] Parità numerica del report di `evaluate` col Python (come fatto per il fit).
- [ ] Playground/snake/logo (UI dev): decidere se portarli o lasciare 501.
- [ ] Etichette multi-token (>26 opzioni) via chain rule; `/metrics`.
- [ ] Confermare la forma della proposta (fork pubblico o pull request) — decisione di
      Rizzo AI Academy.
