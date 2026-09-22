# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Cos'è

Rizzo Flow: versione locale del pattern di **Jev** (TypeSafe, "System One": state non strutturato →
decisioni tipizzate con probabilità, zero generazione di testo), ispirata a **SemIf**
(`~/Git-projects/SemIf`, commit `ca3ba65`). Nessun training: si leggono i logit delle sole lettere
di risposta in un forward pass. Modello: **XHToken/Spark-X2.5-4B** (architettura `spark2_5`, rev.
`0bcb356…`). **Runtime predefinito dal 22 settembre 2026: llama.cpp** (release `b11081`, pacchetti
precompilati ufficiali scaricati da `rizzo download`, binding ctypes; Metal, CUDA, Vulkan per
AMD/Intel/NVIDIA, ROCm, SYCL, CPU) con i GGUF ufficiali di XHToken: `q8_0` (default), `q4_k_m`,
`bf16`. **MLX resta come secondo runtime** (`--backend mlx`, extra `mlx|cuda|cpu`; pesi originali,
BF16 di default, Q8/Q4 affini in memoria con group size 64): è quello con cui sono stati misurati
tutti i numeri fino al 21 settembre e l'unico provato sul Mac (M4 Pro 24 GiB).
`config.MODELS` elenca i checkpoint supportati: `4b` (default) e `1.7b` (XHToken/Spark-X2.5-1.7B, rev.
`14d6e83…`, stessa architettura/tokenizer/contesto 1M, ~3.4 GB). CLI: `--size 4b|1.7b` su
`download/decide/serve/evaluate`; entrambi i backend riconoscono il checkpoint da `hidden_size`
(`config.identify`) e l'ID servito diventa `rizzo-spark-x2.5-1.7b-q8_0` (con MLX: `…-q8`). Il 1.7B è stato provato solo su Windows/CUDA (sezione in fondo): funziona ma è molto meno accurato.

README, doc e messaggi all'utente sono in italiano; codice, commenti e docstring in inglese.
Repository pubblico: <https://github.com/Rizzo-AI-Academy/rizzo-flow> (remote `origin` via SSH, branch `main`;
il README pubblico è in inglese, quello italiano storico è in `docs/README.it.md`).

## Comandi

```bash
uv sync --extra test --locked                     # llama.cpp non richiede extra; MLX: --extra mlx|cuda|cpu
.venv/bin/rizzo download                          # runtime llama.cpp per questa macchina (runtimes/) + GGUF Q8_0 (~4.4 GB, models/)
.venv/bin/rizzo download --only runtime --runtime vulkan   # un'altra build; --backend mlx scarica i pesi originali (~8 GB)
.venv/bin/rizzo devices                           # device visti da llama.cpp e scelta di auto; su Windows gli eseguibili sono in .venv/Scripts/
.venv/bin/pytest -q                               # 76 test (+4 saltati), ~3 s, nessun peso richiesto
RIZZO_REAL=1 .venv/bin/pytest -q -m integration   # 4 test con runtime e GGUF reali (il più piccolo Q8_0 presente)
.venv/bin/pytest tests/test_compat.py::test_systemone_wire_shape   # test singolo
.venv/bin/ruff check src tests scripts && .venv/bin/ruff format --check src tests scripts
.venv/bin/rizzo serve                             # API + playground su 127.0.0.1:8017 (MLX: --backend mlx --bits 8)
.venv/bin/rizzo decide examples/ticket.json       # --quant q8_0|q4_k_m|bf16, --device auto|gpu|cpu|cuda|vulkan|metal|rocm|sycl
.venv/bin/rizzo evaluate benchmarks/smoke.jsonl --compare-modes --output results/local-x.json
.venv/bin/rizzo schema > request.schema.json      # rigenerare dopo modifiche a schema.py
.venv/bin/python scripts/validate_checkpoint.py --output results/local-validation
.venv/bin/python scripts/semif_compare.py --system rizzo --semif .research/SemIf --output results/local-semif
.venv/bin/python scripts/semif_report.py results/local-semif --semif .research/SemIf --against NOME=CARTELLA   # held-out + differenze appaiate
.venv/bin/python scripts/record_demo.py           # ri-registra l'oggetto DEMO di docs/index.html
```

`.claude/launch.json` definisce il server di anteprima `rizzo-q8` (porta 8017, `rizzo serve`).
Caricare il modello con llama.cpp richiede ~11 s (di cui l'hash sha256 del GGUF) e ~5.6 GiB di GPU
(Q8_0) o ~9.3 GiB (BF16): **un solo processo con pesi alla volta**, e fermare il server prima di
misurare tempi. Tutti gli script (`semif_compare`, `validate_checkpoint`, `record_demo`) accettano
`--backend llama|mlx`. SemIf è clonato in `.research/SemIf` (ignorato da git) al commit `ca3ba65`.

I test non caricano mai il checkpoint 4B: `test_service.py`/`test_compat.py` usano `FakeBackend`
+ `CharacterTokenizer` di `tests/fakes.py` (il fake favorisce sempre il secondo candidato);
`test_formats.py` ricostruisce i record dai report in `results/` e pretende lo stesso
fingerprint, lo stesso JSON e lo stesso ordine delle chiavi; `test_backend_llama.py` usa
una `FakeSession` che registra ogni chiamata (token, posizioni, sequenze, righe di logit lette);
`test_llama_release.py` copre scelta del pacchetto, download ripreso (server HTTP locale che cade a
metà), sha256 ed estrazione sicura, senza rete; `test_mlx.py` usa la vera architettura Spark ridotta
con pesi casuali (saltato senza MLX); `test_llama_real.py` è opt-in (`RIZZO_REAL=1`). La verifica sul modello reale si fa a mano (server +
curl/playground) o con `validate_checkpoint.py`.

## Architettura (flusso di una richiesta)

`schema.py` (contratto Pydantic strict, `extra=forbid`) → `prompts.compile_request` →
`backend_llama.LlamaBackend.score` (o `backend.SparkBackend.score` con MLX; scelta in
`loader.load_backend`) → `decisions.decode` → `responses.py` (la risposta è ri-validata
prima di uscire). `engine.Engine` orchestra tutto sotto un `Lock` (un solo modello residente:
richieste HTTP concorrenti sono serializzate; il parallelismo è *dentro* la richiesta).

- **Runtime llama.cpp (tre moduli, nessuna dipendenza Python oltre a `jinja2`).**
  `llama_release.py`: release pinnata (`RELEASE`/`COMMIT`), tabella `PACKAGES` (os, macchina,
  famiglia) → archivi ufficiali con sha256, `pick("auto")` = Metal su Mac ARM, CUDA se si carica
  `nvcuda.dll`/`libcuda.so.1`, altrimenti Vulkan, altrimenti CPU; `fetch` riprende i download
  interrotti (Range) e decide solo lo sha256; i tarball perdono la cartella esterna così runtime e
  librerie CUDA finiscono nella stessa directory; `locate(famiglia)` sceglie fra più build
  installate in `runtimes/` (o `RIZZO_LLAMA_DIR`). `llama_cpp.py`: struct `ModelParams`,
  `ContextParams`, `Batch` trascritti dall'header di **quel** commit (passati per valore: una build
  di un altro commit può crashare; **cambiando `RELEASE` ricontrollare `llama.h` campo per campo**),
  backend caricati con `ggml_backend_load_all_from_path`, device enumerati da ggml
  (`choose_device`: GPU discreta con più memoria, poi integrata, poi CPU; una famiglia chiesta
  esplicitamente non viene mai declassata), un solo device (`split_mode NONE`, lista `devices`),
  log di llama.cpp filtrati a warning/errori (`RIZZO_LLAMA_LOG=1` per vederli tutti).
  `backend_llama.py`: `LlamaTokenizer` rende il chat template del GGUF con jinja2 come fa
  transformers (`trim_blocks`, `lstrip_blocks`, `raise_exception`) e tokenizza con
  `llama_tokenize(parse_special=True)`: **prompt e token identici a quelli di MLX** (stessi
  `prompt_sha256` e `input_tokens`; il template dell'1.7B differisce solo per un commento).
  `score`: prefisso sulla sequenza 0, poi per ogni microbatch `seq_cp` 0→k (cache unificata: le
  celle del prefisso sono condivise, non copiate), suffissi uno dopo l'altro **senza padding** in
  una sola `llama_decode`, logit letti all'ultima posizione di ciascuno, `seq_rm`. Una domanda
  sola → un solo passaggio anche in `shared` (`shared_prefix_tokens` = 0). Contesto prenotato:
  `--ctx` + 2048 celle, `n_seq_max` = batch + 1, `n_outputs_max` = `n_seq_max` (il default
  riserverebbe `n_batch` righe di vocabolario), **`swa_full = True`**: con la cache a finestra
  compatta una cella condivisa da più sequenze non viene mai riciclata e dopo un prefisso lungo i
  rami non trovano posto (`llama_decode` = 1, verificato) → ~144 KiB/token invece di ~36.
  Niente proiezione selettiva (llama.cpp proietta tutto il vocabolario: irrilevante, poche
  posizioni). `peak_device_bytes` = calo della memoria libera della GPU da prima del load.
- **Slot a token singolo.** Ogni candidato (opzioni + speciali `__insufficient__`,
  `__below_range__`, `__above_range__`) è una lettera maiuscola A–Z. `prompts.py` verifica che
  ogni lettera sia un token singolo e che `encode(prompt + lettera) == tokens + [id]`. Da qui il
  limite `MAX_SLOTS = 26` in `schema.py` (`require_slots`): 26 opzioni/livelli con
  `allow_abstain: false`, 25 con astensione, 24/23 ancore per `numeric`. SemIf usa lo stesso
  trucco con 16 lettere. Per andare oltre servirebbero etichette a 2 token (chain rule),
  sì/no per opzione, o etichette `AA…ZZ` (497/676 sono token singoli in Spark) — non implementato,
  decisione dell'utente: restare a 26.
- **Prefisso condiviso (descrizione del backend MLX).** Messaggio user = `render_state(state)` + `render_question(...)`.
  Lo state è identico per tutte le domande → prefill una volta (blocchi da 512), confine del
  prefisso verificato token per token (ultimo token scartato per i merge BPE, mai dedotto dalla
  lunghezza). `branch_cache` clona le cache native (attenzione piena + sliding-window rotante),
  i suffissi vanno in microbatch (`--batch-size`, default 4, max 16) ordinati per lunghezza con
  padding a destra; si legge l'ultima posizione reale. Cache scartata a fine richiesta.
  `mode: "direct"` disattiva il riuso (riferimento di verifica).
- **Contesto.** `--ctx` (alias storico `--max-tokens`, default 8192) è il limite di token per domanda
  (state + domanda): con MLX guardia, non prenotazione; con llama.cpp dimensiona anche la cache
  prenotata all'avvio (~144 KiB/token, 1.4 GiB di default). Con MLX KV ≈ 36 KiB/token nel 4B (9 layer su 36 a
  attenzione piena; gli sliding sono fissi, 54 MiB), × `--batch-size` durante i microbatch perché
  `branch_cache` replica il prefisso. Lo `state` ha inoltre un tetto fisso di 256 KB in `schema.py`
  (~60k token). 1M token = ~36 GiB di sola cache. Confronto pubblicato in README e landing: Jev
  32k token per state + domanda più lunga, 64k per richiesta (<https://docs.typesafe.ai/models>,
  `jev-1.13.0`); SemIf = Qwen3.5-4B, 262.144 nativi (~1M con YaRN), default `--max-tokens 4096`.
- **Proiezione selettiva (solo MLX).** `selected_logits` moltiplica l'hidden state solo per le righe di
  vocabolario delle lettere ammesse (anche con pesi quantizzati): delta 0 rispetto al vocabolario
  pieno.
- **Decodifica pura.** `decisions.py` è deterministico e senza I/O: softmax (con temperatura),
  status `ok | insufficient_evidence | out_of_range | uncertain` da `policy`, medie pesate per
  `score`/`numeric` condizionate alle sole opzioni valide.
- **Calibrazione.** `calibration.py`: temperature scaling per tipo, legato a un `fingerprint`
  (hash di pesi, tokenizer, precisione, runtime, backend di calcolo, `PROMPT_VERSION`: una
  calibrazione MLX non vale su llama.cpp, né una CUDA su Vulkan). **Cambiare il prompt impone
  di incrementare `PROMPT_VERSION`**, il che invalida le calibrazioni esistenti (voluto).
- **Template Spark.** Il chat template antepone `you are a helpful assistant.\n\n` al system
  prompt e, con `enable_thinking=False`, il prompt termina con `<|Bot|></think>`: la lettera è il
  primo token dopo `</think>`, senza spazio (`"A"` = 46, `" A"` = 401 sono token diversi).

### Due API nello stesso server (`api.py`)

- `POST /v1/decisions` — nativa: `boolean | choice | score | numeric`, astensione, `policy`,
  logit, statistiche, hash dei prompt.
- `POST /v1/systemone`, `GET /v1/models` — **stessa interfaccia di TypeSafe/Jev**
  (<https://docs.typesafe.ai/api>), implementata in `compat.py` come traduzione verso le primitive
  native: `noul`→`boolean`, `choice` (chiavi libere → ID posizionali `o{i}` e ritorno), `score`
  (max 10 livelli). Sempre `allow_abstain: false`. `instructions`/`criteria` strutturati →
  JSON canonico. `confidence = (n·p_max − 1)/(n − 1)` (formula della demo nella doc TypeSafe; quella
  reale non è pubblica, non è calibrata). `model` accetta `rizzo-latest`, l'ID locale e qualunque
  `jev-*`; la risposta riporta **sempre** l'ID locale (`rizzo-spark-x2.5-4b-q8_0`), mai Jev.
  `usage.output_tokens` è sempre 0. Tempi/fingerprint in `x_rizzo` (estensione; gli SDK TypeSafe
  ignorano campi extra). Bearer auth solo se è impostata `RIZZO_API_KEY`. Pensato per funzionare
  con gli SDK ufficiali via `TYPESAFE_BASE_URL=http://127.0.0.1:8017` (non ancora provato con
  l'SDK reale).
- `GET /playground` (`playground.html`, pagina singola senza dipendenze esterne, servita dal
  package): builder noul/choice/score, esempi, editor JSON per entrambi gli endpoint, barre di
  probabilità, metriche (round-trip, inferenza, prefill, microbatch, token in cache), cURL.
  Bilingue IT/EN: dizionario `I18N` + `t(key, vars)` nello script, attributi `data-i18n`,
  `data-i18n-html`, `data-i18n-attr`; ogni nuova stringa UI va aggiunta in entrambe le lingue.
  Logo servito da `/playground/logo.png` (`src/rizzo_flow/logo.png`, copia ridotta di `assets/`).

- `GET /snake` (`snake.html`, stessa impostazione del playground: pagina singola, nessuna dipendenza,
  bilingue con `I18N`/`data-i18n`): Snake in cui ogni mossa è una `POST /v1/decisions` (`choice` sulle
  mosse legali, opzioni mescolate, opzionale domanda `score` "pericolo" sullo stesso state). Lo state
  è costruito in `buildRequest()` in tre viste: sensori per mossa (default; il modello gioca bene),
  sensori + griglia ASCII, sola griglia (muore subito). Il testo del prompt lato gioco è sempre in
  inglese ed è stato provato sul modello reale: numeri in README. Registratore GIF integrato
  (`startRecording`/`capture`/`gifFrame`: canvas offscreen 800×480 griglia + pannello decisioni,
  palette per frame da istogramma a 15 bit, LZW, durata reale dei frame, max 900 frame, download via
  `<a download>`). `paintBoard` è condivisa tra canvas visibile e registratore.

### Sito del progetto (GitHub Pages)

`docs/index.html` è la landing page statica, **nello stile del sito di rizzo-pii** (chiaro, centrato,
poco testo: hero con mascotte, badge, stats, card brevi, tabella di confronto, footer scuro), senza
dipendenze esterne, bilingue con `<span class="it">`/`<span class="en">` e `body.lang-it|en`.
La sezione `#demo` è un'animazione interattiva (state → riccio → bool/classe/score): usa risposte
e tempi **reali** (oggetto `DEMO` inline; ri-registrato il 22 settembre 2026 con
`scripts/record_demo.py`: 4B Q8_0 su llama.cpp/CUDA, prompt v3, RTX 5060 Ti, via
`compat.to_native`/`from_native` in-process, una domanda per richiesta, mediana di 5 chiamate a
caldo: ~44 ms a decisione; nessuna risposta scelta a mano — p.es. seniority del CV: IT "Intermedio"
0.70, EN "Senior" 0.997) e la fase "pensa" dura davvero quei millisecondi. Se cambiano modello o prompt, rigenerare quei dati. Pages serve **solo**
`docs/`: gli asset del sito stanno in `docs/assets/` (copie di `assets/`), `docs/.nojekyll` disattiva
Jekyll. URL: <https://rizzo-ai-academy.github.io/rizzo-flow/> (Settings → Pages → branch `main`,
cartella `/docs`). Anteprima locale: server `site` in `.claude/launch.json` (porta 8020). I numeri
sulla pagina vanno tenuti allineati a README e `results/`.

## Regole del progetto

- **Onestà sulle misure.** Probabilità dichiarate non calibrate; mai presentarsi come Jev; mai
  dichiarare superiorità su Jev/SemIf senza dati. Ogni limite osservato va scritto nei README.
- **Risultati create-only.** `cli.write_json` e gli script aprono i file in modalità `"x"`: non
  sovrascrivere né riscrivere report in `results/`; per nuovi esperimenti usare un percorso nuovo
  (`results/local-*` è ignorato). `results/SHA256SUMS` copre i report storici.
- **Niente troncamento silenzioso.** Input oltre i limiti → `ValueError` → HTTP 422.
- **Non ottimizzare sul test.** Le fixture proprie (`benchmarks/smoke.jsonl`) sono già state usate
  per rivedere il prompt (vedi `benchmarks/README.md`). Per il lavoro sul prompt esiste uno split
  dev/held-out (sotto): l'held-out si guarda una volta sola.

## Stato del lavoro (22 settembre 2026)

### Passaggio a llama.cpp (22 settembre 2026, su `main`)

Origine: la PR #1 di BiG86 (backend llama.cpp per AMD, ctypes su una build compilata a mano,
solo Linux). Non mergiata per scelta dell'utente; **riscritto da zero** (nessun codice della PR),
con llama.cpp come default e MLX opzionale (scelte dell'utente). Differenze di progetto rispetto
alla PR: binari precompilati ufficiali con sha256 invece della compilazione, GGUF ufficiali di
XHToken invece di una conversione di terzi, device enumerati da ggml invece di sondare `/sys`,
cache unificata (prefisso condiviso, non `ctx × (batch+1)`), Windows/macOS/Linux.

Provato **solo** su Windows 10 + RTX 5060 Ti, build CUDA 13.4 e build Vulkan sulla stessa scheda.
**Mai eseguiti: macOS/Metal (il Mac dell'utente!), Linux, AMD, Intel, ROCm, SYCL, sola CPU** —
regola dell'utente: su questo PC niente prove su CPU. Prima cosa da fare sul Mac:
`uv sync --locked && rizzo download && rizzo devices && RIZZO_REAL=1 pytest -m integration`.

Numeri (4B, prompt v3; report in `results/semif-compare/*-llama-*`, `analysis.json` accanto;
tabelle complete in `results/README.md` e `README.md`):

| | Q8_0 CUDA | BF16 CUDA | Q4_K_M CUDA | Q8_0 Vulkan | MLX-CUDA Q8 (prima) |
| --- | ---: | ---: | ---: | ---: | ---: |
| authored144 / perturbations108 | 0.812 / 0.848 | 0.829 / 0.859 | 0.769 / 0.835 | 0.807 / 0.854 | 0.829 / 0.865 |
| held-out | 0.793 / 0.861 | 0.824 / 0.875 | 0.730 / 0.801 | 0.781 / 0.861 | 0.824 / 0.875 |
| latenza p50 / p95 | 49 / 52 ms | 60 / 63 | 51 / 54 | 90 / 94 | 87 / 94 |
| shape777 shared / direct (dec/s) | 20.99 / 2.60 | 17.75 / 1.97 | 19.98 / 2.39 | 14.84 / 1.74 | 7.52 / 1.65 |
| cambi argmax shared/direct | 13/777 | 1/777 | 2/63 | 0/63 | 2/777 |
| picco GPU | 5.6 GiB | 9.3 | 3.9 | 6.0 | 6.55 |

Qualità **pari a MLX entro il rumore, non migliore**: Q8_0 vs MLX Q8 5 righe diverse su 252,
−0.017 [−0.043, 0.000]; vs SemIf Q8 −0.007 [−0.076, +0.065] (pari). Velocità 1.8× (decisione
singola) e 2.8× (shared). Con llama.cpp Q8_0 è più veloce di BF16 (con MLX-CUDA era il contrario).
Q4_K_M perde −0.043 [−0.079, −0.008]. Vulkan = CUDA nelle risposte (3 righe su 252). I 13 cambi
shared/direct a Q8_0 sono tutti quasi-pareggi (margine < 0.24) e non dipendono dal microbatch.
Debolezze invariate: 6/36 scelte sicure sbagliate su evidenza mancante, `rule_application`
perturbata 0.611 (NLL 1.69). 1.7B Q8_0: 0.678 / 0.640, 25/27 ms, 31.59 dec/s, 2.3 GiB.
Smoke proprio 0.95 (NLL 0.459), mediana 66 ms; server reale con raffiche concorrenti ok.
Snake GIF e screenshot del playground sono ancora quelli registrati con MLX (dichiarato nelle
didascalie).

### Fatto
1. API compatibile Jev + playground + test (`compat.py`, `api.py`, `playground.html`,
   `tests/test_compat.py`). Verificato con pesi reali Q8: 3 domande ≈ 540 ms; 8 noul = 1 prefill
   (126 token, 184 ms) + 2 microbatch, 932 ms totali, 0 token generati.
2. Limite portato da 23 a 26 slot. Verificato sul modello reale: 26 opzioni, slot Z scelto
   correttamente, ≈ 610 ms.
3. `scripts/semif_compare.py`: esegue le fixture di SemIf (`authored144`, `perturbations108`,
   `shape777`; hash verificati identici a quelli pubblicati) con il **loro** `evaluate.py` e lo
   stesso perimetro di tempo (modello caldo; prompt+tokenizzazione+forward+readout inclusi;
   caricamento e scrittura esclusi). `--system rizzo|semif`.
4. `prompts.py` rifattorizzato in `render_state` / `render_question` (comportamento v2 invariato,
   `PROMPT_VERSION = "spark-decisions-v2"`), così le varianti si provano per monkeypatch.

### Risultati Rizzo Q8 su fixture SemIf (prompt v2) — `results/semif-compare/rizzo-q8/`

| Misura | Rizzo Q8 (M4 Pro) | SemIf Qwen3.5-4B Q8 pubblicato (M5 Max) |
| --- | ---: | ---: |
| authored144, balanced accuracy media per famiglia | 0.758 | 0.819 |
| perturbations108 | 0.706 | 0.766 |
| Latenza per decisione, stato corto, p50 / p95 | 254 / 259 ms | non confrontabile |
| shape777 shared (37 stati ~2k token × 21 criteri) | 3.92 dec/s, 5.33 s/stato | non confrontabile |
| shape777 direct (3 stati) | 0.31 dec/s | non confrontabile |
| Cambi argmax shared vs direct (63 decisioni) | 0 (max Δp 0.057) | — |

Shared ≈ 12.6× direct. Famiglia debole: `rule_application` (0.689; 0.481 sulle perturbazioni,
NLL 1.83 = errori molto sicuri). I tempi SemIf vengono da un altro Mac: **valgono solo per la
qualità**. Le etichette SemIf non sono adjudicate da umani; 6 punti ≈ 9 righe.
Differenze volute: ogni sistema usa il proprio prompt; WANLI/TypeSafe/Every non inclusi.

Risultati storici sulle fixture proprie (smoke Q8: mediana 304 ms, 18/20, picco 4.88 GiB; stato
lungo shared 2.8× direct) e limiti osservati: `results/README.md`.

### In sospeso: miglioramento del prompt (interrotto dall'utente)

`scripts/prompt_lab.py dev|held [varianti]` — split per `group_id` alternato dentro ogni famiglia
(72 base + 54 perturbate per parte), Q8, più lo smoke proprio come controllo sui tipi nativi.
Log dei due giri su **dev**: `results/prompt-lab/`. Testo esatto di ogni prompt, tabelle complete e
conclusioni: `docs/prompt-lab.md`.

| Variante (dev) | base | perturbato | flip inversione ordine | smoke |
| --- | ---: | ---: | ---: | ---: |
| `v2-current` (system vecchio, state e domanda in JSON) | 0.754 | 0.711 | 6 | 0.90 |
| `a-text-all` (system nuovo, `<evidence>` testo, MCQ testo) | 0.827 | 0.852 | 2 | 0.95 |
| `i-systemA-json-mcq` (system nuovo, state JSON, MCQ testo) | 0.806 | 0.852 | 1 | 0.95 |
| `b` solo domanda in testo | 0.784 | 0.852 | 1 | 0.90 |
| `c` solo system nuovo | 0.811 | 0.796 | 2 | 0.95 |
| con più istruzioni (`f`, `g`, `k`: regole, astensione, nota ordine) | 0.79–0.81 | 0.80–0.82 | 2–3 | 0.90–0.95 |

Conclusioni: la domanda come scelta multipla in testo ("Question:", "A. …", "Answer with the
letter of the best option.") e un system prompt corto e orientato alla decisione aiutano entrambi
e riducono il bias di posizione; allungare il system prompt non aiuta. `a` e `i` sono pari entro
il rumore (1 riga ≈ 1.4 punti); candidata preferita `i` (NLL e stabilità migliori, state in JSON
più robusto a testo che imita i delimitatori; usa `json.dumps` senza `sort_keys` per conservare
l'ordine delle chiavi dell'utente).

**Adottata `a-text-all` (scelta dell'utente, 21 settembre 2026):** `prompts.py` spedisce ora
`SYSTEM_A` + `text_state` + `mcq`, `PROMPT_VERSION = "spark-decisions-v3"` (le calibrazioni v2 non
valgono più). `prompt_lab.py` contiene il v2 per esteso, così `v2-current` resta riproducibile.
Verificato su Windows/CUDA (sotto): test unitari, smoke 4B Q8 0.95 (NLL 0.428, identico al lab).

**Numeri v3 (21 settembre 2026, Windows/CUDA, RTX 5060 Ti):** `semif_compare.py` rilanciato con v3,
`results/semif-compare/rizzo-q8-v3-cuda/` (completo, 777 decisioni anche direct) e
`rizzo-bf16-v3-cuda/` (sola qualità). Lo script ora calcola anche la stabilità di SemIf
(`stability`, equivalente a `evaluate_perturbations.py`; verificata sulle loro predizioni
pubblicate: 0.723 e 10/9/4 flip). Q8: authored144 **0.829** (SemIf Q8 0.819), perturbations108
**0.865** (0.766); BF16 0.819 / 0.842. Held-out (guardato una sola volta, qui): 0.824 / 0.875
(SemIf Q8 sulle stesse righe 0.811 / 0.824); il dev coincide col prompt-lab (0.827 / 0.852).
Differenza appaiata su authored144 +0.010 [−0.051, +0.076] → **pari, nessuna superiorità**.
Flip reversal/wrapper/contesto 4/2/3 (SemIf 9/7/4). **Peggio di SemIf su evidenza mancante:** 6/36
scelte sicure (p ≥ 0.8) sbagliate contro 1; accuratezza 0.778 contro 0.861. `rule_application`
perturbata 0.630 (NLL 1.63). Tempi Q8: 87/94 ms p50/p95; shape777 shared 7.52 dec/s (1.76 s/stato),
direct 1.65 dec/s, shared ≈ 4.6× direct, 2 cambi argmax su 777 (max Δp 0.144), picco 6.55 GiB.
I numeri v2 sopra restano come storico (Mac). Tabelle in `results/README.md` e `README.md`.
Il processo CUDA esce con codice 9 dopo aver scritto il report (innocuo, come il 127 già noto).
**1.7B sulle stesse prove** (`rizzo-1.7b-q8-v3-cuda/`, `rizzo-1.7b-bf16-v3-cuda/`): Q8 0.700 / 0.633
(held-out 0.697 / 0.514), BF16 0.683 / 0.646; −0.128 [−0.211, −0.046] dal 4B; 17/36 flip
invertendo le opzioni; `rule_application` perturbata 0.296 (NLL 3.59); si astiene troppo
(`insufficient` 52 volte su 144, attese 36). 40/44 ms, shared 20.57 dec/s, direct 3.72, 12 cambi
argmax su 777, picco 2.82 GiB.
**BF16 completi** (`rizzo-bf16-v3-cuda-full/`, `rizzo-1.7b-bf16-v3-cuda-full/`; le cartelle senza
`-full` sono run di sola qualità, numeri identici): 4B held-out 0.806 / 0.843, 76/78 ms, shared
**15.99 dec/s** (1.31 s/stato), direct 1.97, 2 cambi argmax, picco 10.13 GiB → **su CUDA BF16 è
2.1× più veloce di Q8 in shared**; Q8 serve solo a risparmiare memoria. 1.7B BF16: shared 26.12
dec/s, direct 4.37, 22 cambi argmax su 777, picco 4.31 GiB.
Non eseguiti: WANLI/Every (download da confermare), TypeSafe (non ridistribuibile).

### Runtime MLX su Windows + CUDA (provato il 21 settembre 2026, RTX 5060 Ti 16 GB, prompt v3)

**Scelta del backend (per tutti gli utenti).** I tre runtime sono sempre MLX, cambia il pacchetto
di calcolo: extra `mlx` (Apple/Metal), `cuda` (`mlx-cuda-13`, Windows/Linux), `cpu` (`mlx-cpu`).
`cuda` e `cpu` sono dichiarati in conflitto in `[tool.uv]` (stessi file); `nvidia-nccl-cu13` è
limitato a Linux con `override-dependencies` perché uv legge i metadati del wheel Linux.
`runtime.py`: `prepare()` (chiamata da `__init__.py`; su Windows stub del modulo solo-Unix
`resource` importato da `mlx-lm` + DLL `nvidia/cu13`, `nvidia/cudnn` nel `PATH`), `resolve()` per
`--device auto|gpu|mlx|cuda|cpu` (default `auto`: GPU se l'installazione ne ha una), errori con il
comando di installazione, `rizzo devices`. `metadata` riporta `device` (gpu/cpu) e `backend`
(mlx/cuda/cpu), entrambi nel fingerprint. File letti/scritti sempre in UTF-8.
`tests/conftest.py` imposta `MLX_ENABLE_TF32=0` (solo nei test) e `tests/test_runtime.py` copre la
selezione con un MLX finto. `.claude/launch.json` usa `uv run --no-sync rizzo serve --bits 8`
(cross-platform; `--no-sync` perché un sync senza extra toglierebbe il backend).
Prove solo su GPU (scelta dell'utente: il backend `mlx-cpu` su i7-7700K impiega ~185 s per 8
token con l'1.7B Q8, inutilizzabile; l'extra `cpu` è dichiarato ma non validato sui modelli reali).
CUDA su Linux e l'extra `mlx` dopo queste modifiche **non sono stati provati** (nessun Mac qui).

- `pytest`: 41/41 (TF32 disattivato da `conftest.py`); con TF32 attivo 4 test di `test_mlx.py` falliscono
  per arrotondamento (Δ logit ≤ 8e-4 contro tolleranze 1e-4/2e-4 tarate su Metal).
- 4B, `examples/ticket.json` (4 domande) a caldo, shared/direct: BF16 223/386 ms (picco 10.5 GiB),
  Q8 ~310/425 ms (6.9 GiB), Q4 305/403 ms (4.9 GiB); caricamento ~21 s. Risposte tutte corrette.
  Smoke Q8: 0.95 (NLL 0.428), mediana 103 ms; perturbazioni 9/9. Con prompt v2 lo smoke era 0.90.
- 1.7B: BF16 104/186 ms, Q8 ~140/205 ms, Q4 137/203 ms; caricamento ~9.5 s. **Qualità scarsa:**
  smoke Q8 0.45 (v2: 0.35), perturbazioni 5/9. Con `allow_abstain: true` sceglie quasi sempre
  "cannot determine" (sul ticket: D 0.76; senza astensione A 0.9999) → usarlo con
  `allow_abstain: false` o via `/v1/systemone`; anche lì sbaglia (`team=billing` 0.997 sul ticket).
- Server uvicorn reale: `/v1/systemone`, `/v1/decisions`, 422 sugli input invalidi, regge pause
  di 15 s. La prima richiesta dopo l'avvio paga la compilazione JIT dei kernel (fino a ~48 s la
  prima volta in assoluto, poi in cache su disco).
- Limite MLX-CUDA/Windows: il processo va in abort (exit `3221226505`, nessun traceback) quando
  termina un thread che ha eseguito inferenza. Colpiva anche `rizzo serve`: con richieste
  concorrenti (demo Snake, playground) uvicorn usa più thread di lavoro e ne elimina gli inattivi →
  crash dopo pochi minuti. **Fix:** `Engine` esegue `backend.score` su un solo thread dedicato
  (`ThreadPoolExecutor(max_workers=1)`) che vive quanto il processo; verificato con raffiche di 6
  richieste + pause di 12 s (prima moriva al primo giro). Resta l'uscita con codice 127 alla
  chiusura di ogni processo CUDA (innocua) — e chi chiama `backend.score` direttamente da thread
  propri di breve vita, fuori da `Engine`, resta esposto.

### Da fare
- **Provare llama.cpp sul Mac (Metal)** e, appena possibile, su Linux e su una GPU AMD o Intel
  vera; rispondere a BiG86 sulla PR #1 (la sua RX 7900 XTX sarebbe la prima prova AMD).
- Ri-registrare Snake GIF e screenshot del playground con llama.cpp.
- Cache a finestra compatta con rami condivisi (oggi `swa_full`): servirebbe potare la sequenza 0
  solo nella cache SWA, cosa che l'API pubblica non permette.
- Lato SemIf del confronto sullo stesso Mac: serve scaricare `Qwen/Qwen3.5-4B` (~9 GB, rev.
  `851bf6e…`) e un venv separato con `mlx==0.32.2` + `mlx-lm` al commit `a63e24c` (diverso da
  quello di questo progetto). **Chiedere conferma all'utente prima di scaricare.** Poi
  `semif_compare.py --system semif --bits 8` con `PYTHONPATH`/venv di SemIf.
- Probabilità molto concentrate (spesso 0.9999) → la `confidence` compatibile Jev è quasi sempre
  ≈ 1: serve temperature scaling (`rizzo calibrate`) su dati del dominio prima di usare soglie.
- Bias di posizione residuo: debiasing per permutazione (un microbatch in più) non implementato.
- Il modello sceglie poco astensione/fuori scala (casi documentati in `results/README.md`).
