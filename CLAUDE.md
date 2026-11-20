# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Cos'è

Rizzo Flow: versione locale del pattern di **Jev** (TypeSafe, "System One": state non strutturato →
decisioni tipizzate con probabilità, zero generazione di testo), ispirata a **SemIf**
(`~/Git-projects/SemIf`, commit `ca3ba65`). Nessun training: si leggono i logit delle sole lettere
di risposta in un forward pass. Modello: **XHToken/Spark-X2.5-4B** (architettura `spark2_5`, rev.
`0bcb356…`). **Runtime predefinito dal 22 settembre 2026: llama.cpp** (release `b11081`, pacchetti
precompilati ufficiali scaricati da `rizzo download`, binding ctypes; Metal, CUDA, Vulkan per
AMD/Intel/NVIDIA, ROCm, SYCL, CPU). **Pesi predefiniti dal 25 settembre 2026: il nostro
fine-tuning LoRA** (`--weights flow`, repo pubblici `rizzoaiacademy/rizzo-flow` rev. `55633c8c…` e
`rizzoaiacademy/rizzo-flow-1.7b`: `q8_0`, `q4_k_m` (nostra, `llama-quantize` b11081 senza
imatrix), `bf16`); `--weights base` = GGUF
ufficiali di XHToken: `q8_0`, `q4_k_m`, `bf16`. `config.GGUF` è indicizzato da (size, quant,
variant); `config.gguf_spec` dà errore per combinazioni inesistenti (niente ripieghi silenziosi). **MLX resta come secondo runtime** (`--backend mlx`, extra `mlx|cuda|cpu`; pesi originali,
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
.venv/bin/rizzo download                          # runtime llama.cpp per questa macchina (runtimes/) + GGUF fine-tuned Q8_0 (~4.4 GB, models/rizzo-flow/); --weights base = XHToken
.venv/bin/rizzo download --only runtime --runtime vulkan   # un'altra build; --backend mlx scarica i pesi originali (~8 GB)
.venv/bin/rizzo devices                           # device visti da llama.cpp e scelta di auto; su Windows gli eseguibili sono in .venv/Scripts/
.venv/bin/pytest -q                               # 82 test (+6 saltati), nessun peso richiesto
RIZZO_REAL=1 .venv/bin/pytest -q -m integration   # 6 test con runtime e GGUF reali (il più piccolo Q8_0 presente)
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
+ `CharacterTokenizer` (il fake favorisce sempre il secondo candidato); `test_backend_llama.py` usa
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

- `GET /snake`: demo Snake in cui ogni mossa è una `POST /v1/decisions`. Dettagli (viste dello
  state, registratore GIF) in `notes/stato-lavoro.md`.

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
  dev/held-out (`notes/stato-lavoro.md`): l'held-out si guarda una volta sola.

## Stato del lavoro

Storico dettagliato (fine-tuning LoRA come default dal 25/9, passaggio a llama.cpp, numeri su
CUDA/Vulkan/MLX, prompt lab v2→v3, runtime MLX su Windows, PR aperte, lista "Da fare") in
**`notes/stato-lavoro.md`**: leggerlo prima di lavorare su prestazioni, prompt, pesi, backend o
confronti con SemIf.
