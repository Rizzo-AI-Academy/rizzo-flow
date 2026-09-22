# Report di verifica — `rizzo-flow-rs`

**Oggetto**: port in Rust di Rizzo Flow (backend llama.cpp/GGUF), proposto come fork.
**Data**: 22/09/2026 · **Autrice**: Sempre (per Alfonso Riva) · **Destinatario**: Simone Rizzo (via Alfonso)
**Riferimenti**: [SPEC.md](SPEC.md) (contratto e vincoli) · [parity/README.md](parity/README.md)
(suite di parità e report grezzi in `parity/results/`)

Questo report non è una nota di presentazione: è il **verbale di verifica**. Ogni
affermazione ha sotto un comando o un file che la produce, e ogni limite è dichiarato.

---

## 1 · Sommario

| consegna | stato | evidenza |
|---|---|---|
| Core decisionale (`schema`, `prompts`, `decisions`) | ✅ | 10 test originali portati; **parità esatta su 20 casi golden** |
| Backend llama.cpp/GGUF su CUDA | ✅ | build e inferenza sul nodo; **3 famiglie di modelli** provate (2 ibride) |
| CLI drop-in | ✅ | `devices` `schema` `download` `calibrate` `decide` `evaluate` `serve` |
| Server HTTP | ✅ | `/health`, `/v1/decisions`, `/v1/systemone`, `/v1/models`, auth bearer |
| Contratto TypeSafe (`compat`) | ✅ | `noul`/`choice`/`score`, `usage`, `x_rizzo`; test dedicati |
| Calibrazione | ✅ | `fit_temperature` + `calibrate`; **parità numerica col Python** e **percorso end-to-end verificato** (pilota, §3.10) |
| Benchmark riproducibile | ✅ | `evaluate`: coverage, NLL/Brier/ECE, latenze, confronto modalità |
| Suite di parità | ✅ | due livelli, eseguibile con un comando |
| Test automatici | ✅ | **44 verdi** (locali e sul nodo con CUDA) |
| Documentazione | ✅ | SPEC, README, NOTICE, `parity/README.md`, questo report |
| Playground/snake/logo (UI dev) | ⬜ | non portati: rispondono `501` (dichiarato, non silenzioso) |
| Etichette multi-token (>26 opzioni), `/metrics` | ⬜ | non ancora |

---

## 2 · Metodo di verifica

Il problema centrale di un port è che "sembra funzionare" non è una prova. Ho separato
quattro classi di verifica, in ordine di forza probatoria.

### A · Parità esatta sul core decisionale (senza modello)

La logica decisionale non dipende dal modello: dato lo stesso vettore di logit e la stessa
domanda, la risposta è determinata. Quindi ho tolto il modello dall'equazione: **20 casi
golden** (`parity/core_vectors.json`) passati sia al riferimento Python
(`rizzo_flow.decisions.decode`) sia al port Rust, confrontati campo per campo con
tolleranza numerica 1e-9.

Copertura: `boolean` / `choice` / `score` / `numeric`; astensione vincente; fuori range
(`__below_range__` / `__above_range__`); `allow_abstain`; `min_top_probability`;
`max_unavailable_probability`; parità di logit (vince il primo massimo); temperature
0,5 / 1,0 / 2,0.

> Qualsiasi differenza qui è un **bug del port**, non un effetto del modello.

### B · Parità di forma end-to-end (contratto)

Il confronto dei *valori* tra i due runtime **non è apples-to-apples**: il riferimento
gira su Spark-X2.5/MLX, il port su un GGUF/llama.cpp — sono modelli diversi. Confrontare
le risposte sarebbe disonesto. Ho quindi confrontato la **struttura**: stesse chiavi,
stessi tipi, stessa nidificazione, sullo stesso file di richiesta
(`rizzo_test_6.json`). Le differenze di valore sono raccolte come *informative*; le
differenze strutturali sono bloccanti, **salvo quelle dichiarate** (§4).

### C · Verifiche indipendenti delle funzioni pure

Confronto diretto col Python, su input identici, delle funzioni che il contratto usa:

- `canonical` + `sha256` (usati nel fingerprint e nel digest dei dataset) — stessi digest;
- `fit_temperature` — stesse temperature, stesso `dataset_sha256`, stessi NLL, su 3 dataset;
- `confidence` (formula peak-over-uniform del contratto TypeSafe);
- derivazione del nome modello servito (`model_name`).

### D · Verifiche empiriche sul ferro

Non basta che il codice sia corretto: deve funzionare sulla macchina.
End-to-end HTTP con `curl` (codici 200/401/404/422/501), determinismo (due run identiche),
tempi e VRAM misurati, comportamento su modelli **ibridi** (recurrent state).

---

## 3 · Evidenze

### 3.1 Parità esatta sul core — `parity/results/report_core.json`

```
casi: 23        identici: 23        con differenze: 0
campi confrontati: 350 numerici + 168 scalari = 518
delta numerico massimo: 4.44e-16     (ordine di somma in doppia precisione)
differenze bloccanti: 0
```

### 3.2 Parità di forma end-to-end — `parity/results/report_shape.json`

```
sezioni confrontate: model, mode, answers, calibration, timing
campi confrontati: 100 numerici + 64 scalari
differenze strutturali bloccanti: 0
differenze dichiarate: 19            (identità backend + contatori MLX, §4)
differenze informative: 100          (valori: modelli diversi per progetto)
sezioni identiche senza alcuna differenza: mode, calibration
```

### 3.3 Funzioni pure contro il riferimento Python

```
canonical + sha256      digest identici (stringa e oggetto annidato)
fit_temperature         3 dataset: temperature identiche, dataset_sha256 identico,
                        NLL identici fino alla 15ª cifra decimale
```

### 3.4 Test automatici

```
44 test verdi (21 unit + 6 compat + 10 decisions + 7 service)
eseguiti sia in locale sia sul nodo con CUDA
```

### 3.5 End-to-end HTTP (nodo2070:8017)

```
GET  /health         200  {"status":"ready","model":{...}}
POST /v1/decisions   200  risposta completa; 422 su JSON invalido e su tipo ignoto
GET  /v1/models      401 senza chiave / 401 con chiave errata / 200 con chiave (3 voci)
POST /v1/systemone   401 senza chiave / 200 con chiave (noul + choice + score)
GET  /playground     501  (UI dev non portata, dichiarato)
GET  /nope           404
```

### 3.6 Prestazioni misurate (RTX 2070 8 GB, `Qwen3.5-4B-Instruct-SingleTurn` Q4_K_M, `--ctx 4096`)

> Le misure della tabella sono state prese a **`--ctx 4096`** (il default è poi diventato
> 8192, allineato al riferimento): senza questo dato la tabella non è riproducibile.

| | valore |
|---|---|
| decisione calda (mono-domanda) | **0,268 s** mediana · 0,28 s p95 |
| throughput | 3,7 decisioni/s |
| carico modello | ~1,3 s |
| prefill (6 domande, prefisso 242 token) | 0,66 s |
| VRAM residente | ~3,4 GB |
| test canone 6 domande | 5/6, ~1,1-1,4 s totali |

Riferimento sullo stesso test: Rizzo Flow (Spark-X2.5-4B, MLX-CUDA, Q8) → 5/6 in
**~26-27 s** di inferenza (misurato durante la suite, `parity/`).

### 3.7 Il port legge il contenuto (non la posizione)

Sul banco a 6 domande i quattro quesiti `choice` hanno prodotto **quattro indici vincenti
diversi** (0, 1, 2, 3): la risposta dipende dal contenuto, non dalla posizione della
lettera. È la controprova che il difetto osservato sui modelli fine-tuned di casa
(LFM2.5, che rispondono sempre alla stessa posizione) è dei modelli, non del port.

### 3.8 Allineamento al progetto corrente (verifica del 22/09/2026)

Prima di consegnare ho confrontato il port con lo stato **corrente** del progetto originale
(5 commit avanti rispetto alla copia su cui avevo lavorato). Esito:

- **Il contratto è intatto.** `decisions.py`, `schema.py`, `prompts.py`, `compat.py`,
  `calibration.py`, `evaluation.py`, `responses.py`, `api.py`, `engine.py` **non sono
  cambiati**: il port è ancora sul contratto corrente, e la parità esatta del core resta
  valida.
- **È cambiato il backend.** Il progetto ha aggiunto un backend **llama.cpp in Python**
  (`backend_llama.py`, `llama_cpp.py`, `llama_release.py`, `loader.py`), con runtime
  prebuilt scaricato via `rizzo download` e pilotato via `ctypes`, più tre file di test.
  **Conseguenza sulla premessa**: non è più «llama.cpp invece di MLX» — l'originale l'ha
  già fatto (dopo la PR #1 di un contributore, che il README cita come il motivo per cui
  llama.cpp è diventato il runtime predefinito). La differenza di questo port è quindi
  **Rust invece di Python**: binario singolo, nessuno stack Python, nessuna gestione di
  binari prebuilt — e l'apparato di verifica.
- **Sono cambiate le opzioni CLI.** Il riferimento accetta ora `--backend`, `--quant`,
  `--threads`, e su `download` `--accelerator` e `--only`. Il port le accetta tutte:
  `--quant` seleziona il file pinnato, `--backend` accetta `llama` e **rifiuta `mlx`** con
  spiegazione, `--threads` è **reale** (passa a llama.cpp), `--accelerator`/`--only`
  riconoscono che qui il runtime è compilato nel binario.
- **Il modello del progetto ha ora un GGUF.** `XHToken/Spark-X2.5-{4B,1.7B}-GGUF`, pinnato
  per revisione **e sha256**. Il port è stato ripuntato su quella tabella: il file scaricato
  durante la verifica (`Spark-X2.5-4B-Q8_0.gguf`) ha **sha256 corrispondente al pin**.
- **Il maintainer chiede esplicitamente contributi di questa forma**: il README invita a
  eseguire il progetto «su hardware che non abbiamo: ... Linux ...» e a riportare
  `rizzo devices` e i tempi. La PR #1 è il precedente: una run su GPU AMD via llama.cpp,
  «with a pinned and verified toolchain and honest measurements», che ha convinto il
  progetto a cambiare runtime predefinito.

**Il limite che ne resta** è quello di §5.1: il port non carica ancora `spark2_5` (dipende
dalla versione di llama.cpp incorporata nella crate Rust), quindi **non può eseguire il
modello del progetto** finché la dipendenza non avanza.

### 3.9 Audit interno (22/09/2026) — difetti trovati e risolti

Prima di considerare chiuso il lavoro ho fatto un audit a freddo, verificando ogni
affermazione con una nuova esecuzione invece che con la memoria. Ha prodotto quattro
correzioni, tutte riverificate:

| # | difetto trovato | gravità | stato |
|---|---|---|---|
| 1 | **`n_batch` dimensionato sul prefisso**: una domanda con suffisso più lungo del batch faceva **abortire il processo** (`GGML_ASSERT(n_tokens_all <= cparams.n_batch)`, exit 134) — in `mode: shared`, che è il **default**; su `serve` una sola richiesta lunga abbatteva il servizio. Lo stesso input in `direct` funzionava, il che ha isolato la causa. | alta | risolto: `n_batch = max(prefisso, suffisso più lungo).max(512)`; riverificato (suffisso 1412 token: exit 0, risposta corretta, 0,51 s) |
| 2 | **Checkpoint predefinito inutilizzabile con errore criptico**: il default pinnato (Spark-X2.5) non è caricabile e l'errore era `null result from llama cpp`, senza indizi. | media | risolto: l'architettura viene letta **dal file GGUF** (`src/gguf.rs`, fail-open) e l'errore ora spiega il limite noto e indica `--model` |
| 3 | **Percorsi di errore non coperti**: 0 casi su 20 nei vettori golden producevano un errore, benché il comparatore li confronti. | media | risolto: 3 casi aggiunti (logits disallineati, temperatura 0, temperatura negativa) → **23/23 identici**, messaggi d'errore compresi |
| 4 | **«`--threads` è reale» non dimostrato**: il valore raggiunge llama.cpp, ma la misura era inconcludente (0,33 vs 0,30 s). Il test che avrebbe dato segnale era **bloccato dal difetto 1**. | bassa | rimisurato dopo il fix 1: input lungo su CPU, **0,54 s (1 thread) vs 0,50 s (4 thread)** — effetto reale ma piccolo su un modello da 0,5 B: l'affermazione ora dice questo, non di più |

Due inesattezze nei documenti sono state corrette: il conteggio dei test (**41 → 44**,
con il dettaglio aggiornato) e il **`--ctx`** con cui erano state prese le misure (§3.6).

Una nota di metodo: la prima verifica del fix 1 sembrava fallire perché lo script di
parità compila l'*esempio* con le feature di default, che **non compila affatto il
backend** (è dietro `feature = "llama"`): stavo misurando un binario stantio. Il fix era
corretto. Da qui la regola: **verificare che l'artefatto sia quello nuovo** prima di
concludere che una correzione non funziona.

### 3.10 Il percorso di calibrazione, end-to-end (`calibration/pilot/`)

I test unitari coprono `fit_temperature` in isolamento e la parità ne prova l'identità
numerica col Python; nessuno dei due risponde alla domanda operativa: *un file di
calibrazione si produce, si carica e viene applicato?* Un pilota su inferenza reale lo
verifica, in 5 passi (run → righe dai logit dell'engine → fit → applicazione → test
negativo):

```
24 righe (12 choice, 12 boolean)   fingerprint e240e9f020916baa…
blocco calibration nella risposta   sì
temperature applicate               {boolean: 0.05, choice: 0.05}
probability_status                  temperature_scaled_requires_held_out_validation
probabilità cambiate                24/24 risposte (spostamento massimo 0.331)
calibrazione di un altro modello    RIFIUTATA ("Calibration was fitted for a
                                    different model/runtime/prompt configuration")
```

Il percorso funziona in ogni suo passo, compreso il rifiuto di una calibrazione estranea.

**Ma i numeri del pilota non vanno citati come calibrazione.** Il fit ha scelto T = 0,05,
il **bordo inferiore** della griglia, portando l'NLL in-sample a ~1e-9: è overfitting su
12 righe in cui il modello ha sempre ragione. È esattamente il motivo per cui
l'implementazione limita la griglia e per cui l'artefatto porta
`status: fitted_requires_held_out_validation` — lo strumento **rifiuta di dichiarare una
validazione**, e qui ha ragione. Inoltre la temperatura è una trasformazione monotona dei
logit: **non può cambiare l'argmax**, quindi l'accuratezza è invariante (24/24 prima e
dopo) e ciò che cambia è la qualità delle probabilità — che è lo scopo della calibrazione.
Con 24 righe una valutazione held-out non è possibile rispettando il vincolo di ≥10
esempi per tipo: serve un dataset reale (§5).

---

## 4 · Differenze dichiarate (le uniche strutturali)

Tutto il resto — `mode`, l'intera struttura di `answers`, `calibration`, e tutti i campi
di `timing` tranne i tre elencati — è **identico**.

| percorso | riferimento Python | questo port | motivo |
|---|---|---|---|
| `model.mlx`, `model.mlx_lm` | versioni MLX | assenti | non è un runtime MLX |
| `model.requested_revision`, `model.runtime_revision`, `model.source_files`, `model.quantization_group_size` | revisione HF pinnata, SHA-256 per file, gruppo di quantizzazione | assenti | il GGUF porta i propri metadati |
| `model.n_gpu_layers`, `model.name`, `model.path` | assenti | offload e identità del file | runtime llama.cpp |
| `model.source`, `model.device`, `model.precision`, `model.prompt_version`, `model.backend`, `model.fingerprint`, `model.load_seconds` | presenti | presenti | stesso campo, valore diverso (backend diverso) |
| `timing.evaluated_tokens_including_padding` | posizioni padded del batch | assente | artefatto del batching MLX: questo motore non fa padding |
| `timing.peak_mlx_bytes` | picco memoria MLX | assente | contabilità MLX |
| `timing.load_seconds` | sotto `model` | sotto `model` e `timing` | comodità |

Un client che legge `answers`, `mode`, `calibration` o i campi condivisi di `timing`
**non è toccato**.

---

## 5 · Cosa NON è verificato (limiti)

Li dichiaro per intero: sono la parte che di solito manca.

1. **Il livello B confronta la forma, non i valori — e oggi non può fare altrimenti.**
   Verificato sui due lati (22/09/2026): il riferimento **accetta solo Spark2.5** — con un
   controllo esplicito di architettura, «Only the Spark2.5 architecture is supported, not
   qwen35» — mentre questo port **non carica Spark2.5**: la crate `llama-cpp-2` 0.1.156 (la
   più recente, 02/09/2026) incorpora una versione di llama.cpp **precedente** al supporto
   dell'architettura `spark2_5` (il progetto originale usa la release **b11081**). Le due
   implementazioni, oggi, **non hanno nessun modello in comune**: la parità di *valori* a
   parità di modello è quindi impossibile da entrambi i lati, non per un difetto del port
   ma per un divario di dipendenza.
   **Via d'uscita**: la crate deve incorporare un llama.cpp ≥ b11081 (o linkarne uno di
   sistema). Il codice del port è agnostico rispetto al modello: quando la dipendenza
   avanza, Spark gira senza modifiche.
   La parità dei valori resta dimostrata dove la logica vive: sul core, esatto (§2.A).
2. **Un solo modello provato a fondo per le decisioni** (`Qwen3.5-4B-Instruct-SingleTurn`).
   `Qwen2.5-7B` e i tre `LFM2.5` di casa sono stati usati come controprove, non come
   candidati validati.
3. **Nessuna calibrazione di dominio — ma nemmeno l'originale ne ha una.** Il *meccanismo*
   è verificato: il fit è numericamente identico al Python e il percorso
   `calibrate → file → decide --calibration` è provato end-to-end su inferenza reale
   (§3.10), incluso il rifiuto di una calibrazione estranea. Ciò che manca è il **dataset
   etichettato di dominio**: il pilota usa 24 righe etichettate per costruzione, il cui fit
   risulta degenere (T al bordo della griglia, NLL in-sample → 0 = overfitting), quindi
   **nessuna temperatura utilizzabile in produzione** è stata prodotta e non è stata fatta
   alcuna validazione held-out.

   Vale la pena dirlo con precisione, perché riqualifica il limite: **il progetto originale
   dichiara le proprie probabilità non calibrate in quattro punti indipendenti**, e nessun
   artefatto di calibrazione è consegnato nel suo repository. Il port eredita questa
   posizione *per costruzione* — stesse stringhe, stesso default, stesso rifiuto
   strutturale di dichiarare una validazione — quindi non indebolisce alcuna garanzia
   dell'originale. Il "prossimo passo" è il medesimo per entrambi.

   | dove | cosa dichiara l'originale |
   |---|---|
   | README | «Probabilities are **uncalibrated** unless you calibrate them on your own data» |
   | `calibration.py` | `status: Literal["fitted_requires_held_out_validation"]` — valore singolo: l'artefatto **non può** dire "validato" |
   | `decisions.py` | `probability_status` = `uncalibrated_conditional_option_scores` (oppure `temperature_scaled_requires_held_out_validation`) |
   | `playground.html` | «Probabilities are uncalibrated: confidence describes the shape of the distribution, not the probability of being right» |
   | `compat.py` | la `confidence` è «not a calibrated accuracy» |
   | `results/*` | ogni risposta dei suoi benchmark pubblicati ha `"calibration": null` |

   Il port mantiene le stesse due stringhe di stato e la stessa semantica; l'unica
   differenza è che qui il percorso è stato *esercitato* fino in fondo.
4. **`evaluate` non è confrontato col Python su un dataset reale** (solo unit test + una
   esecuzione end-to-end). Le metriche sono portate fedelmente, ma la parità numerica del
   report di valutazione non è stata verificata come per il fit.
5. **Il riferimento MLX aborta al teardown** dell'interprete (`terminate called without an
   active exception`) dopo aver scritto una risposta completa: la suite tollera l'exit
   status e valida l'artefatto. È un difetto dell'ambiente di riferimento, non del port.
6. **Determinismo verificato sullo stesso backend** (due run identiche). Non ho misurato
   la variabilità cross-backend, che per costruzione esiste (ordine di somma, kernel).
7. **UI dev non portata** (playground/snake/logo → 501) e **nessun supporto per etichette
   multi-token** oltre le 26 lettere.
8. **Un solo hardware**: RTX 2070 8 GB, Linux. Nessun test su CPU-only né su altre GPU.

---

## 6 · Riproducibilità

Ambiente delle misure:

| | |
|---|---|
| nodo | Linux 7.0.0-31-generic, NVIDIA RTX 2070 8192 MiB, driver 595.91.07, CUDA 12.0 (V12.0.140) |
| toolchain Rust | 1.98.1 (nodo), 1.96.0 (postazione di sviluppo) |
| binding llama.cpp | crate `llama-cpp-2` 0.1.156 |
| riferimento Python | Python 3.12.3, mlx 0.32.2, mlx_lm 0.31.3, Spark-X2.5-4B con `bits=8` |
| modello del port | `Qwen3.5-4B-Instruct-SingleTurn.Q4_K_M.gguf` (2,7 GB, Apache-2.0) |

Comandi:

```bash
cargo test --release --features cuda        # 44 test
parity/run_parity.sh --end-to-end           # parità col riferimento Python
rizzo-flow-rs decide rizzo_test_6.json --model <gguf> --ctx 4096 --ngl 99
rizzo-flow-rs evaluate fixtures.jsonl --model <gguf> --repeats 2 --compare-modes
rizzo-flow-rs serve --model <gguf> --port 8017
```

I report grezzi delle due esecuzioni di parità sono versionati in `parity/results/`:
si possono rigenerare e confrontare, non vanno creduti sulla parola.

---

## 7 · Prima dell'adozione

- Suite di parità da estendere a un dataset reale etichettato (per la calibrazione).
- `evaluate`: verifica di parità numerica col Python come fatto per il fit.
- Etichette multi-token (>26 opzioni) via chain rule; `/metrics`.
- UI dev: decidere se portarla o lasciarla fuori (oggi 501 dichiarato).
- Licenza e attribuzione: `NOTICE` pronto; resta da confermare la forma della proposta
  (fork pubblico o pull request) — decisione vostra.
