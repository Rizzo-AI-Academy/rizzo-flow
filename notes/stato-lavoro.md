# Rizzo Flow — stato del lavoro e note estese

Spostato da `CLAUDE.md` per tenerlo sotto le 200 righe. Contenuto invariato.

## Demo Snake (`GET /snake`)

- `GET /snake` (`snake.html`, stessa impostazione del playground: pagina singola, nessuna dipendenza,
  bilingue con `I18N`/`data-i18n`): Snake in cui ogni mossa è una `POST /v1/decisions` (`choice` sulle
  mosse legali, opzioni mescolate, opzionale domanda `score` "pericolo" sullo stesso state). Lo state
  è costruito in `buildRequest()` in tre viste: sensori per mossa (default; il modello gioca bene),
  sensori + griglia ASCII, sola griglia (muore subito). Il testo del prompt lato gioco è sempre in
  inglese ed è stato provato sul modello reale: numeri in README. Registratore GIF integrato
  (`startRecording`/`capture`/`gifFrame`: canvas offscreen 800×480 griglia + pannello decisioni,
  palette per frame da istogramma a 15 bit, LZW, durata reale dei frame, max 900 frame, download via
  `<a download>`). `paintBoard` è condivisa tra canvas visibile e registratore.

## Stato del lavoro (22 settembre 2026)

### Fine-tuning LoRA come default (25 settembre 2026)

Addestrato dall'utente con `training/` (pipeline in `docs/training.md`) su RTX PRO 6000, pesi fusi
e pubblicati in GGUF sui due repo HF (resi pubblici il 25 settembre su richiesta dell'utente).
`--weights flow|base` su CLI e script, per entrambi i backend: MLX scarica dagli stessi repo il
checkpoint BF16 safetensors fuso (radice del repo, `config.FLOW_CHECKPOINTS`, sha256 dei shard
fissati; solo i file di `CHECKPOINT_FILES`, niente GGUF né adapter) in `models/rizzo-flow*/`.
Rifatto qui con `training/export_gguf.py` (`RIZZO_TRAIN_SIZE=1.7b` per il piccolo): il GGUF BF16
ottenuto ha lo **stesso sha256** di quello pubblicato, quindi i safetensors sono i pesi esatti;
`tokenizer_config.json` caricato è quello originale (l'export toglie il template solo per la
conversione). MLX-CUDA BF16 = llama.cpp BF16 entro 0.001 sul ticket, stessi `prompt_sha256`. Il fingerprint dei file
fine-tuned ha la chiave in più `weights: "flow"` (quelli base restano invariati), l'ID servito è
`rizzo-flow-4b-q8_0` (base: `rizzo-spark-x2.5-4b-q8_0`). `llama_release.fetch` accetta un token
(`config.hf_token`: `HF_TOKEN`, `.env` nella cartella di lavoro, token di `hf auth login`),
mandato solo al primo host e non nei redirect; 401/403/404 falliscono subito.
typed-decisions test (`scripts/typed_decisions.py`, report in `results/local-typed-decisions/`,
ignorati da git), stessa macchina RTX 5060 Ti, Q8_0: 4B 0.574 → **0.648** (+0.074 [+0.050,
+0.101]), KL 2.90 → 0.45, Brier 0.480 → 0.205, ECE 0.349 → 0.112; 1.7B 0.530 → 0.544 (rumore),
KL 3.03 → 0.69, `security_incidents` 0.612 → 0.514. Q4_K_M: 4B 0.650 (pari a Q8_0, +0.002
[−0.011, +0.015]), 1.7B 0.490 (−0.054 [−0.070, −0.037]). Jev dalla card: 0.727.
Fixture SemIf, 4B Q8_0 fine-tuned (`results/semif-compare/rizzo-flow-q8_0-v3-llama-cuda/`, SemIf
**non** rieseguito: confronto con le sue predizioni pubblicate): authored144 **0.845** (base
0.812, SemIf 0.819; vs SemIf +0.027 [−0.038, +0.099], pari), perturbations108 **0.946** (base
0.848, SemIf 0.766; vs SemIf **+0.180 [+0.108, +0.267]**, prima superiorità dimostrata), held-out
0.809 / 0.949, flip 1/1/1, `rule_application` perturbata 0.870 (NLL 0.33). **Peggiora
sull'evidenza mancante**: 0.583 (base 0.750, SemIf 0.861), 5 scelte sicure sbagliate su 36.
Latenza uguale ai pesi base (rilanciati in sequenza lo stesso giorno: 66 ms entrambi; quel giorno
la macchina era più lenta dei 49 ms del 22/9). Landing, README, `results/README.md` aggiornati.
**Non ancora rimisurati sul fine-tuning:** smoke, 1.7B su SemIf, demo della landing (`DEMO` in
`docs/index.html`, registrato sui pesi base): le altre tabelle storiche sotto sono dei pesi base.

### Passaggio a llama.cpp (22 settembre 2026, su `main`)

Origine: la PR #1 di BiG86 (backend llama.cpp per AMD, ctypes su una build compilata a mano,
solo Linux). Non mergiata per scelta dell'utente; **riscritto da zero** (nessun codice della PR),
con llama.cpp come default e MLX opzionale (scelte dell'utente). Differenze di progetto rispetto
alla PR: binari precompilati ufficiali con sha256 invece della compilazione, GGUF ufficiali di
XHToken invece di una conversione di terzi, device enumerati da ggml invece di sondare `/sys`,
cache unificata (prefisso condiviso, non `ctx × (batch+1)`), Windows/macOS/Linux.

Provato **da noi solo** su Windows 10 + RTX 5060 Ti, build CUDA 13.4 e build Vulkan sulla stessa
scheda. Segnalazioni della community (24 settembre 2026, non riprodotte da noi): M3 Pro/Metal
(issue #5, con le due correzioni della PR #4: contesto liberato prima dell'uscita, avviso Rosetta),
Radeon 780M/Vulkan su Linux (#11), Iris Xe/Vulkan e sola CPU su Windows (#7), RX 7900 XTX/Vulkan
su Linux (PR #6: la sonda CUDA ora chiede un device). Mai eseguiti: ROCm, SYCL, macchina senza
GPU; il Mac dell'utente (M4 Pro) non ancora — regola dell'utente: su questo PC niente prove su
CPU. `--kv-type q8_0|q4_0` (PR #10) misurato solo su CUDA. Prima cosa da fare sul Mac:
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
- **Provare llama.cpp sul Mac dell'utente (M4 Pro, Metal)**; ROCm, SYCL e `--kv-type` su Vulkan
  restano da provare.
- PR aperte non unite: #8 (tipi strutturati; saltata dall'utente: CRLF e conflitti) e #2 (port
  Rust: da decidere, proposto un repository separato).
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
