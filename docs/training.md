# Fine-tuning LoRA / QLoRA di Spark-X2.5-4B

Guida completa per rifare da zero, su qualunque macchina con GPU NVIDIA, il fine-tuning LoRA
(o QLoRA) del modello 4B sulle decisioni tipizzate, e per valutarne il risultato sul benchmark
`LocalLLaMA/typed-decisions`. Tutti gli script stanno in [`training/`](../training/).

> **Stato al 25 settembre 2026.** Pipeline completa e verificata a pezzi (sezione
> [Verifiche](#verifiche-già-fatte)), **training completo non ancora eseguito**: il primo run
> si è interrotto allo step 20/1770 (chiusura della sessione, nessun checkpoint). Nessun numero
> di qualità del modello addestrato esiste ancora.

## Indice

1. [Idea: cosa si addestra](#1-idea-cosa-si-addestra)
2. [Requisiti](#2-requisiti)
3. [Ambiente](#3-ambiente)
4. [Download](#4-download)
5. [Dataset](#5-dataset)
6. [Script](#6-script)
7. [Smoke test](#7-smoke-test)
8. [Training](#8-training)
9. [QLoRA](#9-qlora)
10. [Export GGUF e valutazione](#10-export-gguf-e-valutazione)
11. [Macchina più potente: cosa cambiare e cosa no](#11-macchina-più-potente-cosa-cambiare-e-cosa-no)
12. [Verifiche già fatte](#verifiche-già-fatte)
13. [Problemi noti e decisioni](#13-problemi-noti-e-decisioni)

---

## 1. Idea: cosa si addestra

Rizzo Flow non genera testo. Per ogni domanda il modello legge un prompt (system prompt,
`<evidence>` con lo state, domanda a scelta multipla con opzioni `A.`, `B.`, …), fa **un solo
forward pass** e all'ultima posizione si leggono i logit delle sole lettere ammesse: la softmax su
quei logit è la risposta.

Il training usa **esattamente lo stesso meccanismo**:

1. Il prompt è compilato da `rizzo_flow.prompts.compile_request` (lo stesso codice del server,
   prompt `spark-decisions-v3`): un esempio di training è byte per byte ciò che il modello vede
   in produzione.
2. Forward pass, logit delle lettere all'ultima posizione reale.
3. **Loss = cross-entropy morbida** tra la distribuzione target e la softmax sulle lettere:
   `loss = −Σₖ targetₖ · log softmax(logit_lettere)ₖ`. Il target può essere one-hot (risposta
   certa), morbido (voti di annotatori umani divisi) o una probabilità esatta (es. 0.667).
4. Il gradiente aggiorna **solo gli adapter LoRA**; i 4 miliardi di pesi del modello sono
   congelati.

Nessun altro token del vocabolario è supervisionato: si insegna a spostare probabilità tra le
lettere, non a scrivere. Quando nei log compare `tok/s` si intendono **token di prompt letti**
(avanti e indietro) al secondo, non token generati.

## 2. Requisiti

| | Minimo provato | Note |
| --- | --- | --- |
| GPU | NVIDIA 16 GB (RTX 5060 Ti, Blackwell sm_120) | picco misurato 11.5 GiB con i parametri sotto |
| RAM | 48 GB (32 GB dovrebbero bastare) | l'export carica il modello e salva ~8 GB |
| Disco | ~40 GB liberi | modello 8 GB, dati 0.2 GB, ogni export ~21 GB (HF + GGUF BF16 + Q8_0) |
| Driver | 595.97, wheel PyTorch CUDA 12.8 | Blackwell richiede CUDA ≥ 12.8 |
| Python | 3.12 | |
| OS | Windows 10 provato; Linux previsto (percorsi indipendenti dall'OS) | |

Tempo su RTX 5060 Ti: **~740 token/s**, 14.7 M token per epoca → **5.5–6.7 ore** per epoca
(LoRA BF16 e QLoRA vanno alla stessa velocità). La GPU è al limite di calcolo: il tempo scala
circa con i TFLOPS BF16 della scheda (vedi [sezione 11](#11-macchina-più-potente-cosa-cambiare-e-cosa-no)).

## 3. Ambiente

Due ambienti Python separati:

- `.venv` — quello del progetto (llama.cpp, `rizzo`), serve per smoke test di parità e valutazione.
- `.venv-train` — PyTorch + transformers + peft, solo per il training.

```bash
# progetto (se non c'è già): runtime llama.cpp + GGUF ufficiali
uv sync --locked
.venv/bin/rizzo download                      # Windows: .venv/Scripts/rizzo download
.venv/bin/rizzo download --only weights --quant bf16   # serve allo smoke test di parità

# ambiente di training
uv venv .venv-train --python 3.12
uv pip install -p .venv-train --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
uv pip install -p .venv-train -r training/requirements.txt
uv pip install -p .venv-train -e . --no-deps   # rizzo_flow importabile (prompt e schema)
```

Versioni esatte in [`training/requirements.txt`](../training/requirements.txt). `transformers`
è fissato a **4.57.1**, la versione che ha scritto il checkpoint. Su Windows gli eseguibili sono
in `.venv-train/Scripts/`, su Linux in `.venv-train/bin/`: negli esempi sotto `PY` sta per
`.venv-train/Scripts/python` o `.venv-train/bin/python`.

## 4. Download

Un solo comando scarica tutto alle revisioni fissate, riprende i download interrotti e verifica
lo sha256 dei pesi:

```bash
$PY training/download_data.py                 # tutto
$PY training/download_data.py --only model    # model | data | bench | sources
```

| Cosa | Sorgente | Revisione | Dove finisce |
| --- | --- | --- | --- |
| Checkpoint HF 4B (~8 GB, BF16) | `XHToken/Spark-X2.5-4B` | `0bcb356…` | `models/Spark-X2.5-4B/` |
| Dati di training | vedi [sezione 5](#5-dataset) | fissate | `.research/train-data/raw/` |
| Benchmark (test + train) | `LocalLLaMA/typed-decisions`, config `all` | `c76749e…` | `.research/typed-decisions/all/*.jsonl` |
| Fixture SemIf (solo esclusione) | `github.com/TheoLeeCJ/SemIf` | `ca3ba65…` | `.research/SemIf/` |
| Sorgente llama.cpp (solo export) | `github.com/ggml-org/llama.cpp` tag `b11081` | `161755f…` | `.research/llama.cpp-b11081/` |

**Codice remoto.** Il checkpoint HF si carica con `trust_remote_code=True`
(`modeling_spark.py`, `configuration_spark.py`). Il codice è stato letto prima di eseguirlo:
definisce solo l'architettura, niente rete, file o `exec`. Hash dei file letti:

```
9cf0d1ad2b54b9f7088792779ddd8b5cb4d4fe63bfea054da7f2dd16362bcaf4  modeling_spark.py
02218597240490f490659b184052db940b08163ff9c3af7b7e59323f6f963722  configuration_spark.py
```

Il modello è Apache-2.0: il fine-tuning è permesso.

## 5. Dataset

### Sorgenti incluse

| Sorgente | Revisione | Licenza | Etichette | Cosa si usa |
| --- | --- | --- | --- | --- |
| [`tasksource/procedural-typed-decisions`](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) | `b4f55be` | Apache-2.0 | calcolate esattamente da regole (niente rumore) | tutte le 11 config |
| [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev) `release-v2-redistributable` | `c67699e` | CC0 | programmatiche (oracoli, minimax, BFS, simulatori) | tutto **tranne** 2 sorgenti (sotto) |
| [`Praveenrajus/jev-bench`](https://huggingface.co/datasets/Praveenrajus/jev-bench) | `b41e6b2` | mista, per sorgente | umane (con voti degli annotatori dove esistono) | 12 config su 22 (sotto) |

Nessuna etichetta viene da Jev/TypeSafe (niente distillazione): in Open-Jev il campo
`teacher` dei metadati indica script (`tile_platformer_bfs_baseline`, oracolo fisico T-Rex),
non modelli.

**jev-bench, config usate** (licenza fra parentesi): `civil_comments` (CC0),
`strategyqa_closed`, `strategyqa_grounded`, `mmlu` (MIT), `paws` (uso commerciale libero),
`helpsteer2_helpfulness`, `helpsteer2_verbosity`, `measuring_hate_speech` (CC-BY-4.0: citare),
`arc_challenge`, `boolq`, `fever_evidence`, `stsb` (**CC-BY-SA**: incluse per decisione
dell'utente del 25/09; se si pubblica l'adapter valutare lo share-alike).

### Escluso, e perché

| Escluso | Motivo |
| --- | --- |
| `AndeyTait/JevForge-Mind2Web` (intero) | qualità: nel 28% delle righe la descrizione dell'elemento giusto è identica a un altro candidato, nel 40% è generica (`div`, `a`) → etichetta non ricavabile dallo state |
| Open-Jev `customer-control-v1` (4.206 righe) | usa descrizioni della doc TypeSafe con licenza dichiarata "non verificata" dagli autori |
| Open-Jev `workflow-controls-v1` (22.416 righe) | stessi 4 workflow di typed-decisions (0 state in comune, ma addestrarci toglierebbe il "zero-shot" al benchmark) |
| jev-bench `banking77`, `clinc150`, `massive`, `ledgar`, `go_emotions` | più di 26 opzioni: oltre il limite di lettere A–Z |
| jev-bench `mnli`, `yelp5` | licenza solo ricerca |
| jev-bench `sst5`, `sms_spam` | licenza non dichiarata / sconosciuta |
| jev-bench `chaosnli` | ha solo lo split di test |
| split `test`/`ood`/`calibration` di tutte le sorgenti | si usano solo `train` (training) e `validation` (dev) |
| typed-decisions `train` | il benchmark si valuta zero-shot: nessun dato di quei workflow |

### Esclusione dei set di valutazione (contaminazione)

`build_data.py` raccoglie tutte le stringhe ≥ 60 caratteri di: typed-decisions `test`,
`benchmarks/smoke.jsonl`, `benchmarks/perturbations.jsonl`, SemIf `authored144`,
`perturbations108`, `shape777` (703 stringhe) e **scarta ogni state di training che ne
contiene una**. Se uno di quei file manca lo script si ferma: altrimenti il controllo
passerebbe in silenzio. Risultato del build del 25/09: **0 state contaminati**.

### Conversione in esempi

Ogni riga di ogni sorgente diventa una richiesta nel formato wire System One
(`state` + domanda `noul`/`choice`/`score`), tradotta con `compat.to_native` (come
`POST /v1/systemone`) e compilata con `prompts.compile_request`. Un esempio = **una domanda**:
`{group, id, tokens, slots, target}`, dove `slots` sono gli ID dei token-lettera e `target` la
distribuzione nell'ordine delle lettere.

| Tipo | Target |
| --- | --- |
| `noul` | `[1 − p, p]` (lettere A = no, B = sì) |
| `choice` | probabilità per opzione, nell'ordine delle `criteria` |
| `score` | probabilità per livello |
| etichetta umana secca (jev-bench senza `soft_label`) | one-hot con **label smoothing 0.03** |
| etichetta umana morbida (`soft_label`) | la distribuzione dei voti, così com'è |
| etichetta programmatica (tasksource, Open-Jev) | così com'è, senza smoothing (è esatta) |

Alcune domande `score` di tasksource hanno 11 livelli: fuori dal limite di 10 dell'API
compatibile, validi per quella nativa (fino a 26). La conversione salta i limiti del wire
format; lo schema nativo valida comunque tutto.

### Campionamento

Tetti per gruppo (numero di **domande**, non di state), seed 0:

| Famiglia | train / gruppo | dev / gruppo | Gruppi |
| --- | ---: | ---: | --- |
| tasksource | 1.400 | 120 | 11 config |
| jev-bench | 800 | 80 | 12 config |
| Open-Jev | 700 | 60 | per sorgente (7): pittura pixel e giochi sono il 60% di Open-Jev, qui ridimensionati |

Prompt oltre **2.048 token scartati, mai troncati**: 133 in train, 13 in dev.

### Risultato (build del 25/09/2026)

| | Domande | Token totali | Mediana | Massimo |
| --- | ---: | ---: | ---: | ---: |
| train | 28.321 | 14.739.738 | 460 | 2.043 |
| dev | 2.604 | 1.324.018 | 445 | 2.043 |

Composizione train: tasksource 15.012, jev-bench 9.081, Open-Jev 4.228 (dev: 1.290 / 958 / 356).
Il build è deterministico: sha256 attesi

```
4997776bb794c412037213a5f447da901ace1db494b71f6da4dc790897486da9  .research/train-data/train.jsonl
d87d1d704f083ba5283f5442a95838d5b82f8ccb0ced7a4cf221a9df716097ba  .research/train-data/dev.jsonl
```

Distribuzione delle lunghezze (train):

```
  129-256    4922  17.4%  ####################################
  257-384    6924  24.4%  ##################################################
  385-512    4714  16.6%  ##################################
  513-768    6525  23.0%  ###############################################
  769-1024   3558  12.6%  ##########################
 1025-1280   1051   3.7%  ########
 1281-1536    278   1.0%  ##
 1537-1792    192   0.7%  #
 1793-2048    157   0.6%  #
```

p25 275, p75 701, p90 906, p95 1.065, p99 1.613. Il minimo è 179 perché system prompt e template
occupano già ~170 token. La coda lunga viene da `needle_retrieval` (max 2.043),
`helpsteer2_*` (~2.030), `record_aggregation` (1.958), `table_lookup` (1.868).

## 6. Script

| Script | Ambiente | Cosa fa |
| --- | --- | --- |
| [`training/common.py`](../training/common.py) | train | percorsi, log con orario, `Progress` (step N/M, tempo trascorso, ETA), tokenizer, caricamento del modello BF16/NF4, tetto memoria GPU |
| [`training/download_data.py`](../training/download_data.py) | train | download di modello, dati, benchmark, sorgenti (sezione 4) |
| [`training/build_data.py`](../training/build_data.py) | train | costruisce `train.jsonl`, `dev.jsonl`, `manifest.json` (sezione 5) |
| [`training/smoke_parity.py`](../training/smoke_parity.py) | progetto + train | parità transformers ↔ llama.cpp su 40 decisioni |
| [`training/check_batching.py`](../training/check_batching.py) | train | il padding a destra non cambia la lettura oltre il rumore BF16 |
| [`training/train_lora.py`](../training/train_lora.py) | train | training LoRA/QLoRA, valutazione sul dev, checkpoint, ripresa |
| [`training/export_gguf.py`](../training/export_gguf.py) | train | fusione dell'adapter → checkpoint HF → GGUF BF16 → GGUF Q8_0 |
| [`scripts/typed_decisions.py`](../scripts/typed_decisions.py) | progetto | benchmark typed-decisions (anche su un GGUF qualsiasi con `--model`) |

Ogni ciclo lungo stampa righe `step N/M (x%) elapsed … ETA …`.

## 7. Smoke test

Da fare su ogni macchina nuova, prima del training (pochi minuti):

```bash
# 1) parità: token identici e stesse risposte tra transformers e llama.cpp (BF16)
.venv/bin/python training/smoke_parity.py llama ref.json    # progetto, llama.cpp
$PY training/smoke_parity.py hf ref.json                     # training, transformers
# atteso: token mismatches 0, argmax flips 0, |dp| mediana ≈ 0.0003 → PARITY OK

# 2) batching
$PY training/check_batching.py
# atteso: OK (padded ≈ 0.08 contro 0.06 di copie senza padding: rumore BF16)

# 3) mini-training: 10 step, deve scendere la loss e non andare in OOM
$PY training/train_lora.py --max-steps 10 --grad-accum 8 --dev-limit 50 --out .research/lora-runs/smoke
```

Perché le soglie non sono "identico": due implementazioni BF16 diverse (e lo stesso modello
con forme di batch diverse) danno probabilità che differiscono fino a ~0.08 **solo sulle
risposte incerte** (p tra 0.2 e 0.8). Un errore strutturale sposterebbe anche le risposte
sicure. Risultati del 25/09 su RTX 5060 Ti: parità OK (0 token diversi, 0 risposte cambiate su
40); batching OK; 10 step di mini-training: loss dev da 1.95 a 1.14.

## 8. Training

### Comando usato (run principale)

```bash
$PY training/train_lora.py --quant bf16 --r 16 --lr 5e-5 --epochs 1 \
    --grad-accum 16 --batch 8 --max-batch-tokens 4096 \
    --dev-limit 600 --eval-every 200 --log-every 10 \
    --out .research/lora-runs/lora-bf16-r16 > .research/lora-runs/lora-bf16-r16.log 2>&1
```

Se si interrompe, si riparte dall'ultimo checkpoint con **gli stessi parametri**:

```bash
$PY training/train_lora.py --quant bf16 --r 16 --lr 5e-5 --epochs 1 --grad-accum 16 \
    --batch 8 --max-batch-tokens 4096 --resume .research/lora-runs/lora-bf16-r16/step-800
```

### Configurazione LoRA

| Parametro | Valore | Perché |
| --- | --- | --- |
| rango `r` | **16** | compito stretto (spostare probabilità tra poche lettere); con ~6 h per run non si può fare una ricerca su `r`. Se il dev mostra overfitting presto, prossimo tentativo `r = 8` |
| `lora_alpha` | **32** (= 2·r) | scala standard 2 |
| `lora_dropout` | 0.05 | |
| `bias` | `none` | |
| moduli | `q_k_v_proj`, `g_proj`, `out_proj`, `gate_proj`, `up_proj`, `down_proj` (tutti i layer lineari di attenzione e MLP, 36 layer) | |
| esclusi | embedding e `lm_head` | sono **condivisi** (`tie_word_embeddings`): toccarli cambierebbe anche l'input |
| parametri addestrabili | 32,4 M (0,8% di 4 B) | |

### Ottimizzazione

| Parametro | Default | Usato | Note |
| --- | --- | --- | --- |
| `--lr` | 5e-5 | 5e-5 | jev-bench ha trovato lr bassi (3e-5) migliori per calibrazione e generalizzazione rispetto a 1e-4 |
| schedule | warmup lineare + coseno fino a 0 | | `--warmup 0.03` (3% degli step) |
| ottimizzatore | AdamW, weight decay 0 | | |
| clipping | norma dei gradienti 1.0 | | |
| `--grad-accum` | 16 | 16 | **domande per step** di ottimizzazione (batch effettivo). I gradienti sono riscalati sul numero reale di domande dello step |
| `--epochs` | 1.0 | 1 | 28.321 / 16 = **1.770 step** |
| `--max-steps` | – | – | sostituisce `--epochs` (per prove) |
| `--seed` | 0 | 0 | ordine dei dati e init LoRA |

### Batch e memoria

| Parametro | Default | Note |
| --- | --- | --- |
| `--batch` | 8 | domande per forward (massimo) |
| `--max-batch-tokens` | 4096 | token **con padding** per forward. 8192 va in OOM su 16 GB (4 × 2048) |
| gradient checkpointing | attivo | `--no-checkpointing` lo spegne (serve molta più memoria) |
| tetto memoria | 92% della VRAM | su Windows il driver, finita la VRAM, usa la RAM di sistema e il run si trascina invece di fallire: col tetto si ha subito un OOM chiaro |

I batch sono formati per lunghezza (pool di 64 × batch mescolato, ordinato, tagliato, poi i
batch sono mescolati) con padding a destra; si legge solo lo stato nascosto dell'ultima
posizione reale di ogni domanda e lo si proietta solo sulle righe-lettera dell'embedding (nessun
logit sull'intero vocabolario).

**Pre-volo.** Prima del training lo script esegue forward + backward sui due batch più pesanti
del piano: più token con padding (attivazioni) e più `n × L²` (l'attenzione del modello è solo
"eager", cresce col quadrato della lunghezza). Se va in OOM lo fa in 30 secondi e non dopo ore.
Misurato: `2 × 2043` → 11,48 GiB, `4 × 1024` → 10,86 GiB.

### Valutazione, checkpoint, log

- `--dev-limit 600` domande del dev (mescolate con seed 1), valutate allo step 0, ogni
  `--eval-every 200` step e alla fine: loss (cross-entropy morbida) e accuratezza (argmax
  contro argmax del target), anche per famiglia.
- Ogni valutazione salva `RUN/step-N/` (adapter + `trainer.pt` con stato dell'ottimizzatore,
  ordine dei dati, RNG di Python, PyTorch e CUDA) e `RUN/history.json`. Alla fine `RUN/final/`.
  Con `--resume` il run continua **esattamente** come se non si fosse fermato (verificato);
  i parametri devono essere gli stessi, altrimenti lo script si rifiuta (possono cambiare solo
  `--eval-every`, `--log-every`, `--dev-limit`). Si perde al massimo il lavoro dopo l'ultimo
  checkpoint (200 step ≈ 45 minuti su 5060 Ti).
- Il log riporta ogni `--log-every` step:
  `train step 10/1770 (0.6%) elapsed 0h02m28s ETA 7h14m54s loss 2.4873 lr 9.43e-06 739 tok/s peak 11.3 GiB`.
- Una directory di run non viene mai sovrascritta (`--out` esistente → errore).

Baseline del dev (modello non addestrato, 600 domande): loss 1.914, accuratezza 0.545
(jev-bench 0.601, Open-Jev 0.384, tasksource 0.551).

## 9. QLoRA

`--quant nf4` carica il modello base in NF4 (bitsandbytes, doppia quantizzazione, calcolo in
BF16) e addestra gli stessi adapter.

- Il codice remoto converte le attivazioni nel dtype di `mlp.gate_proj.weight`, che in NF4 è
  `uint8`: senza correzione il forward fallisce (`"baddbmm_cuda" not implemented for 'Byte'`).
  `common.patch_quantized_cast` sostituisce a runtime il forward del layer con uno identico che
  converte in BF16. Verificato: applicato al modello BF16 dà risultati identici all'originale.
- **La base NF4 è molto più debole**: su 48 domande del dev loss 2,88 contro 1,88,
  accuratezza 0,40 contro 0,54 (il Q4_K_M di llama.cpp perdeva solo ~4 punti su SemIf).
- Velocità uguale alla LoRA BF16 (~720 tok/s), picco 9,0 GiB contro 11,5.

Con 16 GB la QLoRA non serve per far stare il modello in memoria; è prevista come **run di
confronto** (stessi parametri, `--quant nf4`, `--out .research/lora-runs/qlora-nf4-r16`).
Un adapter QLoRA fuso nei pesi BF16 non è esattamente il modello addestrato (era addestrato
sopra pesi NF4): va valutato così e dichiarato.

## 10. Export GGUF e valutazione

Il server usa llama.cpp, quindi la valutazione finale si fa sul GGUF:

```bash
# fusione dell'adapter nei pesi BF16 → HF → GGUF BF16 → GGUF Q8_0
$PY training/export_gguf.py .research/lora-runs/lora-bf16-r16/final lora-bf16-r16

# benchmark, stesso script e stesso split del numero di partenza
.venv/bin/python scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl \
    --model .research/merged/lora-bf16-r16/lora-bf16-r16-q8_0.gguf \
    --output results/local-typed-decisions/lora-bf16-r16-q8_0.json
```

`export_gguf.py --base NOME` esporta il modello **senza** adapter: serve a verificare che la
pipeline di conversione riproduca il GGUF ufficiale di XHToken prima di fidarsi dei numeri.
La conversione usa `convert_hf_to_gguf.py` di llama.cpp allo stesso commit dei binari
(`161755f`, che ha `conversion/spark2_5.py`) e `llama-quantize` da `runtimes/`.

**Verifica della pipeline (25/09).** `export_gguf.py --base` (7 minuti su RTX 5060 Ti) produce
GGUF BF16 e Q8_0 con **tutti i 290 tensori identici bit per bit** a quelli ufficiali di XHToken;
differiscono solo i metadati `general.name` e `general.sampling.penalty_repeat`, che Rizzo Flow
non usa. Il chat template (da cui il backend costruisce i prompt) è quello di
`chat_template.jinja`, come nel GGUF ufficiale: lo script toglie la copia compattata da
`tokenizer_config.json`, che il convertitore preferirebbe. Le due versioni del template
producono comunque lo stesso testo su tutti i 2.000 prompt di typed-decisions.

**1.7B e pubblicazione per MLX.** `RIZZO_TRAIN_SIZE=1.7b` punta tutti gli script al checkpoint
piccolo (`models/Spark-X2.5-1.7B`, a cui servono anche `configuration_spark.py`,
`modeling_spark.py` e `merges.txt` della stessa revisione: identici a quelli del 4B). La cartella
`hf/` dell'export è il checkpoint fuso che il backend MLX scarica dalla radice dei repo HF: prima
di pubblicarla va rimesso il `tokenizer_config.json` originale. Rifatto il 25/09 su RTX 5060 Ti a
partire dagli adapter pubblicati: i GGUF BF16 e Q8_0 ottenuti hanno lo stesso sha256 di quelli
prodotti sulla RTX PRO 6000, quindi la fusione è deterministica e i safetensors sono i pesi esatti.

Numeri di partenza da battere (modello non addestrato, 25/09, RTX 5060 Ti, llama.cpp CUDA):

| | Accuracy ↑ | KL from gold ↓ | Brier ↓ | p50 per caso |
| --- | ---: | ---: | ---: | ---: |
| Spark 4B Q8_0 | 0.574 | 2.899 | 0.480 | 201 ms |
| Spark 4B BF16 | 0.574 | 2.935 | 0.479 | 250 ms |
| TypeSafe Jev 1.13.0 (dalla card) | 0.727 | 1.442 | 0.148 | 710 ms |
| Prior (ignora l'input) | 0.470 | 0.347 | 0.189 | – |

Debolezza principale di partenza: `agent_trace_observability` 0.366 (risponde quasi sempre
"Critical"/"human_review" con p ≈ 0.9999). Oltre a typed-decisions vanno ripetuti smoke
(`rizzo evaluate benchmarks/smoke.jsonl`) e SemIf (`scripts/semif_compare.py`) per controllare
che il fine-tuning non peggiori altrove. Le probabilità restano non calibrate finché non si
applica `rizzo calibrate`.

## 11. Macchina più potente: cosa cambiare e cosa no

**Si può cambiare** (non tocca il risultato, solo velocità e memoria):

- `--batch` e `--max-batch-tokens`: con 24 GB provare `--max-batch-tokens 8192`, con 40–80 GB
  `16384`; il pre-volo dice subito se ci sta.
- `--no-checkpointing`: ~1/3 di calcolo in meno, molta più memoria (su 16 GB impossibile).
- Il tetto al 92% della VRAM (`common.load_model`) è pensato per Windows; su Linux è innocuo.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (impostato da `common.py`) riduce la
  frammentazione su Linux; su Windows PyTorch lo ignora con un avviso innocuo.

Su una GPU col doppio dei TFLOPS BF16 il tempo si dimezza circa (la 5060 Ti lavora al limite
di calcolo: il batching non la accelera).

**Non cambiare** se si vuole confrontare con i numeri di questo documento: `--grad-accum 16`
(batch effettivo), `--lr`, `--epochs`, `--r`, `--alpha`, `--seed`, i dati (`train.jsonl` con lo
sha256 sopra), il prompt (`PROMPT_VERSION` in `src/rizzo_flow/prompts.py`). Cambiare il prompt
dopo il training rende l'adapter inutile: è addestrato su quel testo esatto.

## Verifiche già fatte

Tutte il 25/09/2026 su Windows 10, RTX 5060 Ti 16 GB:

| Verifica | Esito |
| --- | --- |
| Tokenizer HF = tokenizer del GGUF | identici su 40 prompt |
| Parità transformers ↔ llama.cpp BF16 | 0 risposte cambiate su 40, \|Δp\| mediana 0,0003, max 0,082 (solo su risposte incerte) |
| Padding a destra | entro il rumore BF16 (0,079 contro 0,061 senza padding) |
| Correzione NF4 | identica all'originale sul modello BF16 |
| Contaminazione | 0 state di training contengono testo dei set di valutazione |
| Mini-training 10 step | loss dev 1,95 → 1,14 |
| Pre-volo memoria | OOM a 8.192 token/forward, 11,5 GiB a 4.096 |
| `--resume` | ripreso da `step-2` di un run di 4 step: loss degli step 3 e 4 identiche al run non interrotto (0,7886 / 0,5349) |
| Export base → GGUF | 290 tensori identici a XHToken in BF16 e Q8_0; template equivalente su 2.000 prompt |

## 13. Problemi noti e decisioni

- **Attenzione solo "eager"** nel codice del modello: niente SDPA/flash-attention, memoria
  quadratica nella lunghezza. Da qui il limite di 2.048 token per prompt e 4.096 per forward.
- **Posizioni RoPE da `cache_position`**, non da `position_ids`: il padding deve stare a destra e
  non si possono impacchettare più esempi nella stessa sequenza.
- **Run interrotto il 25/09** allo step 20/1770 (la sessione che lo aveva lanciato si è chiusa):
  da qui `--resume` e i checkpoint con stato dell'ottimizzatore. Lanciare i run lunghi in un
  terminale che resta aperto (o `nohup`/`tmux` su Linux).
- Primo tentativo andato in OOM a 8.192 token per forward dopo ~25 step (log in
  `.research/lora-runs/lora-bf16-r16-oom.log`).
- Decisioni dell'utente (25/09): includere i dataset CC-BY-SA di jev-bench, escludere
  `workflow-controls-v1`, rivalutare su typed-decisions dopo il training. Rango LoRA scelto da
  Claude. Ogni script di training deve loggare step N/M ed ETA.
- Tutti i file prodotti (`.research/`, `models/`, `runtimes/`, `.venv-train/`) sono ignorati da git.
