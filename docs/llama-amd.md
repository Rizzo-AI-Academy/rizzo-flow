# Rizzo Flow su GPU AMD con llama.cpp

Questo documento descrive il backend `llama` — l'alternativa a MLX per girare Rizzo Flow su GPU
AMD. Il livello decisionale (`Engine`, `prompts`, `decisions`, schema, API) è **identico**: cambia
solo il motore di inferenza sotto `Engine`.

```
Rizzo Flow (Engine, prompts, decisions, API)
        │
        ▼
LlamaBackend          src/rizzo_flow/backend_llama.py
        │
        ▼
LlamaRuntime (ctypes) src/rizzo_flow/llama_runtime.py
        │
        ▼
libllama.so · build Vulkan · commit pinnato b29c606 (tag v0.4.1)
        │
        ▼
Spark-X2.5-4B GGUF Q8_0 (stornic56, revision e sha256 pinnati)
```

Non si usa `llama-cpp-python`, non si usa Ollama, non si tenta di far girare MLX su Radeon.
`spark2_5` è supportato dal mainline di llama.cpp dal 6 settembre 2026
([PR #27868](https://github.com/ggml-org/llama.cpp/pull/27868)), quindi non serve nessun fork.

## 1 · Prerequisiti

Ubuntu 24.04, driver Vulkan, `cmake` di sistema (quello su `PATH` può essere uno shim rotto),
`ninja`, `ccache`, `g++`:

```bash
sudo apt-get install -y glslc libvulkan-dev spirv-headers glslang-tools
/usr/bin/cmake --version     # >= 3.20
vulkaninfo --summary         # il driver Vulkan deve vedere la GPU
```

`glslc` e `spirv-headers` servono solo per **compilare** gli shader Vulkan; a runtime bastano
`libllama.so` e `libggml-vulkan.so`.

## 2 · Build di llama.cpp (una volta, poi pinnata)

```bash
git clone --depth 1 --branch v0.4.1 https://github.com/ggml-org/llama.cpp ~/git/llama.cpp-amd
cd ~/git/llama.cpp-amd
/usr/bin/cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF
/usr/bin/cmake --build build --target llama-cli llama-tokenize -j"$(nproc)"
./build/bin/llama-cli --list-devices     # Vulkan0: Radeon RX 7900 XTX (RADV NAVI31)
```

La build condivisa produce `libllama.so`, `libggml.so`, `libggml-cpu.so`, `libggml-vulkan.so`
in `build/bin`. Rizzo Flow carica le ggml prima di `libllama` e dichiara le firme C del commit
pinnato in `llama_runtime.py` (`LLAMA_COMMIT`). Una build diversa può cambiare l'ABI delle
struct: **aggiornare la costante insieme al commit**.

Indica la directory a Rizzo Flow:

```bash
export RIZZO_LLAMA_LIB=~/git/llama.cpp-amd/build/bin
```

Senza la variabile il default è `llama.cpp-amd/build/bin` relativo alla directory corrente.

## 3 · Download del GGUF (artefatto pinnato)

```bash
export RIZZO_MODEL_DIR=/mnt/model-cache/rizzo-flow   # opzionale: default models/
uv sync --locked --extra llama --extra test
.venv/bin/rizzo download --format gguf --quant q8_0    # 4.375 GB
.venv/bin/rizzo download --format gguf --quant bf16    # 8.23 GB
```

Il download verifica lo **sha256 pinnato** in `config.GGUF_MODELS` e riprende solo se il file
manca o è corrotto. Con `--gguf PATH` si può caricare un GGUF qualsiasi: in quel caso lo sha256
non è imposto e il campo `gguf_source` nei metadati resta `null` (provenienza non verificata).

## 4 · Uso

```bash
.venv/bin/rizzo devices                       # stack MLX e llama, libreria e device
.venv/bin/rizzo decide examples/ticket.json --backend llama --quant q8_0
.venv/bin/rizzo serve --backend llama --quant q8_0 --port 8017
.venv/bin/rizzo evaluate benchmarks/smoke.jsonl --backend llama --gguf /path/Q8_0.gguf \
  --output results/llama-q8-vulkan-smoke/
```

`--backend auto` (default) scegle da solo in base all'**hardware presente**: Apple Silicon → MLX,
GPU NVIDIA → MLX-CUDA se l'extra `cuda` è installato *e usabile*, altrimenti llama.cpp; GPU AMD o
Intel → llama.cpp (Vulkan o HIP); senza GPU → llama.cpp su CPU. I nomi dei device sono quelli
dell'hardware (`nvidia`, `amd`, `intel`, `apple`, `gpu`) oppure dello stack (`mlx`, `vulkan`,
`hip`); un pin impossibile (`--device nvidia` su una macchina AMD) **fallisce**, non ripiega in
silenzio sulla CPU. `rizzo devices` stampa hardware, stack installati e la scelta di `auto`.

Un dettaglio che conta: `auto` guarda se lo stack ha un **acceleratore usabile**, non se il
pacchetto è importabile. `mlx-cpu` si importa ovunque, ma su una macchina AMD verrebbe scelto solo
per quello e costerebbe minuti per decisione: ora vince llama.cpp/Vulkan.

`--bits` vale solo per MLX: con il backend llama va scelto un GGUF già quantizzato (`--quant`), e
il CLI lo rifiuta con un messaggio esplicito. `--gguf` implica il backend llama.

## 5 · Dimensionamento del contesto (importante)

`--ctx` è il limite **per sequenza** (come in MLX), ma in llama.cpp il contesto è una
prenotazione reale e il branching del prefisso **duplica** la KV cache. Rizzo Flow quindi chiede
`n_ctx = ctx × (batch_size + 1)` e verifica che `n_ctx_seq ≥ ctx`; se llama.cpp concedesse meno
contesto per sequenza, il caricamento fallisce invece di troncare.

Costo pratico sul 4B: circa **36 KiB per token** di KV. Con i default (`--ctx 8192`,
`--batch-size 4`) la prenotazione è `n_ctx = 40960`, circa **1.4 GB**. Un batch più grande
aumenta la prenotazione; un `--ctx` più piccolo la riduce.

Una `llama_memory_seq_cp` verso una sequenza oltre `n_seq_max` **aborta il processo**: il backend
rifiuta la combinazione invece di lasciar morire llama.cpp (è successo durante lo sviluppo di
questo backend).

## 6 · Limiti noti

- **Niente proiezione selettiva.** MLX calcola solo le righe del vocabolario delle lettere
  ammesse; `llama_get_logits_ith` restituisce il vettore pieno e la selezione avviene in Python.
  È la prima ottimizzazione da misurare se serve.
- **Calibrazioni non portabili.** Il `fingerprint` include pesi GGUF, sha256, precisione, commit
  di llama.cpp, backend e `PROMPT_VERSION`: una calibrazione MLX non è valida su llama e
  viceversa. Comportamento voluto.
- **Picco di memoria non riportato.** L'API di llama.cpp non lo espone; il timing riporta
  `context_tokens` e `context_tokens_per_sequenza` invece di `peak_mlx_bytes`.
- **Parità cross-backend non misurata.** Lo smoke è stato eseguito anche su llama.cpp, ma su
  hardware e toolchain diversi da MLX/CUDA: non è un confronto di parità e non va presentato
  come tale.
- **Hashing all'avvio.** Ogni load calcola lo sha256 del GGUF (4.4 GB) per l'auditabilità: circa
  3–4 secondi in più sul caricamento.

## 7 · Numeri misurati (21 settembre 2026)

RX 7900 XTX (gfx1100), Vulkan, `Spark-X2.5-4B-Q8_0`, prompt v3, `--ctx 8192`, `--batch-size 4`.
Report: [`results/llama-q8-vulkan-validation/validation.json`](../results/llama-q8-vulkan-validation/validation.json).

| Misura | Valore |
| --- | ---: |
| Smoke (17 righe), accuracy | 0.95 |
| Smoke, NLL / Brier / ECE | 0.450 / 0.079 / 0.041 |
| Smoke, copertura / status accuracy | 0.80 / 0.95 |
| Throughput endpoint | 7.19 decisioni/s |
| Latenza p50 / p95 | 148 ms / 375 ms |
| Ticket: prefisso condiviso / batch | 186 token / 2 |
| Ticket: inferenza shared / direct | 0.327 s / 0.379 s |
| Ticket: caricamento (hash + modello) | ~3.7 s |

Riferimento, non confronto: la stessa fixture su MLX/CUDA (RTX 5060 Ti, Q8) dava accuracy 0.95 e
NLL 0.428. Hardware e runtime diversi.

## 8 · Verifica

```bash
RIZZO_LLAMA_TEST=1 .venv/bin/pytest -q -m llama        # 4 test con pesi reali
.venv/bin/pytest -q                                    # unit weight-free
RIZZO_LLAMA_LIB=... .venv/bin/python scripts/validate_llama.py \
  --gguf /path/Q8_0.gguf --output results/llama-q8-vulkan-validation
```

Lo script verifica tre prove locali del criterio "prompt v3 riprodotto": il template del GGUF
rende byte-identico quello della fixture HF, ogni lettera A–Z è un token singolo, e il nostro
tokenizer coincide con `llama-tokenize` della stessa build. Registra inoltre `prompt_sha256` e
`input_tokens` per ogni risposta, per confronti futuri.
