# Rizzo Flow

Decisioni tipizzate e stime numeriche locali con **XHToken/Spark-X2.5-4B**.
Il modello valuta le opzioni; il software restituisce JSON verificabile senza generare token.

Implementazione indipendente ispirata al pattern di Jev e SemIf. Usa i pesi originali di Spark:
non è un modello addestrato da zero, non replica l'architettura proprietaria di Jev e non presume
di avere probabilità calibrate o qualità superiore a SemIf.

**Verificato con il checkpoint reale su M4 Pro:** 28 test superati, API funzionante,
circa 304 ms di mediana nello smoke Q8 e picco MLX di 4.88 GiB. Risultati, errori e limiti
sono documentati in [results/README.md](../results/README.md).

## Avvio

Scegli la riga adatta alla tua macchina (è l'unico passo che dipende dalla piattaforma):

```bash
uv sync --locked --extra mlx      # Mac con Apple Silicon (GPU Metal)
uv sync --locked --extra cuda     # Windows o Linux con GPU NVIDIA (driver CUDA 13)
uv sync --locked --extra cpu      # Windows o Linux senza GPU: molto lento, ultima spiaggia
source .venv/bin/activate         # macOS / Linux
.venv\Scripts\activate           # Windows
rizzo devices                     # backend rilevati: mlx, cuda, vulkan, hip, cpu
rizzo download
rizzo decide examples/ticket.json
rizzo decide examples/numeric.json --bits 8
rizzo serve --bits 8              # --device auto|mlx|cuda|vulkan|hip|cpu, default auto
```

Provato su Apple Silicon (tutti i risultati pubblicati) e su Windows 10 + RTX 5060 Ti. CUDA su
Linux non è stato provato; il backend CPU installa e passa i test unitari ma nell'unico tentativo
(i7-7700K, 1.7B a 8 bit) ha impiegato circa 3 minuti per 8 token.

**GPU AMD (e altre supportate da llama.cpp).** MLX non ha un backend AMD: per Radeon c'è un
secondo backend che parla direttamente con `libllama.so`. Non è un extra: si compila llama.cpp
una volta con Vulkan (o HIP) e si punta Rizzo Flow alla build. `spark2_5` è nel mainline di
llama.cpp dal [PR #27868](https://github.com/ggml-org/llama.cpp/pull/27868), quindi non serve un
fork; qui è pinnata la v0.4.1.

```bash
export RIZZO_LLAMA_LIB=/percorso/llama.cpp/build/bin
uv sync --locked --extra llama
rizzo download --format gguf --quant q8_0
rizzo decide examples/ticket.json --backend llama --quant q8_0
```

`--backend auto` sceglie llama.cpp se si passa `--gguf`, altrimenti MLX; `--bits` vale solo per
MLX (con llama.cpp si sceglie un GGUF già quantizzato). Provato su Linux + RX 7900 XTX via Vulkan
(Q8_0): smoke 0.95, 7.19 decisioni/s, p50 148 ms. Guida completa, dimensionamento del contesto
(`--ctx` è per-sequenza) e limiti: [llama-amd.md](llama-amd.md).

Il download richiede circa 8 GB. Il modello viene salvato in `models/Spark-X2.5-4B`.
BF16 è la precisione predefinita; `--bits 8` e `--bits 4` quantizzano i pesi in memoria.
Le piccole proiezioni dei gate di attenzione mantengono la precisione originale.
La quantizzazione modifica le probabilità: confrontare i risultati sul proprio carico.
I comandi vanno eseguiti dalla radice del progetto. È possibile indicare `--model /percorso/assoluto`.

L'API mantiene un solo modello residente. Interfaccia interattiva: <http://127.0.0.1:8017/docs>.
Gli schemi completi sono in `request.schema.json` e `response.schema.json`; il server valida
sia le richieste sia le risposte. `rizzo schema --response` esporta lo schema dell'output.

```bash
curl http://127.0.0.1:8017/v1/decisions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/ticket.json
```

`GET /health` restituisce stato e provenienza del modello. Il server ascolta soltanto su localhost
per impostazione predefinita; la distribuzione pubblica non è inclusa.

## Playground

Con il server avviato: <http://127.0.0.1:8017/playground>. Builder di domande noul/choice/score,
esempi pronti, editor JSON grezzo per entrambi gli endpoint, barre di probabilità, tempi
(round-trip, inferenza, prefill), token dello state in cache, numero di microbatch e comando cURL.
Pagina singola senza dipendenze esterne, servita dallo stesso processo.

## Demo Snake

Con il server avviato: <http://127.0.0.1:8017/snake>. Ogni mossa del serpente è una richiesta
`POST /v1/decisions` (una domanda `choice` con le mosse legali); le probabilità del modello
compaiono in tempo reale sulle celle candidate, con barre, logit, tempi e registro delle
decisioni. Zero token generati. **Registra GIF** cattura griglia e pannello delle decisioni
direttamente nella pagina (encoder GIF89a scritto a mano, nessuna dipendenza) e, quando la fermi,
salva il file nei download.

Si può scegliere cosa vede il modello. Con i *sensori per mossa* (contenuto della cella, distanza
dal cibo, celle libere raggiungibili: calcolati dal gioco, la scelta è del modello) Q8 su M4 Pro
gioca a circa 2 mosse/s (≈ 490 ms a decisione, ~300 token): in tre partite informali 10×10 ha
mangiato 12 e 7 cibi in 80 mosse senza morire e 22 cibi in 208 mosse prima di chiudersi senza
mosse sicure. Con la *sola griglia ASCII* è morto entro 26 e 13 mosse con 0 punti in due partite.
Sono poche partite, non un benchmark. Le opzioni vengono mescolate a ogni mossa per attenuare il
bias di posizione; la rete di sicurezza è facoltativa, spenta di default e segnata nel registro.

## API compatibile con TypeSafe

`POST /v1/systemone` e `GET /v1/models` seguono la forma pubblica documentata in
<https://docs.typesafe.ai/api>: stesso corpo (`state`, `model`, `questions`), stessi tipi
(`noul`, `choice`, `score` con `instructions` e `criteria`), stessa risposta (`model`, `answers`, `usage`).

```bash
curl http://127.0.0.1:8017/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "rizzo-latest",
    "questions": {
      "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
      "department": {"type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages", "sales": null}},
      "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]}
    }
  }'
```

Un client scritto per l'API ospitata può puntare qui cambiando soltanto l'URL di base
(per gli SDK ufficiali: `TYPESAFE_BASE_URL=http://127.0.0.1:8017`). **Compatibile è l'interfaccia, non il modello:**

- `model` accetta `rizzo-latest`, l'ID locale (es. `rizzo-spark-x2.5-4b-q8`) e qualunque nome `jev-*`
  come alias di comodo. La risposta riporta sempre l'ID locale: nessuna risposta si presenta come Jev.
- Le domande sono tradotte nelle primitive native con `allow_abstain: false`, perché il formato
  non prevede astensione. `noul` è la probabilità di "sì" tra due sole opzioni.
- `confidence = (n × p_max − 1) / (n − 1)`, la statistica mostrata nella pagina Confidence di TypeSafe.
  La formula esatta di Jev non è pubblica. Descrive la forma della distribuzione: non è calibrata.
  Sul checkpoint Spark le distribuzioni sono spesso molto concentrate (0.9999): senza temperature
  scaling sul proprio dominio le soglie di confidence pensate per Jev non sono trasferibili.
- Limiti locali: 26 opzioni per `choice` (una per lettera; Jev: 255), 10 livelli per `score`, 64 domande,
  8000 caratteri per `instructions` e per ogni descrizione. `instructions`/`criteria` strutturati
  vengono serializzati come JSON canonico.
- `usage.input_tokens` conta lo state una volta sola più i suffissi; `output_tokens` è sempre 0.
- `x_rizzo` (tempi, fingerprint, stato delle probabilità) è un'estensione fuori dal contratto.
- Autenticazione Bearer come l'originale, attiva solo se è impostata `RIZZO_API_KEY`
  (altrimenti l'header è ignorato). Errori: 401 e 422. Nessun rate limit, quindi niente 429/529.

`/v1/decisions` resta l'API nativa completa: `numeric`, astensione, policy, logit e statistiche.

## Le quattro primitive

| Tipo | Input specifico | Output principale |
| --- | --- | --- |
| `boolean` | Descrizioni opzionali di vero e falso | `value` booleano e probabilità di vero condizionata alla disponibilità |
| `choice` | `options` con ID e descrizioni | `choice`, probabilità di ciascuna opzione |
| `score` | `levels` ordinati, da basso ad alto | Media dei livelli, `normalized_score`, dispersione |
| `numeric` | `anchors` crescenti con valore e descrizione, `unit` | Media delle ancore, mediana, dispersione, probabilità fuori scala |

Ogni richiesta contiene uno `state` (testo, oggetto o array) e un dizionario `questions`.
Gli ID delle domande non sono inviati al modello. Le descrizioni devono essere autonome.
Massimo 64 domande per richiesta. Ogni candidato, speciali inclusi, è una lettera maiuscola a token
singolo verificata con il tokenizer ufficiale: 26 slot in tutto. Quindi 26 opzioni o livelli con
`allow_abstain: false`, 25 con l'astensione (default); per `numeric` 24 ancore, 23 con l'astensione.

```json
{
  "state": {"measurement": 75, "unit": "percent"},
  "questions": {
    "fill": {
      "type": "numeric",
      "instructions": "Read the reported fill percentage.",
      "unit": "percent",
      "anchors": [
        {"value": 0, "description": "Empty"},
        {"value": 50, "description": "Half full"},
        {"value": 75, "description": "Three quarters full"},
        {"value": 100, "description": "Completely full"}
      ]
    }
  }
}
```

Esempi completi: `examples/ticket.json`, `examples/numeric.json`, `examples/house.json`.
I comparabili immobiliari dell'ultimo esempio sono **sintetici**, non quotazioni di mercato.

## Cosa significa un numero

`score = Σ indice × probabilità`; `numeric.value = Σ valore_ancora × probabilità`.
Quando esistono opzioni di astensione o fuori scala, queste medie sono condizionate alla
distribuzione sulle sole opzioni valide. La risposta espone anche la distribuzione originale
completa e la massa di probabilità esclusa, senza nasconderla.

Le ancore sono valori rappresentativi, **non intervalli statistici**. Se si vogliono usare fasce,
bisogna definirle in modo non sovrapposto nelle descrizioni e scegliere valori rappresentativi
appropriati. I quantili restituiti sono quantili della distribuzione discreta sulle ancore:
non sono intervalli di confidenza o garanzie di copertura sul valore reale.
Anche `stddev` misura soltanto la dispersione tra ancore: non include la variabilità interna a una fascia.
Molti decimali non equivalgono ad alta precisione; la media rimane entro le ancore estreme.

`numeric` aggiunge sempre le opzioni `__below_range__` e `__above_range__`.
Tutti i tipi aggiungono `__insufficient__` per impostazione predefinita.
Quando prevale un'opzione non utilizzabile, o la loro massa complessiva raggiunge il limite,
il risultato principale è `null`, con `status` esplicito:

- `ok`: risultato disponibile secondo la policy;
- `insufficient_evidence`: evidenza insufficiente;
- `out_of_range`: valore fuori dal supporto numerico;
- `uncertain`: probabilità massima sotto la soglia richiesta.

Ogni domanda può includere:

```json
"policy": {
  "allow_abstain": true,
  "max_unavailable_probability": 0.5,
  "min_top_probability": 0.0
}
```

Queste soglie sono regole operative configurabili, non soglie universali di affidabilità.
Un `status: ok` non garantisce correttezza. L'opzione "non so" può anch'essa essere scelta male.
`concentration`, entropia e probabilità massima descrivono la distribuzione, non la probabilità
che la decisione sia corretta. L'API nativa non espone alcuna `confidence`; quella dell'endpoint compatibile è dichiarata non calibrata.

## Come risparmia lavoro

1. Compila ogni domanda in una scelta tra token singoli, disabilitando il thinking nel template Spark.
2. Esegue il prefill dello stato comune una volta, a blocchi di 512 token.
3. Clona le cache native di attenzione completa e sliding-window per le domande indipendenti.
4. Raggruppa i suffissi per lunghezza in microbatch (default 4), con padding causale a destra.
5. Legge l'ultima posizione reale di ogni domanda e proietta soltanto le righe di vocabolario
   delle risposte ammesse, anche con pesi quantizzati.
6. Converte i logit in distribuzioni e output tipizzati in Python.

Nessun ciclo di generazione, parsing del testo prodotto o riparazione JSON. Restano necessari
prefill e calcolo dei suffissi: "zero token generati" non significa latenza zero.
Il numero di domande, la loro lunghezza e la dimensione del batch incidono sulla latenza.
La cache dello stato viene scartata alla fine della richiesta; non conserva evidenza tra utenti.
Le richieste concorrenti condividono un lock per evitare picchi di memoria e interferenze GPU.

`mode: "direct"` nella richiesta disattiva la condivisione come riferimento di verifica.
`--batch-size 1` riduce la memoria e mantiene il riuso del prefisso. `--ctx` (prima `--max-tokens`) cambia
il limite per domanda (default 8192); input oltre il limite vengono rifiutati, mai troncati.
Il limite di cache MLX inattiva è 256 MiB: non limita memoria dei pesi o cache KV attive.

## Misurazioni e calibrazione

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/rizzo evaluate benchmarks/smoke.jsonl --compare-modes --output results/my-smoke.json
.venv/bin/rizzo evaluate benchmarks/perturbations.jsonl --output results/my-perturbations.json
.venv/bin/python scripts/validate_checkpoint.py --bits 8 --output results/my-q8-validation
```

Per confrontare due report per ID semantico:

```bash
.venv/bin/python scripts/compare_reports.py results/my-smoke.json results/my-perturbations.json --perturbations
```

I test MLX usano l'architettura Spark reale con pesi casuali piccoli e verificano la proiezione
selettiva contro l'intero vocabolario, isolamento delle cache, confini sliding-window, batch con
lunghezze diverse e quantizzazione. Il validatore usa invece i pesi 4B reali, API, fixture e stato lungo.

Gli output sono **create-only**. I report conservano distribuzioni, logit, hash dei prompt,
hash dei pesi/tokenizer, revisioni, tempi sincronizzati GPU e memoria MLX. Il caricamento
e il warmup sono esclusi dai benchmark; tempi di compilazione e inferenza sono separati.
La memoria MLX non coincide con l'intera memoria del processo.

Le fixture incluse sono un piccolo smoke test scritto per il progetto, non un benchmark
indipendente né una prova di superiorità su Jev o SemIf. Le perturbazioni verificano cambi
di ordine e contesto irrilevante; vanno confrontate per ID semantico. Per il proprio dominio
servono dati con risposte note, un insieme di calibrazione separato e un test mai usato nel fitting.
La cronologia delle correzioni alle fixture è in `benchmarks/README.md`.

Temperature scaling è disponibile per ogni primitiva. Formato delle righe JSONL:

```json
{"type":"choice","logits":[1.0,3.0,-2.0],"label_index":1}
```

Usare i logit di **tutte** le opzioni, comprese quelle speciali, nello stesso ordine di `option_logits`.
Minimo 10 righe per tipo, solo come guardia tecnica: non basta per garantire qualità statistica.

```bash
.venv/bin/rizzo calibrate calibration.jsonl --fingerprint HASH_DEL_MODELLO --output calibration-fit.json
.venv/bin/rizzo evaluate held-out.jsonl --calibration calibration-fit.json --output results/held-out.json
```

Il fingerprint lega l'artefatto a pesi, tokenizer, precisione, runtime e versione del prompt.
Il fitting riporta NLL sul campione di calibrazione e non si dichiara validato: i report di test
includono accuracy, NLL, Brier, ECE, coverage, errori numerici sulle sole risposte disponibili
e differenze tra esecuzione diretta e condivisa. Una temperatura unica per tipo non garantisce
trasferimento tra domini o rubriche. Valutare anche il costo degli errori e dell'astensione.

## Provenienza

- [Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B), revisione `0bcb35678590218655dff3765b9e61c83b35e9c4`.
- [Runtime Spark MLX ufficiale](https://github.com/XHToken/Spark-MLX-LLM), commit `de2b4379fa1e2f2e1f99d84c83f0e008f651d86c`.
- MLX `0.32.2`, MLX-LM `0.31.3`; dipendenze transitive fissate in `uv.lock`.
- [Score di TypeSafe](https://docs.typesafe.ai/primitives/score) per la semantica della rubrica.

Il runtime usa la propria implementazione Spark: non carica codice Python arbitrario dalla
directory dei pesi e non sostituisce Spark con Qwen. Modello e runtime conservano le loro licenze Apache-2.0.
Il codice applicativo di questo progetto è originale; non sono stati copiati file di SemIf.
