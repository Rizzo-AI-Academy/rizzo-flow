# Risultati locali — 21 settembre 2026

Hardware: **Apple M4 Pro, 24 GiB** di memoria unificata. Modello originale Spark-X2.5-4B,
revisione e hash dei file registrati in ogni risposta. Runtime e dipendenze sono fissati.
Tutti i tempi sotto escludono caricamento del modello e warmup; includono compilazione
della richiesta e inferenza GPU sincronizzata. Non descrivono un servizio remoto.

## Esecuzioni correnti

| Misura | BF16 | 8 bit |
| --- | ---: | ---: |
| Mediana su 17 richieste smoke | 389 ms | 304 ms |
| p95 sulle stesse richieste | 1.256 s | 1.506 s |
| Picco allocazione MLX nello smoke | 8.38 GiB | 4.88 GiB |
| Argmax corretto, 20 decisioni smoke | 17/20 | 18/20 |
| Copertura: decisioni con status `ok` | 17/20 | 17/20 |
| Argmax corretto sulle 9 perturbazioni | 8/9 | 9/9 |
| Cambi argmax shared/direct nello smoke | 1/20 | 0/20 |

Report correnti: [BF16](spark-bf16-final/summary.json), [8 bit](spark-q8-final/summary.json).
Ogni directory contiene la risposta reale all'esempio API, tutte le risposte del benchmark,
logit, probabilità, tempi e differenze tra esecuzione diretta e condivisa.

Sono fixture di sviluppo piccole e semplici, usate anche durante la revisione del prompt.
**I dati non dimostrano che Q8 sia più accurato in generale, né superiorità su Jev o SemIf.**
Il p95 risente della richiesta con quattro domande, più lunga delle altre. Le ripetizioni
sono limitate; confronti temporali più forti richiedono un protocollo dedicato e più misure.
Le misure di memoria sono del solo allocatore MLX, non di tutto il processo macOS.

## Limiti osservati

- Anche con Q8, il modello risponde "no" anziché "dati insufficienti" in un caso di pagamento
  non registrato, e seleziona un'ancora interna anziché "sopra scala" per un prezzo esplicito di 900 EUR
  con ancore 100/200/300. Le opzioni di astensione e fuori scala sono disponibili e il codice le
  tratta correttamente, ma il modello può non selezionarle quando dovrebbe.
- BF16 sbaglia inoltre un booleano in italiano con probabilità molto vicine, circa 0.52 contro 0.48.
  Il confronto direct/shared può cambiare queste decisioni vicine: non è equivalenza bit per bit.
- La quantizzazione modifica 1 dei 20 argmax dello smoke; massimo spostamento di probabilità 0.1263.
  [Confronto di precisione](precision-comparison.json).
- Una perturbazione Q8 cambia scelta e status rispetto all'originale, anche se risulta corretta
  nella variante. Massimo spostamento di probabilità 0.7137. Il contesto irrilevante non è sempre innocuo.
  [Confronto per ID semantico](q8-stability.json).
- Nessuna probabilità è stata calibrata sui dati del dominio dell'utente. Il temperature fitting
  è implementato e testato, ma richiede un insieme di calibrazione e una verifica separata.

## Riuso di uno stato lungo

Misura con prompt v2, quattro domande e contesto oltre la finestra locale di 512 token:

| Precisione | Shared, mediana di 2 | Direct, 1 misura | Rapporto direct/shared |
| --- | ---: | ---: | ---: |
| BF16 | 3.123 s | 8.601 s | 2.75× |
| 8 bit | 3.482 s | 9.841 s | 2.83× |

Nessun argmax è cambiato su queste quattro domande. Probabilità comunque non identiche.
Questa misura appartiene ai report `spark-bf16-v2-validation/long-state.json` e
`spark-q8-v2-validation/long-state.json`, precedenti all'aggiunta della validazione formale
della risposta JSON e alla correzione di una diversa domanda nello smoke. Il codice del
calcolo MLX e il prompt v2 non sono cambiati. Qui Q8 risparmia memoria ma non tempo.

## Controlli eseguiti

- **24 test superati**, inclusi veri calcoli con l'architettura Spark ridotta e pesi casuali.
- Proiezione selettiva confrontata con il vocabolario completo in BF16, Q4 e Q8 nei test piccoli.
- Sul checkpoint 4B, delta massimo dei logit pari a **0** nella domanda usata per il controllo
  della proiezione, sia BF16 sia Q8.
- Cache sliding-window oltre il confine, isolamento delle copie, padding, riordino dei batch,
  ripetizioni, numeri non finiti, input invalidi, policy, fitting della temperatura e API.
- API provata anche con il checkpoint 4B reale e validazione dello schema delle risposte.
- Ruff passa. Due deprecation warning provengono dalle dipendenze di test FastAPI/Starlette;
  non sono fallimenti. Il tokenizer emette anche un avviso sulla configurazione custom Spark:
  l'inferenza usa esplicitamente l'implementazione MLX ufficiale, non AutoModel.

Q4 è disponibile e verificato nei test dell'architettura ridotta; non è stato eseguito
un benchmark di qualità del checkpoint 4B quantizzato a 4 bit.

## Storia ed evidenza

`spark-bf16-validation` conserva l'esperimento con prompt v1 e una copia dei sorgenti
di quell'esperimento in `source/`. `spark-*-v2-validation` conserva il prompt v2 con le
fixture iniziali. La domanda ambigua relativa al pagamento è documentata in
`../benchmarks/README.md`; le fixture originali sono conservate come `*-v1.jsonl`.
I report originali non sono stati riscritti e non vanno usati come risultato corrente.

`SHA256SUMS` verifica l'integrità dei report e delle copie storiche dei sorgenti:

```bash
cd results
shasum -a 256 -c SHA256SUMS
```

## Confronto con SemIf (in corso)

`scripts/semif_compare.py` esegue le fixture di SemIf (`authored144`, `perturbations108`,
`shape777`, commit `ca3ba65`) con le metriche di SemIf (`benchmarks/evaluate.py`) e lo stesso
perimetro di tempo: modello caldo; prompt, tokenizzazione, forward e readout inclusi;
caricamento e scrittura file esclusi. Ogni sistema usa il proprio prompt e il proprio modello.

Rizzo Flow, Spark-X2.5-4B **Q8**, M4 Pro 24 GiB — [report](semif-compare/rizzo-q8/report.json):

| Misura | Rizzo Q8 (M4 Pro) | SemIf Qwen3.5-4B Q8 pubblicato (M5 Max) |
| --- | ---: | ---: |
| authored144, balanced accuracy media per famiglia | 0.758 | 0.819 |
| perturbations108, stessa metrica | 0.706 | 0.766 |
| Latenza per decisione, stato corto (p50 / p95) | 254 / 259 ms | non confrontabile |
| shape777 shared, 37 stati × 21 criteri (~2k token) | 3.92 decisioni/s, 5.33 s per stato | non confrontabile |
| shape777 direct, 3 stati | 0.31 decisioni/s | non confrontabile |
| Cambi argmax shared/direct su 63 decisioni | 0 (max Δp 0.057) | — |

I valori SemIf vengono dal suo `results/mlx/2026-09-17-q8-fixed/summary.json`, misurati su un
altro Mac: **valgono per la qualità, non per i tempi**. Il confronto dei tempi richiede di
eseguire SemIf su questa macchina (`--system semif`), non ancora fatto. La famiglia più debole
di Rizzo è `rule_application` (0.689; 0.481 sulle perturbazioni, NLL 1.83: errori molto sicuri).
Le fixture non hanno etichette adjudicate da umani (dichiarato da SemIf) e sono piccole.

### Prompt v3 su Windows/CUDA (21 settembre 2026, RTX 5060 Ti 16 GB)

Stesse fixture e stesso `evaluate.py`; in più le metriche di stabilità di SemIf
(`evaluate_perturbations.py`, riprodotte in `semif_compare.stability`: sulle predizioni pubblicate
da SemIf la funzione ridà esattamente i loro 0.723 e 10/9/4 flip). Report:
[Q8](semif-compare/rizzo-q8-v3-cuda/report.json) (tutto, 777 decisioni anche in direct),
[BF16](semif-compare/rizzo-bf16-v3-cuda-full/report.json) (tutto; il primo run
[sola qualità](semif-compare/rizzo-bf16-v3-cuda/report.json) dà numeri identici).

| Misura | Rizzo v3 Q8 | Rizzo v3 BF16 | SemIf Q8 (MLX, pubbl.) | SemIf BF16 (3090, pubbl.) |
| --- | ---: | ---: | ---: | ---: |
| authored144, balanced accuracy media per famiglia | 0.829 | 0.819 | 0.819 | 0.813 |
| — solo metà **held-out** (72 righe) | 0.824 | 0.806 | 0.811 | 0.802 |
| perturbations108 | 0.865 | 0.842 | 0.766 | 0.766 |
| — solo metà **held-out** (54 righe) | 0.875 | 0.843 | 0.824 | 0.824 |
| 36 originali | 0.870 | 0.870 | 0.723 | 0.723 |
| option_reversal: accuratezza / flip | 0.889 / 4 | 0.870 / 4 | 0.813 / 9 | 0.813 / 10 |
| criterion_wrapper: accuratezza / flip | 0.833 / 2 | 0.815 / 3 | 0.682 / 7 | 0.706 / 9 |
| irrelevant_context: accuratezza / flip | 0.874 / 3 | 0.841 / 4 | 0.802 / 4 | 0.821 / 4 |
| evidenza mancante (36): accuratezza | 0.778 | 0.750 | 0.861 | 0.861 |
| — scelte ≠ `insufficient` con p ≥ 0.8 | **6** | **6** | 1 | 1 |

Tempi (non confrontabili con SemIf: hardware diverso; SemIf su RTX 3090, BF16: 2.33 fresh /
20.03 parallel dec/s):

| Misura (4B) | Q8 | BF16 |
| --- | ---: | ---: |
| Latenza stato corto p50 / p95 | 87 / 94 ms | 76 / 78 ms |
| shape777 shared | 7.52 dec/s, 1.76 s/stato, 103 s | **15.99 dec/s**, 1.31 s/stato, 49 s |
| shape777 direct | 1.65 dec/s, 12.7 s/stato, 472 s | 1.97 dec/s, 10.7 s/stato, 395 s |
| shared / direct | 4.6× | 8.1× |
| Cambi argmax shared/direct su 777 | 2 (max Δp 0.144) | 2 (max Δp 0.100) |
| Picco MLX | 6.55 GiB | 10.13 GiB |

Su questa GPU **BF16 è più veloce di Q8** (2.1× nei microbatch shared): i kernel quantizzati di
MLX-CUDA costano più della matmul BF16; Q8 conviene solo per la memoria (−3.6 GiB).

**Spark-X2.5-1.7B**, stesse prove e stessa GPU —
[Q8](semif-compare/rizzo-1.7b-q8-v3-cuda/report.json) (completo),
[BF16](semif-compare/rizzo-1.7b-bf16-v3-cuda-full/report.json) (completo):

| Misura | 1.7B Q8 | 1.7B BF16 | 4B Q8 (sopra) |
| --- | ---: | ---: | ---: |
| authored144 | 0.700 | 0.683 | 0.829 |
| — metà held-out | 0.697 | 0.690 | 0.824 |
| perturbations108 | 0.633 | 0.646 | 0.865 |
| — metà held-out | 0.514 | 0.532 | 0.875 |
| 36 originali | 0.628 | 0.628 | 0.870 |
| option_reversal: accuratezza / flip | 0.596 / **17** | 0.596 / **18** | 0.889 / 4 |
| criterion_wrapper: accuratezza / flip | 0.633 / 6 | 0.670 / 5 | 0.833 / 2 |
| irrelevant_context: accuratezza / flip | 0.670 / 6 | 0.670 / 4 | 0.874 / 3 |
| evidenza mancante: accuratezza / scelte sicure sbagliate | 0.833 / 2 | 0.778 / 2 | 0.778 / 6 |
| Latenza stato corto p50 / p95 | 40 / 44 ms | 37 / 41 ms | 87 / 94 ms |
| shape777 shared | 20.57 dec/s, 0.99 s/stato | 26.12 dec/s, 0.79 s/stato | 7.52 dec/s |
| shape777 direct | 3.72 dec/s | 4.37 dec/s | 1.65 dec/s |
| Cambi argmax shared/direct su 777 | 12 (max Δp 0.142) | 22 (max Δp 0.143) | 2 |
| Picco MLX | 2.82 GiB | 4.31 GiB | 6.55 GiB |

Il 1.7B è ~2.2–2.7× più veloce ma nettamente meno accurato: differenza appaiata dal 4B su
authored144 −0.128, intervallo 95% [−0.211, −0.046]. Forte bias di posizione (17 flip su 36
invertendo le opzioni, movimento medio di probabilità 0.43) e `rule_application` perturbata a
livello del caso (0.296, NLL 3.59: errori molto sicuri). Il dato migliore sull'evidenza mancante
non è un pregio: sceglie `insufficient` 52 volte su 144 contro 36 attese, cioè si astiene troppo
(coerente con quanto visto sullo smoke). Sta tra Qwen3-0.6B (0.440) e MiniCPM5-2B (0.686 / 0.693)
della scala pubblicata da SemIf, con lo stesso limite: prompt e modelli diversi.

Come leggerli:

- **Metà delle righe è lo split dev con cui è stato scelto il prompt v3**: il totale è ottimistico.
  La metà held-out, guardata qui per la prima e unica volta, conferma (0.824 / 0.875; il dev
  coincide al millesimo con il prompt-lab fatto sul Mac: 0.827 / 0.852).
- **Su authored144 Rizzo v3 e SemIf sono pari**: differenza appaiata con il bootstrap di SemIf
  +0.010, intervallo 95% [−0.051, +0.076]. Nessuna superiorità dimostrata. Il vantaggio sulle
  perturbazioni è più ampio ma poggia su 108 righe derivate dalle stesse 36 originali.
- **Peggio di SemIf sull'evidenza mancante**: 6 casi su 36 in cui Rizzo sceglie con p ≥ 0.8 una
  risposta quando quella giusta è `insufficient` (SemIf: 1). Le probabilità non sono calibrate.
- `rule_application` resta la famiglia debole sotto perturbazione (0.630 Q8, 0.593 BF16; NLL 1.63).
- Non eseguiti: WANLI ed Every (richiedono il download delle sorgenti), sottoinsieme TypeSafe (non
  ridistribuibile), confronto con generazione JSON e riuso seriale del prefisso (Rizzo non ha
  quei percorsi), SemIf sulla stessa GPU.

## Backend llama.cpp su GPU AMD (`llama-q8-vulkan-validation/`)

Primo run del backend llama.cpp (21 settembre 2026): RX 7900 XTX, Vulkan, `Spark-X2.5-4B-Q8_0`
di `stornic56` (revision `7ce72e5c…`, sha256 `092a263d…`), prompt v3, `--ctx 8192`,
`--batch-size 4`. Report: [`validation.json`](llama-q8-vulkan-validation/validation.json),
runbook: [`../docs/llama-amd.md`](../docs/llama-amd.md).

| Misura | Valore |
| --- | ---: |
| Smoke (17 righe), accuracy | 0.95 |
| Smoke, NLL / Brier / ECE | 0.450 / 0.079 / 0.041 |
| Smoke, copertura / status accuracy | 0.80 / 0.95 |
| Throughput / latenza p50 / p95 | 7.19 dec/s / 148 ms / 375 ms |
| Ticket: prefisso / batch / shared | 186 token / 2 / 0.327 s |
| Ticket: direct (riferimento) | 0.379 s / 8 batch, stesso argmax |

**Non è un confronto di parità cross-backend.** I numeri MLX/CUDA qui sopra vengono da hardware e
runtime diversi: la coincidenza sullo smoke (0.95) è un controllo di sanità, non una prova di
equivalenza. Le prove locali registrate nel report sono: template del GGUF identico a quello HF,
lettere A–Z token singoli, tokenizer uguale a `llama-tokenize` della stessa build. `prompt_sha256`
e `input_tokens` sono salvati per riga per rendere possibile un confronto vero in futuro.

