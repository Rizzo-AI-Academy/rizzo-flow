# Calibration pilot — does the calibration path actually work?

The unit tests cover `fit_temperature` in isolation, and the parity suite proves it is
numerically identical to the Python reference. Neither of them answers the operational
question: **can a calibration file be produced, loaded and applied by the engine?**
This pilot answers that, end to end, on real inference.

```bash
calibration/pilot/run_pilot.py          # needs the binary and the GGUF (see --help)
```

## What it does

1. runs the engine on 24 pilot questions (uncalibrated) and keeps the response;
2. builds `rows.jsonl` by pairing each answer's **own `option_logits`** with the
   ground-truth label, in candidate order (the same order documented in
   `parity/core_vectors.json`);
3. fits temperatures with the `calibrate` subcommand;
4. runs again **with** the calibration and checks that it is really applied;
5. negative test: a calibration whose fingerprint does not match must be refused.

## Result (22/09/2026, Qwen3.5-4B-Instruct-SingleTurn Q4_K_M, RTX 2070)

```
fingerprint                     e240e9f020916baa…
pilot rows                      24   (12 choice, 12 boolean)
accuracy, uncalibrated          24/24
accuracy, calibrated            24/24
temperatures applied            {boolean: 0.05, choice: 0.05}
probability_status              temperature_scaled_requires_held_out_validation
calibration block in response   yes
probabilities moved             24/24 answers (max shift 0.331)
wrong fingerprint refused       yes — "Calibration was fitted for a different
                                model/runtime/prompt configuration (expected …)"
```

**The path works.** Every step that the tests could not reach is now exercised on real
inference: the file is produced, bound to the model fingerprint, loaded by the engine,
the per-answer `temperature` is the fitted one, `probability_status` switches to the
scaled variant, the probabilities actually move, and a foreign calibration is rejected.

## What this pilot is NOT — read this before quoting the numbers

* **Not a domain calibration.** The 24 rows are hand-labelled *by construction* (each
  label follows unambiguously from the evidence pack). It is a single-domain pilot, not a
  dataset.
* **The fitted temperatures are degenerate, and that is the interesting part.** The fit
  chose T = 0.05, the lower bound of the grid, for both types, driving the in-sample NLL
  to ~1e-9. That is textbook overfitting: on 12 rows where the model is always right,
  sharpening to the boundary fits the sample and nothing else. It is exactly why the
  implementation bounds the grid and why the artefact carries
  `status: fitted_requires_held_out_validation` — the tool refuses to claim validation,
  and here it is right to. **Do not use this file as a production calibration.**
* **Accuracy is invariant under temperature scaling.** It is a monotone transform of the
  logits, so it can never change the argmax: 24/24 before and 24/24 after is expected and
  says nothing about the calibration. What changes is the *quality of the probabilities*
  (NLL / Brier / ECE), which is what calibration is for.
* **No held-out evaluation is possible here.** The fit requires ≥ 10 rows per type; with
  24 rows a train/test split would break that constraint. An honest evaluation needs a
  real labelled dataset — that is the next step, and it is declared in
  [REPORT.md](../../REPORT.md) §5.

## Artefacts

`results/` keeps what the run produced: `pilot_summary.json` (the numbers above),
`calibration.json` (the fitted artefact, with its fingerprint and fit metrics),
`rows.jsonl` (the labelled logit rows). The full uncalibrated/calibrated responses are
regenerable with one command and are not committed.
