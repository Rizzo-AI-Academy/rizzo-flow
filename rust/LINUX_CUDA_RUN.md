# Run report — Rizzo Flow on Linux + NVIDIA (hardware the project does not have)

The project's README asks for a run on hardware it does not have, Linux included, with
`rizzo devices` and the timings. Here it is, together with something the numbers alone
would not show: **the prebuilt CUDA runtime does not start on this machine**, while the
same code on CPU does, and this port (which compiles its own llama.cpp against the local
toolkit) runs on the GPU.

## Machine

| | |
|---|---|
| OS | Linux 7.0.0-31-generic, x86-64 |
| GPU | NVIDIA GeForce RTX 2070, 8 GB (8 164 474 880 bytes) |
| Driver | 595.91.07 |
| CUDA toolkit | 12.0 (V12.0.140) |
| CPU | Intel Core i5-6500 (4 cores) |
| Python | 3.12.3 |
| Reference | `main` at `d34665b`, llama.cpp runtime `llama-b11081-linux-x64-cuda` |

## `rizzo devices`

Detects the GPU correctly and installs the CUDA package:

```
llama.cpp: release b11081, host linux/x64, recommended cuda, installed [cuda]
  CUDA0  NVIDIA GeForce RTX 2070  kind=gpu  backend=CUDA  total_bytes=8164474880
mlx: installed false
```

## GPU run — aborts at the first decode

```
$ rizzo decide rizzo_test_6.json --backend llama --device cuda --size 4b --quant q8_0 --ctx 8192

CUDA error: the provided PTX was compiled with an unsupported toolchain.
  current device: 0, in function ggml_cuda_kernel_can_use_pdl at ggml/src/ggml-cuda/common.cuh:1679
  cudaFuncGetAttributes(&attr, kernel)
ggml-cuda.cu:108: CUDA error
Aborted (core dumped)
```

**Diagnosis**: the CUDA kernels in the prebuilt `b11081` binary are ahead of this
machine's driver. PTX is JIT-compiled by the driver, and a driver older than the toolkit
the PTX was produced with cannot compile it. Nothing in the failure is about the model or
the request: it happens while loading the first kernel.

**Suggestion for `download`/`devices`**: the runtime selector could compare the installed
driver version against the toolkit each prebuilt was built with and either pick a matching
build or say so explicitly (today the failure surfaces as an abort at inference time).
The project already has the plumbing to know the driver: `devices` reports the GPU.

## Same code, same model, CPU — works

To isolate the runtime from the code, the same command on one question with `--device cpu`:

```
$ rizzo decide one_question.json --backend llama --device cpu --size 4b --quant q8_0 --ctx 4096 --threads 4

device: cpu · total 14.83 s · inference 14.81 s
tipo_s1 → NORMA (top probability 1.000)
```

Correct answer, exit 0. So on this machine the code path is sound and only the CUDA
prebuilt is unusable.

## For comparison: this port on the same GPU

`rizzo-flow-rs` builds its own llama.cpp against the local CUDA 12.0 toolkit, so it is not
affected by the prebuilt mismatch. Same machine, same GPU, 6-question bench:

| | |
|---|---|
| model | `Qwen3.5-4B-Instruct-SingleTurn` Q4_K_M (2.7 GB) |
| per decision (warm, single question) | **0.268 s** median, 0.28 s p95 |
| throughput | 3.7 decisions/s |
| full 6-question bench | ~1.1-1.4 s |
| resident VRAM | ~3.4 GB |

(Spark-X2.5 itself cannot be loaded by this port yet: the Rust binding pins a llama.cpp
older than the `spark2_5` architecture — see [REPORT.md](REPORT.md) §3.8 and §5.1.)

## Reproducing

```bash
# their project, CPU (works on this machine)
cd rizzo-flow-main && PYTHONPATH=src .venv/bin/python -m rizzo_flow.cli decide \
    one_question.json --backend llama --device cpu --size 4b --quant q8_0 --ctx 4096 --threads 4

# their project, GPU (aborts with the PTX error above)
… --device cuda
```
