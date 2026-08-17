# TPU GEMM performance model

This repository predicts the warm runtime of one `jnp.dot` from a GEMM problem and a small TPU hardware specification. It retains the full evidence chain: deterministic shape families, a resumable benchmark, raw timing summaries and run provenance, historical ablations, the final physical model, evaluation, and every figure used in the write up.

The measured scope is bf16-to-f32 and int8-to-int32 GEMM on one TPU v5e or v6e chip. Compilation, transfers of operands from the host, fused epilogues, multi-chip collectives, and unmeasured operations are out of scope.

## Quick start

```bash
uv sync
uv run pytest
uv run perfmodel predict --sku v5e -M 4096 -N 4096 -K 4096 --dtype bf16
uv run perfmodel evaluate
uv run perfmodel ablate
uv run perfmodel figures
```

The public Python API produces one final prediction:

```python
from hardware import load_hardware
from model import GemmProblem, predict

runtime_s = predict(GemmProblem(4096, 4096, 4096, "bf16"), load_hardware("v5e"))
```

That two-argument call is spec-only. Evaluation supplies the measured launch floor through the optional `launch_overhead_s` argument, whose values comes from `data/runs.json` not the hardware spec.

## Repository map

| Path | Responsibility |
|---|---|
| `hardware.py`, `specs/` | Validated prediction inputs only |
| `model.py` | Final contraction-padded, accumulator-resident pipeline model |
| `shapes.py` | Seeded square, random, skinny, tile-probe, and ridge families |
| `benchmark.py` | TPU collection with warmup, synchronization, median/IQR, and resume |
| `evaluate.py` | Measurement loading, metrics, and result tables |
| `analysis.py` | Textbook/accounted baselines, diagnostic calibration, and padding ablation |
| `figures.py`, `figures/` | Plots made only from evaluator output |
| `data/measurements.csv` | 900 raw timing summaries in one narrow schema |
| `data/runs.json` | Exact collection environments and measured launch floors |
| `tests/` | Physical invariants, API consistency, and empirical accuracy bounds |

The final arithmetic accounting follows the tile-probe result:

```text
FLOPs = 2 * M * N * round_up(K, mxu_dim)
```

Stored arrays are padded independently to their physical memory layout. HBM traffic distinguishes an accumulator that can remain in VMEM from a GEMM that must reread inputs, and compute/memory overlap follows the output-block pipeline depth. The derivation, ablations, all-data descriptive results, and limitations are in the write up.

## Recollecting data

JAX is an optional pinned dependency

```bash
uv sync --extra tpu
uv run perfmodel shapes --sku v5e
uv run perfmodel benchmark --sku v5e \
  --output /tmp/measurements.csv --runs /tmp/runs.json
```

Operands are independently materialized. Each shape receives five warmups (the first compiles), then 20 individually synchronized samples. The harness writes after every completed shape and resumes by the `(sku, M, N, K, dtype)` key.
