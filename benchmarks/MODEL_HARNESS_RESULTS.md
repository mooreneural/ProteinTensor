# Real-model DataLoader throughput (single GPU)

The GPU consumer here is an actual small transformer (MSA + structure
encoder) running a real forward + backward step - not a synthetic matmul.
Two datasets over the same structures feed the same model: `traditional`
(parse mmCIF + A3M + compute distances per item) vs `ptt` (zero-parse).

- **GPU:** NVIDIA GeForce RTX 5080 | torch 2.11.0+cu128 | proteintensor 0.4.0
- **Model:** transformer encoder, dim 256, MSA depth 64; real forward + backward
- **Config:** 48 samples, batch 4 | 2026-07-09T18:35:39+00:00

`wait` = fraction of wall-clock the GPU sits idle waiting for the next batch.

| depth | params | GPU step | workers | trad /s | ptt /s | throughput | trad wait | ptt wait |
|---|---|---|---|---|---|---|---|---|
| 2 | 1.6M | 5.35 ms | 0 | 23.6 | 111.8 | **4.74x** | 0.963 | 0.83 |
| 2 | 1.6M | 5.35 ms | 4 | 78.9 | 263.1 | **3.35x** | 0.871 | 0.578 |
| 6 | 4.8M | 11.35 ms | 0 | 23.8 | 92.6 | **3.89x** | 0.92 | 0.679 |
| 6 | 4.8M | 11.35 ms | 4 | 77.4 | 191.1 | **2.47x** | 0.732 | 0.346 |
| 12 | 9.5M | 21.11 ms | 0 | 22.0 | 67.6 | **3.08x** | 0.867 | 0.562 |
| 12 | 9.5M | 21.11 ms | 4 | 68.3 | 111.3 | **1.63x** | 0.56 | 0.246 |

**Reading the table.** With a real model, the story matches the matmul
harness: `.ptt` wins most when the model step is cheap (data-bound) and the
throughput gap narrows as the model gets deeper (compute-bound) - but the
GPU-idle (`wait`) gap persists, and `num_workers>0` speeds up both loaders
without erasing it.

**Honest scope:** small model on a single GPU; production models (AlphaFold/
Boltz) have far heavier steps, where the throughput speedup converges toward
1x and the durable value is reclaimed GPU-idle time and fewer dataloader
workers. Multi-GPU DDP is not tested here.

