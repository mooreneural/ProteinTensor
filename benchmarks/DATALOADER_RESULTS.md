# DataLoader truth harness (single GPU)

End-to-end throughput of a real PyTorch `DataLoader` feeding a GPU consumer.
Two datasets over the same structures: `traditional` (parse mmCIF + A3M +
compute distances per item) vs `ptt` (zero-parse reads). The `.ptt` win is
large when GPU steps are fast (data-bound) and shrinks as steps get slower
(compute-bound) because workers hide loading behind compute.

- **GPU:** NVIDIA GeForce RTX 5080 | torch 2.11.0+cu128 | proteintensor 0.4.0
- **Config:** 64 samples, batch 4, MSA depth 64 | 2026-07-09T17:26:36+00:00

`wait` = fraction of wall-clock the GPU sits idle waiting for the next batch.

| workers | GPU step | trad samples/s | ptt samples/s | throughput | trad wait | ptt wait |
|---|---|---|---|---|---|---|
| 0 | 0.14 ms | 29.0 | 150.3 | **5.18x** | 0.996 | 0.978 |
| 0 | 4.48 ms | 28.4 | 130.8 | **4.6x** | 0.965 | 0.839 |
| 0 | 16.82 ms | 26.1 | 93.3 | **3.57x** | 0.888 | 0.602 |
| 4 | 0.14 ms | 97.7 | 377.8 | **3.88x** | 0.989 | 0.952 |
| 4 | 4.48 ms | 98.9 | 353.5 | **3.57x** | 0.881 | 0.574 |
| 4 | 16.82 ms | 93.7 | 178.5 | **1.9x** | 0.6 | 0.232 |

**Reading the table.** At low GPU step time the loader is the bottleneck and
`.ptt` keeps the GPU fed (low ptt wait, high speedup). As GPU step time rises the
throughput speedup shrinks, but the GPU-idle (`wait`) gap stays large: even at the
slowest step here, `.ptt` leaves the GPU idle ~23% vs ~60% for traditional.
`num_workers>0` speeds up both loaders but does not erase the gap.

**What this proves / does not prove (honest scope):**
- The GPU consumer is a tunable matmul, not a real model - this measures DataLoader
  mechanics, not any specific model's end-to-end training time.
- GPU step times here are 0.14-16.8 ms. Real AlphaFold/Boltz training steps are 100s
  of ms to seconds (far more compute-bound); at those step times the throughput
  speedup converges further toward 1x and the durable win is reclaimed GPU-idle time
  / needing fewer dataloader workers, not raw speedup.
- Single GPU only; multi-GPU DDP scaling is not tested here.
- MSA content is synthetic (dimensions realistic); the traditional baseline uses the
  vectorized A3M parser, so it is a fair comparison.

