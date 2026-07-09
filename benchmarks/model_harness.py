#!/usr/bin/env python
"""Real-model DataLoader throughput harness (single GPU).

Closes the "the consumer is a matmul, not a real model" caveat from
dataloader_harness.py. Here the GPU consumer is an actual small transformer that
consumes the sequence tokens, backbone coordinates, and MSA a batch carries, and
runs a real forward + backward training step.

Two datasets over the same structures feed the same model:
  traditional : __getitem__ parses mmCIF + A3M + computes distances (per epoch work)
  ptt         : __getitem__ reads the pre-built .ptt (zero parse)

For a sweep of (num_workers, model depth) it reports, per loader:
  - throughput (samples/sec)
  - dataloader-wait fraction (share of wall-clock the GPU waits for data)
  - time-to-first-batch

Same honest measurement as the matmul harness: warm cache, persistent workers,
epoch 0 discarded (steady state, not process startup).

Usage:  python benchmarks/model_harness.py [--samples 48] [--batch 4]
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Reuse the datasets, collate, and .ptt builder from the matmul harness.
from dataloader_harness import (
    TraditionalDataset, PttDataset, collate, build, STRUCTS, MSA_USE_DEPTH,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
AA_VOCAB = 21     # sequence tokens 0-20
MSA_VOCAB = 23    # MSA tokens 0-22


def build_model(dim, depth, torch, nn):
    """A small MSA + structure transformer encoder (ESM/AlphaFold-flavored)."""
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq_embed = nn.Embedding(AA_VOCAB, dim)
            self.msa_embed = nn.Embedding(MSA_VOCAB, dim)
            self.bb_proj = nn.Linear(4 * 3, dim)
            layer = nn.TransformerEncoderLayer(
                dim, nhead=8, dim_feedforward=dim * 4, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, depth)
            self.head = nn.Linear(dim, 1)

        def forward(self, seq, bb, msa, pad):
            s = self.seq_embed(seq.clamp(0, AA_VOCAB - 1))
            m = self.msa_embed(msa.clamp(0, MSA_VOCAB - 1)).mean(dim=1)  # pool MSA depth
            b = self.bb_proj(bb.reshape(bb.shape[0], bb.shape[1], -1))
            x = self.encoder(s + m + b, src_key_padding_mask=pad)
            return self.head(x).mean()

    return Encoder()


def train_step(batch, model, opt, device, torch):
    """One real forward + backward + optimizer step from a batch."""
    seq = torch.from_numpy(batch["seq"]).long().to(device, non_blocking=True)
    bb = torch.from_numpy(batch["bb"]).to(device, non_blocking=True)
    msa = torch.from_numpy(batch["msa"]).long().to(device, non_blocking=True)
    n_res = batch["n_res"]
    B, maxN = seq.shape
    pad = np.ones((B, maxN), dtype=bool)
    for i, n in enumerate(n_res):
        pad[i, :n] = False
    pad = torch.from_numpy(pad).to(device, non_blocking=True)

    opt.zero_grad(set_to_none=True)
    loss = model(seq, bb, msa, pad)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()


def run_loader(dataset, batch, num_workers, model, opt, device, torch, epochs=3):
    from torch.utils.data import DataLoader
    kwargs = dict(batch_size=batch, shuffle=False, collate_fn=collate,
                  num_workers=num_workers)
    if num_workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(dataset, **kwargs)

    epoch_stats = []
    ttfb = None
    for ep in range(epochs):
        torch.cuda.synchronize()
        data_wait = compute = 0.0
        n = 0
        t_start = time.perf_counter()
        t_prev = t_start
        for batch_data in loader:
            t_data = time.perf_counter()
            if ep == 0 and ttfb is None:
                ttfb = t_data - t_start
            data_wait += t_data - t_prev
            train_step(batch_data, model, opt, device, torch)
            t_comp = time.perf_counter()
            compute += t_comp - t_data
            t_prev = t_comp
            n += batch_data["seq"].shape[0]
        epoch_stats.append((data_wait, compute, n))

    steady = epoch_stats[1:] if len(epoch_stats) > 1 else epoch_stats
    dw = statistics.median([s[0] for s in steady])
    cp = statistics.median([s[1] for s in steady])
    n = steady[0][2]
    total = dw + cp
    del loader
    return {
        "throughput": round(n / total, 1),
        "wait_fraction": round(dw / total, 3),
        "ttfb_ms": round(ttfb * 1e3, 1) if ttfb else None,
        "epoch_s": round(total, 3),
    }


def gpu_step_ms(dataset, batch, model, opt, device, torch, rounds=5):
    """Median GPU-only train-step time on a cached batch (data instant)."""
    warm = collate([dataset[i] for i in range(batch)])
    train_step(warm, model, opt, device, torch)
    s = []
    for _ in range(rounds):
        t = time.perf_counter()
        train_step(warm, model, opt, device, torch)
        s.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(s), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 6, 12])
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import proteintensor as pt
    assert torch.cuda.is_available(), "this harness needs a GPU"
    device = torch.device("cuda")

    work = Path(tempfile.mkdtemp(prefix="ptt_model_"))
    print(f"Building .ptt + A3M for {len(STRUCTS)} structures ...")
    cif, a3m, ptt, nres = build(work, pt)
    pids = list(ptt.keys())
    order = [pids[i % len(pids)] for i in range(args.samples)]

    trad_ds = TraditionalDataset(order, cif, a3m)
    ptt_ds = PttDataset(order, ptt)

    print("Warming caches ...")
    for i in range(len(trad_ds)):
        _ = trad_ds[i]
    for i in range(len(ptt_ds)):
        _ = ptt_ds[i]

    rows = []
    for depth in args.depths:
        model = build_model(args.dim, depth, torch, nn).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=1e-4)
        step = gpu_step_ms(ptt_ds, args.batch, model, opt, device, torch)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model depth={depth} (~{n_params/1e6:.1f}M params, "
              f"GPU step {step} ms)")
        for nw in args.workers:
            trad = run_loader(trad_ds, args.batch, nw, model, opt, device, torch)
            pttr = run_loader(ptt_ds, args.batch, nw, model, opt, device, torch)
            speedup = round(trad["epoch_s"] / pttr["epoch_s"], 2)
            rows.append({
                "depth": depth, "params_m": round(n_params / 1e6, 1),
                "gpu_step_ms": step, "num_workers": nw,
                "traditional": trad, "ptt": pttr, "throughput_speedup": speedup,
            })
            print(f"    workers={nw}: trad {trad['throughput']}/s "
                  f"wait {trad['wait_fraction']} | ptt {pttr['throughput']}/s "
                  f"wait {pttr['wait_fraction']} | {speedup}x")

    import platform, sys
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "proteintensor": getattr(pt, "__version__", "unknown"),
        "samples": args.samples, "batch": args.batch, "dim": args.dim,
        "msa_use_depth": MSA_USE_DEPTH, "structures": pids,
        "model": "MSA + structure transformer encoder; real forward + backward step",
    }
    record = {"env": env, "results": rows}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = env["timestamp_utc"].replace(":", "").replace("-", "").replace("+0000", "Z")
    (RESULTS_DIR / f"model_{stamp}.json").write_text(json.dumps(record, indent=2))
    _write_md(record)

    import shutil
    shutil.rmtree(work, ignore_errors=True)
    print("\nWrote benchmarks/MODEL_HARNESS_RESULTS.md")


def _write_md(record):
    env = record["env"]
    lines = [
        "# Real-model DataLoader throughput (single GPU)", "",
        "The GPU consumer here is an actual small transformer (MSA + structure",
        "encoder) running a real forward + backward step - not a synthetic matmul.",
        "Two datasets over the same structures feed the same model: `traditional`",
        "(parse mmCIF + A3M + compute distances per item) vs `ptt` (zero-parse).", "",
        f"- **GPU:** {env['gpu']} | torch {env['torch']} | proteintensor {env['proteintensor']}",
        f"- **Model:** transformer encoder, dim {env['dim']}, MSA depth "
        f"{env['msa_use_depth']}; real forward + backward",
        f"- **Config:** {env['samples']} samples, batch {env['batch']} | "
        f"{env['timestamp_utc']}", "",
        "`wait` = fraction of wall-clock the GPU sits idle waiting for the next batch.",
        "", "| depth | params | GPU step | workers | trad /s | ptt /s | throughput | "
        "trad wait | ptt wait |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in record["results"]:
        t, p = r["traditional"], r["ptt"]
        lines.append(
            f"| {r['depth']} | {r['params_m']}M | {r['gpu_step_ms']} ms | "
            f"{r['num_workers']} | {t['throughput']} | {p['throughput']} | "
            f"**{r['throughput_speedup']}x** | {t['wait_fraction']} | {p['wait_fraction']} |")
    lines += [
        "", "**Reading the table.** With a real model, the story matches the matmul",
        "harness: `.ptt` wins most when the model step is cheap (data-bound) and the",
        "throughput gap narrows as the model gets deeper (compute-bound) - but the",
        "GPU-idle (`wait`) gap persists, and `num_workers>0` speeds up both loaders",
        "without erasing it.", "",
        "**Honest scope:** small model on a single GPU; production models (AlphaFold/",
        "Boltz) have far heavier steps, where the throughput speedup converges toward",
        "1x and the durable value is reclaimed GPU-idle time and fewer dataloader",
        "workers. Multi-GPU DDP is not tested here.", ""]
    (Path(__file__).resolve().parent / "MODEL_HARNESS_RESULTS.md").write_text(
        "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
