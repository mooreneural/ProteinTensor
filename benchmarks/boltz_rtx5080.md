# Boltz-2 end-to-end benchmark (RTX 5080)

Real measurements of the `.ptt` -> Boltz-2 path on an RTX 5080. Unlike
`RESULTS.md` (CPU format path), this records an actual GPU structure prediction
driven from a ProteinTensor file, to verify the pipeline works end-to-end and to
measure where the format actually helps.

## Environment

- **GPU:** NVIDIA GeForce RTX 5080, 16 GB, driver 610.47
- **torch:** 2.11.0+cu128 (CUDA available)
- **boltz:** 2.2.1
- **proteintensor:** 0.1.3
- **Date:** 2026-06-21

## Run configuration

- Target: **1UBQ** (ubiquitin, 76 residues), single chain
- Mode: **single-sequence** (query-only A3M, no MSA homologs) - self-contained,
  no MSA server / network required
- Boltz: `model=boltz2`, `diffusion_samples=1`, `recycling_steps=3`,
  `sampling_steps=200`, `accelerator=gpu`, `seed=42`, `no_kernels=True`

## Results

### Input preparation (where the format helps)

Time to turn the source into Boltz-ready input (YAML + A3M), median of 11 rounds:

| Path | Median |
|---|---|
| from `.ptt` (zero-parse) | **2.27 ms** |
| from mmCIF (gemmi parse each time) | 9.05 ms |
| **speedup** | **4.0x** |

### End-to-end GPU prediction

| Metric | Value |
|---|---|
| Wall-clock (mmCIF -> .ptt -> adapter -> Boltz-2 -> structure) | **102.8 s** |
| Predicted structure | `1UBQ_model_0.cif` (76 res, 601 atoms) |
| Re-parses cleanly with `from_mmcif` | yes |
| confidence_score | 0.9188 |
| pTM | 0.9098 |
| complex pLDDT | 0.921 |

Boltz also wrote PAE, PDE, and pLDDT matrices alongside the structure.

## Honest scope notes

- **The end-to-end time is format-independent.** It is dominated by GPU diffusion,
  which Boltz runs identically regardless of input source. The `.ptt` advantage is
  in **input prep** (4.0x above) and in MSA/embedding caching (see `RESULTS.md`),
  not in the inference itself. The 102.8 s is a *correctness/verification* result,
  not a format speedup.
- **Single-sequence mode** was used for a self-contained run. A real MSA (cached in
  the `.ptt`) would raise accuracy and is where the format's caching advantage is
  largest, but needs MSA generation first.
- **`no_kernels=True`** was required because the optional `cuequivariance_ops_torch`
  CUDA kernels are not installed. With the kernels installed, inference would be
  faster. The first attempt (kernels on) failed inside Boltz's own kernel import -
  the `.ptt`/adapter path itself was not implicated.
- **This is Boltz, not AlphaFold3.** AF3 is not installed on this machine (it needs
  separately licensed weights), so no AF3 numbers are reported.
