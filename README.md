# HelixDB / ProteinTensor

**ProteinTensor** is an AI-native biomolecular storage format designed to eliminate
the preprocessing bottleneck in modern structural biology machine learning pipelines.

---

## The Problem

Every time a researcher trains AlphaFold, Boltz, RoseTTAFold, OpenFold, or any
structure-prediction model, the same work happens before a single GPU operation runs:

```
for each protein in dataset:
    parse mmCIF / PDB file          (30 – 350 ms per structure)
    extract sequence tokens
    build atom coordinate arrays
    construct backbone geometry
    compute covalent bond graph
    load or regenerate MSA           (2 – 480 min with JackHMMER)
    run ESM2 / ESM3 inference        (seconds per protein on GPU)
    compute distance matrices
    ...

    -> finally: model.forward(features)
```

For a 100,000-structure training run this preprocessing costs **thousands of CPU-hours
per epoch** — most of it producing identical results every time. The mmCIF file has not
changed. The sequence has not changed. The physics has not changed. Yet every run
recomputes everything from scratch.

ProteinTensor solves this by converting the PDB entry once into a `.ptt` file — a
Zarr-backed, LZ4-compressed, memory-mappable store that holds every tensor a model
needs — and then loading those tensors directly at training time with zero parsing.

```
once:   mmCIF  ->  ProteinTensor (.ptt)
always: .ptt   ->  model.forward()
```

---

## Who This Is For

**Structural biology researchers** running AlphaFold 3, Boltz, or Chai-1 who spend
hours waiting for MSA generation before every new experiment.

**ML engineers at pharma and biotech companies** iterating over large structure
databases (PDB, AlphaFold Database, ESMAtlas) where I/O throughput is a training
bottleneck measured in wall-clock days.

**Academic labs** with limited GPU budgets who cannot afford to waste compute cycles
on re-parsing text files when those GPU-hours should go toward model training.

**Software engineers building structural biology pipelines** who want a single,
well-defined intermediate format that works with PyTorch, JAX, and NumPy without
writing custom loaders for every model.

ProteinTensor is to structural biology what Parquet is to analytics, what safetensors
is to model weights, and what ONNX is to model exchange — a common, open, high-
performance format that turns a recurring computational tax into a one-time cost.

---

## Benchmark: Traditional Pipeline vs ProteinTensor

All timings are median over 30 rounds on an NVIDIA RTX 5080, CUDA 12.8, Python 3.11.
Proteins span the full range from a 76-residue domain to a 3,525-residue CRISPR enzyme.

### Per-structure load times

| Structure | Method | Res | MSA seqs | mmCIF parse | ptt: full | ptt: backbone | ptt: bonds | ptt: MSA | ptt: dist mx |
|---|---|---|---|---|---|---|---|---|---|
| 1UBQ — Ubiquitin | X-ray | 76 | 512 | 7.4 ms | 2.9 ms | 1.2 ms | 0.7 ms | 1.6 ms | 0.8 ms |
| 6LU7 — SARS-CoV-2 Mpro | X-ray | 312 | 1,024 | 28.7 ms | 2.9 ms | 1.2 ms | 0.7 ms | 5.1 ms | 1.6 ms |
| 4HHB — Hemoglobin | X-ray | 574 | 2,048 | 55.0 ms | 2.9 ms | 1.2 ms | 0.7 ms | 11.9 ms | 3.4 ms |
| 6M0J — ACE2 + RBD | Cryo-EM | 791 | 2,048 | 73.1 ms | 2.9 ms | 1.2 ms | 0.7 ms | 14.9 ms | 6.4 ms |
| 6VXX — Spike trimer | Cryo-EM | 2,916 | 8,192 | 278.5 ms | 3.3 ms | 1.3 ms | 0.9 ms | 207.8 ms | 67.8 ms |
| 6OHW — Cas12a | Cryo-EM | 3,525 | 8,192 | 345.5 ms | 3.3 ms | 1.2 ms | 0.9 ms | 240.3 ms | 105.7 ms |

**Column definitions**
- `ptt: full` — `read()` — all atoms, backbone, bonds, metadata
- `ptt: backbone` — `read_backbone()` — N/CA/C/O coordinates + sequence only
- `ptt: bonds` — `read_bonds()` — covalent graph only
- `ptt: MSA` — `read_msa()` — MSA tokens + profile (loaded from .ptt cache)
- `ptt: dist mx` — `read_pair_feature("distance_matrix")` — Ca-Ca distance matrix

### Speedup vs mmCIF baseline

| Structure | Res | full | backbone | bonds | MSA | dist mx |
|---|---|---|---|---|---|---|
| 1UBQ — Ubiquitin | 76 | 3x | 6x | 11x | 5x | 9x |
| 6LU7 — SARS-CoV-2 Mpro | 312 | 10x | 24x | 41x | 6x | 17x |
| 4HHB — Hemoglobin | 574 | 19x | 44x | 76x | 5x | 16x |
| 6M0J — ACE2 + RBD | 791 | 25x | 61x | 98x | 5x | 11x |
| 6VXX — Spike trimer | 2,916 | 85x | 217x | 304x | 1x* | 4x |
| 6OHW — Cas12a | 3,525 | 106x | 281x | 366x | 1x* | 3x |

*MSA speedup shown as 1x vs mmCIF parse because both are in the same time range for
large proteins — the real MSA comparison is vs JackHMMER generation (see below).

### Scale projection: 100,000 structures, one training epoch

| Operation | Traditional pipeline | ProteinTensor | Speedup |
|---|---|---|---|
| Structure load (parse mmCIF each epoch) | 3.6 hours | 5 min | **43x** |
| Backbone-only load (template search) | 3.6 hours | 2 min | **106x** |
| MSA generation (JackHMMER, 32-core CPU, once) | 4,000 hours | 2.2 hours | **~1,800x** |

> MSA generation assumes 2.4 min/protein on a 32-core server (PDB90 database, standard
> AlphaFold settings). With ProteinTensor, MSAs are generated once and loaded from the
> `.ptt` cache on every subsequent run. The 4,000-hour figure is the real cost that
> AlphaFold2 users pay to build training datasets.

### Disk tradeoff

A structure-only `.ptt` (no MSA, no embeddings) is 5–11x **smaller** than the source
mmCIF. A full-featured `.ptt` (8,192-sequence MSA + distance matrix + ESM2-650M
embedding at float16) is larger because it stores precomputed features that previously
lived in separate files or were never persisted at all. The tradeoff is deliberate:
pay disk space once to avoid paying GPU-hours and CPU-hours repeatedly.

---

## Install

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.9, `gemmi`, `zarr`, `numpy`, `click`, `rich`.

---

## Quick Start

### Convert a structure

```bash
proteintensor convert 1abc.cif 1abc.ptt
proteintensor info 1abc.ptt
```

### Benchmark against mmCIF

```bash
proteintensor benchmark 1abc.cif --rounds 20
```

### Cache an MSA (after running JackHMMER / ColabFold)

```python
import proteintensor as pt

msa = pt.from_a3m("1abc_uniref90.a3m",
                  tool="jackhammer", tool_version="3.3.2",
                  database="uniref90", database_date="2024-01")
pt.add_msa("1abc.ptt", msa, source="uniref90")
```

### Cache ESM2 embeddings (after GPU inference)

```python
pt.add_embedding("1abc.ptt", esm_representations,
                 model="esm2_t33_650M_UR50D", layer=-1, dtype="float16",
                 sequence_hash=pt.embedding_sequence_hash(data.sequence_tokens))
```

### Run Boltz2 directly from a .ptt file

```python
from proteintensor import BoltzAdapter

adapter = BoltzAdapter("1abc.ptt")
predictions = adapter.predict(
    "boltz_output/",
    model="boltz2",
    diffusion_samples=5,
    recycling_steps=3,
    accelerator="gpu",
)
# -> boltz_output/predictions/1abc/1abc_model_0.cif  (predicted structure)
# -> boltz_output/predictions/1abc/pae_*.npz          (PAE matrix)
# -> boltz_output/predictions/1abc/plddt_*.npz        (per-residue confidence)
```

---

## Python API

```python
import proteintensor as pt

# ------ Structure ------
data = pt.read("1abc.ptt")
data.atom_positions.shape      # (N_atoms, 3)   float32
data.sequence_tokens.shape     # (N_res,)        int32
data.backbone_positions.shape  # (N_res, 4, 3)  float32  N/CA/C/O
data.bond_edge_index.shape     # (2, N_edges)   int32   bidirectional

# Backbone only (fastest structural load)
bb = pt.read_backbone("1abc.ptt")
bb.positions.shape  # (N_res, 4, 3)

# Bond graph only
bonds = pt.read_bonds("1abc.ptt")

# ------ MSA ------
msa = pt.read_msa("1abc.ptt", source="uniref90")
msa.tokens.shape    # (N_seq, N_res)  int32
msa.profile.shape   # (N_res, 23)    float32

# ------ Pair features ------
pt.compute_and_store_distances("1abc.ptt")       # Ca-Ca distance matrix
pt.compute_and_store_contacts("1abc.ptt", threshold=8.0)

dist = pt.read_pair_feature("1abc.ptt", "distance_matrix")
dist.data.shape     # (N_res, N_res, 1)  float32

# Store arbitrary pair tensors (template features, MSA covariance, …)
pt.add_pair_feature("1abc.ptt", my_array, name="template_pair",
                    symmetric=False, dtype="float16")

# ------ PLM embeddings ------
emb = pt.read_embedding("1abc.ptt", "esm2_t33_650M_UR50D")
emb.data.shape      # (N_res, 1280)  float32  (upcast from float16 on load)

# ------ Lazy / zero-copy access ------
positions = pt.mmap_positions("1abc.ptt")       # zarr.Array — no full load
backbone  = pt.mmap_backbone("1abc.ptt")        # [N_res, 4, 3]
msa_lazy  = pt.mmap_msa_tokens("1abc.ptt", "uniref90")  # [N_seq, N_res]
emb_lazy  = pt.mmap_embedding("1abc.ptt", "esm2_t33_650M_UR50D")

# Slice without loading the full tensor
ca_window = backbone[100:164, 1, :]             # 64 Ca positions
top_100   = msa_lazy[:100, :]                   # first 100 MSA sequences

# ------ PyTorch ------
import torch
data   = pt.read("1abc.ptt")
coords = torch.from_numpy(data.atom_positions)   # (N_atoms, 3)
tokens = torch.from_numpy(data.sequence_tokens)  # (N_res,)

# ------ JAX ------
import jax.numpy as jnp
data   = pt.read("1abc.ptt")
coords = jnp.array(data.atom_positions)
```

---

## .ptt file layout

```
structure.ptt/                      Zarr directory store (v0.7)
├── .zattrs                         format, version, pdb_id, resolution, ...
├── sequence/
│   ├── tokens             [N_res]           int32    AA vocab indices (0-20)
│   ├── residue_index      [N_res]           int32    PDB sequence numbers
│   └── chain_id           [N_res]           S1       chain labels
├── atoms/
│   ├── positions          [N_atoms, 3]      float32  Angstrom coordinates
│   ├── mask               [N_atoms]         bool
│   └── b_factors          [N_atoms]         float32  B-factor / pLDDT
├── structure/
│   ├── residue_atom_start [N_res]           int32    first atom index per residue
│   └── residue_atom_count [N_res]           int32    atom count per residue
├── backbone/
│   ├── positions          [N_res, 4, 3]     float32  N/CA/C/O coords
│   └── mask               [N_res, 4]        bool     False = missing atom
├── bonds/
│   ├── edge_index         [2, N_edges]      int32    bidirectional (src, dst)
│   └── edge_type          [N_edges]         uint8    1=SINGLE 2=DOUBLE 4=AROMATIC
│                                                      5=PEPTIDE 6=DISULFIDE
├── msa/
│   └── <source>/                            one sub-group per database source
│       ├── .zattrs                          tool, version, database, date, seq SHA-256
│       ├── tokens         [N_seq, N_res]    int32    0-20=AA 21=GAP 22=MASK
│       ├── deletion_matrix [N_seq, N_res]   float32  insertions before each column
│       ├── profile        [N_res, 23]       float32  per-position residue frequencies
│       └── deletion_mean  [N_res]           float32
├── pairs/
│   └── <name>/                              one sub-group per named feature
│       ├── .zattrs                          channels, symmetric, dtype, description
│       └── data           [N_res, N_res, C] any dtype, chunked 128x128xC
└── embeddings/
    └── <model>/                             one sub-group per PLM model
        ├── .zattrs                          model, layer, dim, dtype, seq SHA-256
        └── data           [N_res, D]        float32 or float16, chunked 256xD
```

---

## Supported models

| Model | Adapter | Status |
|---|---|---|
| Boltz 2 | `BoltzAdapter` | Verified — end-to-end prediction on RTX 5080 |
| Boltz 1 | `BoltzAdapter(model="boltz1")` | Supported |
| OpenFold | — | Roadmap |
| RoseTTAFold-All-Atom | — | Roadmap |

---

## Run tests

```bash
pytest tests/ -v
```

84 tests across structure roundtrip, backbone/bonds/MSA/pairs/embeddings,
A3M parsing, Boltz adapter (YAML + A3M generation, real-protein integration).

---

## Roadmap

- [x] Backbone-only dense layout `[N_res, 4, 3]` for faster backbone access
- [x] Bond graph storage (`edge_index`) — SINGLE / DOUBLE / AROMATIC / PEPTIDE / DISULFIDE
- [x] MSA feature caching — A3M parser, provenance tracking, multi-source per file
- [x] Pair representation block `[N, N, C]` — distance matrix, contact map, generic named tensors
- [x] Pre-embedded ESM2 / ESM3 features — float16 storage, provenance hash, lazy mmap access
- [x] Model adapters: Boltz2 — end-to-end prediction from `.ptt` verified on RTX 5080
- [ ] Multi-structure dataset container — one Zarr store, N structures, batched loading
- [ ] Cloud streaming — S3 / GCS via `fsspec`, training directly from object storage
