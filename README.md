# HelixDB / ProteinTensor

**ProteinTensor** is an AI-native biomolecular storage format.  
It caches the preprocessing work that structural biology models repeat on every run — sequence tokens, residue indices, atom coordinates, masks — so pipelines can load directly into tensors instead of re-parsing mmCIF files.

```
PDB / mmCIF / FASTA
        ↓  (once)
  ProteinTensor (.ptt)
        ↓  (every run)
  Direct tensor loading
        ↓
  Model inference
```

The format is backed by [Zarr](https://zarr.readthedocs.io) with LZ4/Blosc compression and designed for memory-mapped, zero-copy access.

---

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.9, `gemmi`, `zarr`, `numpy`, `click`, `rich`.

---

## Usage

### Convert

```bash
proteintensor convert 1abc.cif 1abc.ptt
```

Output:

```
╭─ Converted → 1abc.ptt ──────────────────────╮
│  PDB ID      1ABC                            │
│  Chains      A, B                            │
│  Residues    512                             │
│  Atoms       4,096                           │
│  Resolution  2.10 Å                          │
│  Method      X-RAY DIFFRACTION               │
│                                              │
│  Parse time  142.3 ms                        │
│  Write time   18.7 ms                        │
│  Source       1.2 MB                         │
│  Output     452.0 KB                         │
│  Ratio        0.37x                          │
╰──────────────────────────────────────────────╯
```

### Inspect

```bash
proteintensor info 1abc.ptt
```

### Benchmark

```bash
proteintensor benchmark 1abc.cif --rounds 20
```

Runs 20 timed iterations of both mmCIF parsing and ProteinTensor loading,
then prints a comparison table with median, mean, min, and P95 latencies.

---

## Python API

```python
import proteintensor as pt

# Full load
data = pt.read("1abc.ptt")
print(data.atom_positions.shape)   # (N_atoms, 3)  float32
print(data.sequence_tokens.shape)  # (N_res,)       int32

# Lazy / memory-mapped (no full load)
positions = pt.mmap_positions("1abc.ptt")  # zarr.Array — index like numpy
tokens    = pt.mmap_tokens("1abc.ptt")

# Convert programmatically
from proteintensor.converters import from_mmcif
data = from_mmcif("1abc.cif", pdb_id="1ABC")
pt.write(data, "1abc.ptt")
```

### PyTorch

```python
import torch
data = pt.read("1abc.ptt")
coords  = torch.from_numpy(data.atom_positions)   # (N_atoms, 3)
tokens  = torch.from_numpy(data.sequence_tokens)  # (N_res,)
```

### JAX

```python
import jax.numpy as jnp
data   = pt.read("1abc.ptt")
coords = jnp.array(data.atom_positions)
```

---

## .ptt file layout

```
structure.ptt/                      Zarr directory store (v0.4)
├── .zattrs                         format, version, pdb_id, resolution, msa_sources, …
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
│   └── edge_type          [N_edges]         uint8    1=SINGLE 2=DOUBLE 4=AROMATIC 5=PEPTIDE 6=SS
└── msa/
    └── <source>/                            one sub-group per database (uniref90, bfd, …)
        ├── .zattrs                          tool, version, database, date, sequence SHA-256
        ├── tokens         [N_seq, N_res]    int32    0-20=AA 21=GAP 22=MASK
        ├── deletion_matrix [N_seq, N_res]   float32  insertions before each column
        ├── profile        [N_res, 23]       float32  per-position residue frequencies
        └── deletion_mean  [N_res]           float32
```

---

## Run tests

```bash
pytest tests/ -v
```

---

## Roadmap

- [x] Backbone-only dense layout `[N_res, 4, 3]` for faster backbone access  
- [x] Bond graph storage (`edge_index`) — SINGLE / DOUBLE / AROMATIC / PEPTIDE / DISULFIDE  
- [x] MSA feature caching — A3M parser, provenance tracking, multi-source per file  
- [ ] Pair representation block `[N, N, C]`  
- [ ] Pre-embedded ESM2 / ESM3 features  
- [ ] Model adapters: Boltz, OpenFold, RoseTTAFold  
- [ ] Multi-structure dataset container (one store, N structures)  
- [ ] Cloud streaming (S3 / GCS via `fsspec`)  
