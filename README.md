# ProteinTensor Introduction

![ProteinTensor - AI-native protein data format: convert structure or sequence into cached tensors](assets/banner.png)

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
per epoch** - most of it producing identical results every time. The mmCIF file has not
changed. The sequence has not changed. The physics has not changed. Yet every run
recomputes everything from scratch.

ProteinTensor solves this by converting the PDB entry once into a `.ptt` file - a
Zarr-backed, LZ4-compressed, memory-mappable store that holds every tensor a model
needs - and then loading those tensors directly at training time with zero parsing.

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
is to model weights, and what ONNX is to model exchange - a common, open, high-
performance format that turns a recurring computational tax into a one-time cost.

---

## Benchmark: Traditional Pipeline vs ProteinTensor

All timings are median over 30 rounds on an NVIDIA RTX 5080, CUDA 12.8, Python 3.11.
Proteins span the full range from a 76-residue domain to a 3,525-residue CRISPR enzyme.
Run `python boltz_benchmark.py` to reproduce.

### Per-structure load times

| Structure | Method | Res | MSA seqs | mmCIF parse | ptt: full | ptt: backbone | ptt: bonds | ptt: MSA | ptt: dist mx |
|---|---|---|---|---|---|---|---|---|---|
| 1UBQ - Ubiquitin | X-ray | 76 | 512 | 7.2 ms | 2.8 ms | 1.2 ms | 0.7 ms | 1.6 ms | 0.8 ms |
| 6LU7 - SARS-CoV-2 Mpro | X-ray | 312 | 1,024 | 29.6 ms | 2.9 ms | 1.2 ms | 0.7 ms | 5.1 ms | 2.0 ms |
| 4HHB - Hemoglobin | X-ray | 574 | 2,048 | 55.3 ms | 2.9 ms | 1.2 ms | 0.7 ms | 11.3 ms | 3.5 ms |
| 6M0J - ACE2 + RBD | Cryo-EM | 791 | 2,048 | 74.7 ms | 2.9 ms | 1.2 ms | 0.7 ms | 14.7 ms | 6.4 ms |
| 6VXX - Spike trimer | Cryo-EM | 2,916 | 8,192 | 283.4 ms | 3.3 ms | 1.3 ms | 0.9 ms | 208.3 ms | 71.1 ms |
| 6OHW - Cas12a | Cryo-EM | 3,525 | 8,192 | 352.4 ms | 3.3 ms | 1.2 ms | 1.0 ms | 240.7 ms | 104.5 ms |

**Column definitions**
- `ptt: full` - `read()` - all atoms, backbone, bonds, metadata
- `ptt: backbone` - `read_backbone()` - N/CA/C/O coordinates + sequence only
- `ptt: bonds` - `read_bonds()` - covalent graph only
- `ptt: MSA` - `read_msa()` - MSA tokens + profile (loaded from .ptt cache)
- `ptt: dist mx` - `read_pair_feature("distance_matrix")` - Ca-Ca distance matrix

### Speedup vs mmCIF baseline

| Structure | Res | full | backbone | bonds | MSA | dist mx |
|---|---|---|---|---|---|---|
| 1UBQ - Ubiquitin | 76 | 3x | 6x | 11x | 4x | 9x |
| 6LU7 - SARS-CoV-2 Mpro | 312 | 10x | 24x | 43x | 6x | 15x |
| 4HHB - Hemoglobin | 574 | 19x | 45x | 78x | 5x | 16x |
| 6M0J - ACE2 + RBD | 791 | 26x | 61x | 102x | 5x | 12x |
| 6VXX - Spike trimer | 2,916 | 87x | 223x | 308x | 1x* | 4x |
| 6OHW - Cas12a | 3,525 | 108x | 284x | 370x | 1x* | 3x |

*MSA speedup shown as 1x vs mmCIF parse because both are in the same time range for
large proteins - the real MSA comparison is vs JackHMMER generation (see below).

### Feature assembly: time to prepare all tensors for model.forward()

Traditional = mmCIF parse + A3M MSA parse + distance-matrix compute. ProteinTensor
= read the structure, MSA, distance matrix, and ESM2 embedding from a single
pre-cached `.ptt`. Reproduce with `python benchmarks/assembly_benchmark.py`
(MSA depth and embedding shape are realistic; numeric content is synthetic, so
timing reflects tensor dimensions, not values).

| Structure | Res | MSA depth | Traditional | ProteinTensor | Speedup |
|---|---|---|---|---|---|
| 1UBQ - Ubiquitin | 76 | 512 | 14.1 ms | 7.1 ms | 2.0x |
| 6LU7 - SARS-CoV-2 Mpro | 312 | 1,024 | 48.7 ms | 13.6 ms | 3.6x |
| 4HHB - Hemoglobin | 574 | 2,048 | 118.0 ms | 22.7 ms | 5.2x |
| 6M0J - ACE2 + RBD | 791 | 2,048 | 196.4 ms | 38.3 ms | 5.1x |
| 6VXX - Spike trimer | 2,916 | 8,192 | 1,395 ms | 309 ms | 4.5x |
| 6OHW - Cas12a | 3,525 | 8,192 | 1,462 ms | 381 ms | 3.8x |

Average speedup across all six structures: **4x** for full feature assembly
(measured on a Windows CPU box - see
[`benchmarks/ASSEMBLY_RESULTS.md`](benchmarks/ASSEMBLY_RESULTS.md)).

> **On an earlier 34x figure:** prior versions reported ~34x here. That number was
> measured against ProteinTensor's original scalar A3M parser, which dominated the
> traditional side (~11 s to parse an 8,192-deep MSA). Vectorizing that parser in
> v0.2.0 cut the traditional baseline ~8x, so the *fair* feature-assembly speedup
> is now ~4x. The `.ptt` read side was unchanged - only the baseline got faster.

### Drug target benchmark

Same methodology across six high-value drug targets spanning KRAS oncology,
HIV antivirals, PD-L1 immunotherapy, p53, cardiovascular (PCSK9), and a full
IgG1 antibody. Numbers are consistent with the structural biology benchmark above.

| Target | Res | mmCIF parse | ptt: full | ptt: backbone | ptt: bonds | ptt: MSA | ptt: dist mx |
|---|---|---|---|---|---|---|---|
| 6OIM - KRAS G12C + Sotorasib | 167 | 16.6 ms | 2.8 ms | 1.2 ms | 0.7 ms | 2.8 ms | 1.1 ms |
| 3HTB - HIV-1 protease | 163 | 16.0 ms | 2.8 ms | 1.2 ms | 0.7 ms | 2.7 ms | 1.1 ms |
| 5WT9 - PD-L1 checkpoint | 533 | 53.8 ms | 2.9 ms | 1.2 ms | 0.7 ms | 13.1 ms | 3.3 ms |
| 1TUP - p53 tumor suppressor | 585 | 56.5 ms | 2.8 ms | 1.2 ms | 0.7 ms | 12.4 ms | 3.4 ms |
| 2P4E - PCSK9 | 586 | 54.7 ms | 2.8 ms | 1.2 ms | 0.7 ms | 12.1 ms | 3.4 ms |
| 1IGT - IgG1 antibody | 1,316 | 123.4 ms | 2.9 ms | 1.2 ms | 0.8 ms | 46.8 ms | 16.4 ms |

| Target | Res | full | backbone | bonds | MSA | dist mx |
|---|---|---|---|---|---|---|
| 6OIM - KRAS G12C + Sotorasib | 167 | 6x | 14x | 24x | 6x | 15x |
| 3HTB - HIV-1 protease | 163 | 6x | 14x | 23x | 6x | 14x |
| 5WT9 - PD-L1 checkpoint | 533 | 19x | 44x | 77x | 4x | 16x |
| 1TUP - p53 tumor suppressor | 585 | 20x | 47x | 80x | 5x | 17x |
| 2P4E - PCSK9 | 586 | 19x | 46x | 77x | 5x | 16x |
| 1IGT - IgG1 antibody | 1,316 | 42x | **100x** | **162x** | 3x | 8x |

### DataLoader batch throughput

Measured using `ProteinDataset` + `ProteinDataset.collate()`, loading structures into
padded batches ready for `model.forward()`. Single process, no prefetch workers.

| Batch size | ms / batch | Structures / sec |
|---|---|---|
| 1 | 0.01 ms | 88,106 |
| 4 | 0.04 ms | 108,696 |
| 8 | 0.37 ms | 21,707 |
| 16 | 0.95 ms | 16,783 |
| 32 | 2.0 ms | **15,854** |

### Scale projection: 100,000 structures, one training epoch

These are **projections**, extrapolated from the measured per-structure timings
above - not end-to-end measurements at 100k scale.

| Operation | Traditional pipeline | ProteinTensor | Speedup |
|---|---|---|---|
| Structure load (parse mmCIF each epoch) | 3.7 hours | 5 min | **45x** |
| Backbone-only load (template search) | 3.7 hours | 2 min | **109x** |
| Full feature assembly (seq + MSA + pairs + emb) | 15 hours | 3.6 hours | **4x** |
| MSA generation (JackHMMER, 32-core CPU, once) | 4,000 hours | 2.2 hours | **1,794x** |

> MSA generation assumes 2.4 min/protein on a 32-core server (PDB90 database, standard
> AlphaFold settings). ProteinTensor generates MSAs once and loads from the `.ptt` cache
> on every subsequent run. The 4,000-hour figure is the real cost AlphaFold2 and Boltz
> users pay to build training datasets from scratch.

> **Measured vs projected - read this.** The **1,794x** above is MSA *generation*
> (building the alignment once with JackHMMER) and is a **literature-based
> projection**, not something benchmarked here. What *is* measured on hardware is
> the recurring per-epoch MSA **load** - reading a cached MSA from `.ptt` vs
> re-parsing A3M text each epoch (against a vectorized A3M parser baseline):
> **3.4x-5.9x**, growing with MSA depth. See
> [`benchmarks/MSA_RESULTS.md`](benchmarks/MSA_RESULTS.md). These are different
> quantities; do not read the 1,794x as a measured load speedup.

### Disk tradeoff

A full-featured `.ptt` (8,192-sequence MSA + distance matrix + ESM2-650M embedding at
float16) averages **23x larger** than the source mmCIF across the six benchmark structures.
The tradeoff is deliberate: pay disk space once to avoid paying GPU-hours and CPU-hours
on every training run. A structure-only `.ptt` with no cached features is smaller than
the source mmCIF.

---

## Install

```bash
pip install -e ".[dev]"           # core + dev tools
pip install -e ".[cloud]"         # adds fsspec, s3fs, gcsfs for remote reads
pip install -e ".[dev,cloud]"     # everything
```

Requires Python >= 3.9, `gemmi`, `zarr`, `numpy`, `click`, `rich`.

---

## Quick Start

### Convert a structure

```bash
proteintensor convert 1abc.cif 1abc.ptt
proteintensor info 1abc.ptt
```

### Convert a sequence (no structure required)

For sequence-driven predictors like AlphaFold and Boltz, the primary input is a
sequence, not a structure. ProteinTensor can build a sequence-only `.ptt` (no
coordinates) directly from a raw string or a FASTA file:

```bash
proteintensor convert-seq MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDG ubq.ptt
proteintensor convert-seq complex.fasta complex.ptt   # multi-record FASTA -> multi-chain
```

```python
import proteintensor as pt

data = pt.from_sequence("MQIFVKTLTGK...", pdb_id="UBQ", chain_id="A")
data.has_structure        # False - sequence-only entry
data.sequence_tokens      # (N_res,)  int32

pt.write(data, "ubq.ptt")

# FASTA: a single record -> one chain; multiple records -> multi-chain complex
data = pt.from_fasta("complex.fasta")
```

### Batch-convert a directory

Convert an entire directory of structures in parallel, with progress reporting.
Files that fail to parse are skipped and listed in the summary; already-converted
outputs are skipped by default.

```bash
proteintensor convert-dir ./pdb_files/ ./ptt_files/            # auto worker count
proteintensor convert-dir ./pdb_files/ ./ptt_files/ --workers 16 --recursive
proteintensor convert-dir ./pdb_files/ ./ptt_files/ --overwrite  # rebuild existing
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

# ------ Ligands / small molecules ------
# Capture drugs, cofactors, and ions from a structure (opt-in)
data = pt.from_mmcif("6oim.cif", include_ligands=True)
[l.name for l in data.ligands]        # ['MG', 'GDP', 'MOV']  (MOV = sotorasib)

ligs = pt.read_ligands("6oim.ptt")
ligs[0].elements                      # (N_atoms,)  S2  element symbols
ligs[0].positions                     # (N_atoms, 3)  float32
pt.list_ligands("6oim.ptt")           # ['MG', 'GDP', 'MOV']

# Build a ligand from SMILES (needs `pip install "proteintensor[ligands]"`)
aspirin = pt.from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="AIN")
pt.add_ligand("target.ptt", aspirin)  # attach to an existing .ptt

# ------ Lazy / zero-copy access ------
positions = pt.mmap_positions("1abc.ptt")       # zarr.Array - no full load
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

# ------ Cloud streaming ------
# Read a single structure directly from S3 (no local download)
data = pt.read("s3://my-bucket/proteins/1abc.ptt")
bb   = pt.read_backbone("s3://my-bucket/proteins/1abc.ptt")
arr  = pt.mmap_positions("s3://my-bucket/proteins/1abc.ptt")  # lazy remote array

# Open a dataset stored in cloud
ds = pt.ProteinDataset("s3://my-bucket/training.ptt")

# Prepare a local .ptt for fast remote reads before uploading (one-time)
pt.consolidate("1abc.ptt")                  # writes .zmetadata
# aws s3 cp -r 1abc.ptt s3://my-bucket/proteins/1abc.ptt

# Pass storage_options for credentials or custom endpoints
data = pt.read(
    "s3://my-bucket/proteins/1abc.ptt",
    storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"},
)

# ------ Multi-structure dataset ------
# Structure .ptt files and sequence-only .ptt files can be mixed in one dataset.
pt.create_dataset("training.ptt")
for ptt_file in Path("ptt_files").glob("*.ptt"):
    pt.add_to_dataset("training.ptt", ptt_file)

ds = pt.ProteinDataset("training.ptt")
len(ds)               # number of structures
ds[0]                 # ProteinTensorData by index
ds["1ABC"]            # ProteinTensorData by PDB ID (case-insensitive)

# PyTorch DataLoader - collate pads variable-length sequences
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=8, collate_fn=pt.ProteinDataset.collate)
for batch in loader:
    coords  = torch.from_numpy(batch["atom_positions"])   # (B, max_atoms, 3)
    pad     = torch.from_numpy(batch["padding_mask"])     # (B, max_res)  True=real
    has_str = torch.from_numpy(batch["has_structure"])    # (B,)  False = sequence-only
```

Sequence-only entries contribute zero atoms to the batch (`n_atoms == 0`,
`has_structure == False`), so sequence-driven and structure-based samples can be
loaded together in one `DataLoader`.

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
├── embeddings/
│   └── <model>/                             one sub-group per PLM model
│       ├── .zattrs                          model, layer, dim, dtype, seq SHA-256
│       └── data           [N_res, D]        float32 or float16, chunked 256xD
└── ligands/
    └── <index>/                            one sub-group per non-polymer ligand
        ├── .zattrs                          name (CCD), chain_id, res_num, smiles
        ├── elements       [N_atoms]         S2       element symbols
        ├── positions      [N_atoms, 3]      float32  Angstrom coordinates
        └── b_factors      [N_atoms]         float32
```

### Multi-structure dataset layout

```
dataset.ptt/                        Zarr directory store
├── .zattrs                         format="proteintensor-dataset", version, num_structures
└── structures/
    ├── 000000/                     zero-padded integer key
    │   └── (same layout as single .ptt above)
    ├── 000001/
    │   └── ...
    └── ...
```

Each sub-group under `structures/` is identical to a standalone `.ptt` root, so all single-structure reader helpers work on sliced groups.

---

## Supported models

| Model | Adapter | Status |
|---|---|---|
| Boltz 2 | `BoltzAdapter` | Verified - end-to-end prediction on RTX 5080 |
| Boltz 1 | `BoltzAdapter(model="boltz1")` | Supported |
| OpenFold | - | Planned |
| RoseTTAFold-All-Atom | - | Planned |
| Chai-1 | - | Planned |

---

## Run tests

```bash
pytest tests/ -v
```

106 tests across structure roundtrip, backbone/bonds/MSA/pairs/embeddings,
A3M parsing, Boltz adapter, multi-structure dataset, and cloud streaming
(memory:// fsspec - no real cloud account required).

---

## Roadmap

- [x] Backbone-only dense layout `[N_res, 4, 3]` for faster backbone access
- [x] Bond graph storage (`edge_index`) - SINGLE / DOUBLE / AROMATIC / PEPTIDE / DISULFIDE
- [x] MSA feature caching - A3M parser, provenance tracking, multi-source per file
- [x] Pair representation block `[N, N, C]` - distance matrix, contact map, generic named tensors
- [x] Pre-embedded ESM2 / ESM3 features - float16 storage, provenance hash, lazy mmap access
- [x] Model adapters: Boltz2 - end-to-end prediction from `.ptt` verified on RTX 5080
- [x] Multi-structure dataset container - one Zarr store, N structures, PyTorch DataLoader compatible
- [x] Cloud streaming - S3 / GCS via `fsspec`, training directly from object storage

**Model coverage**
- [ ] OpenFold adapter
- [ ] RoseTTAFold-All-Atom adapter
- [ ] Chai-1 adapter

**Data pipeline**
- [x] Batch convert CLI - convert entire PDB directories in parallel with progress reporting
- [ ] Sequence-identity dataset splitting - MMseqs2-based cluster splits to prevent data leakage between train / val / test

**Format extensions**
- [x] Ligand / small-molecule support - CCD-based extraction from structures, SMILES input via RDKit, element/coordinate storage (bond graphs and binding-site annotations still to come)
- [ ] MD trajectory storage - time axis `[N_frames, N_atoms, 3]` for conformational ensembles and AlphaFold 3 diffusion trajectories

**Performance**
- [ ] Parallel DataLoader workers - thread-safe multi-worker prefetching verified under PyTorch DDP
- [ ] Format version migration CLI - upgrade .ptt files in-place across version bumps
