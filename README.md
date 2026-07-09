# ProteinTensor Introduction

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

All timings are median over 30 rounds on a Windows workstation (RTX 5080, Python
3.11.9); mmCIF parsing and `.ptt` reads are CPU-bound, so these reflect CPU
performance. Proteins span the full range from a 76-residue domain to a
3,525-residue CRISPR enzyme. Run `python boltz_benchmark.py` to reproduce.

### Per-structure load times

| Structure | Method | Res | MSA seqs | mmCIF parse | ptt: full | ptt: backbone | ptt: bonds | ptt: MSA | ptt: dist mx |
|---|---|---|---|---|---|---|---|---|---|
| 1UBQ - Ubiquitin | X-ray | 76 | 512 | 7.4 ms | 3.2 ms | 1.3 ms | 0.8 ms | 1.8 ms | 0.8 ms |
| 6LU7 - SARS-CoV-2 Mpro | X-ray | 312 | 1,024 | 28.7 ms | 3.3 ms | 1.3 ms | 0.8 ms | 5.2 ms | 1.9 ms |
| 4HHB - Hemoglobin | X-ray | 574 | 2,048 | 54.1 ms | 3.3 ms | 1.3 ms | 0.8 ms | 11.5 ms | 3.6 ms |
| 6M0J - ACE2 + RBD | Cryo-EM | 791 | 2,048 | 73.2 ms | 3.3 ms | 1.4 ms | 0.8 ms | 15.3 ms | 6.9 ms |
| 6VXX - Spike trimer | Cryo-EM | 2,916 | 8,192 | 283.9 ms | 3.7 ms | 1.4 ms | 1.0 ms | 213.7 ms | 74.7 ms |
| 6OHW - Cas12a | Cryo-EM | 3,525 | 8,192 | 346.5 ms | 3.7 ms | 1.3 ms | 1.0 ms | 243.9 ms | 107.3 ms |

**Column definitions**
- `ptt: full` - `read()` - all atoms, backbone, bonds, metadata
- `ptt: backbone` - `read_backbone()` - N/CA/C/O coordinates + sequence only
- `ptt: bonds` - `read_bonds()` - covalent graph only
- `ptt: MSA` - `read_msa()` - MSA tokens + profile (loaded from .ptt cache)
- `ptt: dist mx` - `read_pair_feature("distance_matrix")` - Ca-Ca distance matrix

### Speedup vs mmCIF baseline

| Structure | Res | full | backbone | bonds | MSA | dist mx |
|---|---|---|---|---|---|---|
| 1UBQ - Ubiquitin | 76 | 2x | 6x | 10x | 4x | 9x |
| 6LU7 - SARS-CoV-2 Mpro | 312 | 9x | 21x | 38x | 5x | 15x |
| 4HHB - Hemoglobin | 574 | 17x | 40x | 70x | 5x | 15x |
| 6M0J - ACE2 + RBD | 791 | 22x | 54x | 92x | 5x | 11x |
| 6VXX - Spike trimer | 2,916 | 76x | 201x | 285x | 1x* | 4x |
| 6OHW - Cas12a | 3,525 | 95x | 257x | 343x | 1x* | 3x |

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
| 6OIM - KRAS G12C + Sotorasib | 167 | 17.1 ms | 3.4 ms | 1.3 ms | 0.8 ms | 3.0 ms | 1.3 ms |
| 3HTB - HIV-1 protease | 163 | 16.5 ms | 3.3 ms | 1.4 ms | 0.8 ms | 2.8 ms | 1.3 ms |
| 5WT9 - PD-L1 checkpoint | 533 | 54.8 ms | 3.8 ms | 1.4 ms | 0.8 ms | 11.9 ms | 3.8 ms |
| 1TUP - p53 tumor suppressor | 585 | 57.4 ms | 3.4 ms | 1.4 ms | 0.8 ms | 13.0 ms | 4.0 ms |
| 2P4E - PCSK9 | 586 | 55.4 ms | 3.4 ms | 1.4 ms | 0.8 ms | 12.8 ms | 4.1 ms |
| 1IGT - IgG1 antibody | 1,316 | 127.3 ms | 3.5 ms | 1.4 ms | 0.8 ms | 47.1 ms | 17.9 ms |

| Target | Res | full | backbone | bonds | MSA | dist mx |
|---|---|---|---|---|---|---|
| 6OIM - KRAS G12C + Sotorasib | 167 | 5x | 13x | 22x | 6x | 13x |
| 3HTB - HIV-1 protease | 163 | 5x | 12x | 21x | 6x | 13x |
| 5WT9 - PD-L1 checkpoint | 533 | 15x | 40x | 69x | 5x | 14x |
| 1TUP - p53 tumor suppressor | 585 | 17x | 42x | 71x | 4x | 14x |
| 2P4E - PCSK9 | 586 | 16x | 41x | 70x | 4x | 14x |
| 1IGT - IgG1 antibody | 1,316 | 37x | **92x** | **156x** | 3x | 7x |

### DataLoader batch throughput

Measured using `ProteinDataset` + `ProteinDataset.collate()`, loading structures into
padded batches ready for `model.forward()`. Single process, no prefetch workers.

| Batch size | ms / batch | Structures / sec |
|---|---|---|
| 1 | 0.01 ms | 97,088 |
| 4 | 0.03 ms | 116,279 |
| 8 | 0.42 ms | 19,242 |
| 16 | 0.97 ms | 16,412 |
| 32 | 2.1 ms | **15,033** |

### Scale projection: 100,000 structures, one training epoch

These are **projections**, extrapolated from the measured per-structure timings
above - not end-to-end measurements at 100k scale.

| Operation | Traditional pipeline | ProteinTensor | Speedup |
|---|---|---|---|
| Structure load (parse mmCIF each epoch) | 3.8 hours | 6 min | **37x** |
| Backbone-only load (template search) | 3.8 hours | 2 min | **95x** |
| Full feature assembly (seq + MSA + pairs + emb) | 16 hours | 3.9 hours | **4x** |
| MSA generation (JackHMMER, 32-core CPU, once) | 4,000 hours | 2.7 hours | **1,477x** |

> MSA generation assumes 2.4 min/protein on a 32-core server (PDB90 database, standard
> AlphaFold settings). ProteinTensor generates MSAs once and loads from the `.ptt` cache
> on every subsequent run. The 4,000-hour figure is the real cost AlphaFold2 and Boltz
> users pay to build training datasets from scratch.

> **Measured vs projected - read this.** The **1,477x** above is MSA *generation*
> (building the alignment once with JackHMMER) and is a **literature-based
> projection**, not something benchmarked here. What *is* measured on hardware is
> the recurring per-epoch MSA **load** - reading a cached MSA from `.ptt` vs
> re-parsing A3M text each epoch (against a vectorized A3M parser baseline):
> **3.4x-5.9x**, growing with MSA depth. See
> [`benchmarks/MSA_RESULTS.md`](benchmarks/MSA_RESULTS.md). These are different
> quantities; do not read the 1,477x as a measured load speedup.

### Disk tradeoff

A full-featured `.ptt` (8,192-sequence MSA + distance matrix + ESM2-650M embedding at
float16) averages **23x larger** than the source mmCIF across the six benchmark structures.
The tradeoff is deliberate: pay disk space once to avoid paying GPU-hours and CPU-hours
on every training run. A structure-only `.ptt` with no cached features is smaller than
the source mmCIF. The dominant `O(N^2)` cost - dense pair features - can be stored as a
sparse radius graph instead (`compute_and_store_distances_sparse`), cutting the distance
matrix 2x-76x on disk with no loss inside the cutoff (see
[`benchmarks/SPARSE_PAIRS_RESULTS.md`](benchmarks/SPARSE_PAIRS_RESULTS.md)).

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

### Load a `.ptt`

```python
import proteintensor as pt

data = pt.read("1abc.ptt")
data.backbone_positions.shape   # (N_res, 4, 3)  float32
bb  = pt.read_backbone("1abc.ptt")               # fastest structural load
msa = pt.read_msa("1abc.ptt", source="uniref90") # cached MSA, zero-parse
```

See the [full Python API](docs/API.md) for MSA/embedding caching, sparse pair
features, ligands, cloud streaming, and the multi-structure DataLoader.

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

The full Python API - reading, lazy/zero-copy access, dense and sparse pair
features, MSA and embedding caching, ligands, cloud streaming, model adapters,
and the multi-structure DataLoader - is in **[docs/API.md](docs/API.md)**.

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
├── pairs_sparse/
│   └── <name>/                              sparse COO pair feature (radius graph)
│       ├── .zattrs                          n_residues, channels, nnz, mode, cutoff, symmetric
│       ├── indices        [2, nnz]          int32    kept (row, col) pairs
│       └── values         [nnz, C]          any dtype  value per kept pair
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

## Model adapters

**Boltz** runs end-to-end from a `.ptt`. The **OpenFold**, **Chai-1**, and
**AlphaFold 3** adapters convert a `.ptt` into each model's native input files
(`write_input`) - validated against each format, with any cached MSA embedded so
the model can skip regeneration. Those models are not bundled, so end-to-end
prediction through them is not verified here.

| Model | Native input (per run) | `.ptt` adapter |
|---|---|---|
| Boltz-2 / Boltz-1 | FASTA + optional A3M MSA | `BoltzAdapter` - verified end-to-end on RTX 5080 |
| OpenFold | FASTA + A3M MSA; trains on mmCIF | `OpenFoldAdapter.write_input` - input generation |
| Chai-1 | FASTA + optional MSA / templates / restraints | `ChaiAdapter.write_input` - input generation |
| AlphaFold 3 | JSON (sequences, ligands) + generated MSA | `AlphaFold3Adapter.write_input` - input generation |

```python
from proteintensor import AlphaFold3Adapter, ChaiAdapter, OpenFoldAdapter

AlphaFold3Adapter("1abc.ptt").write_input("af3/1abc.json")   # AF3 fold_input JSON
ChaiAdapter("1abc.ptt").write_input("chai/")                 # Chai FASTA (+ A3M)
OpenFoldAdapter("1abc.ptt").write_input("openfold/")         # FASTA + alignments/
```

Every one of these re-derives the same features before each run - parse the
structure (mmCIF / PDB), tokenize the sequence (FASTA), and generate or parse the
MSA. ProteinTensor replaces that recurring work with a single zero-parse read:

| Input path | Structure | Sequence + MSA | Cost per run |
|---|---|---|---|
| Native (mmCIF/PDB + FASTA + MSA files) | parse mmCIF/PDB text | parse FASTA + generate/parse MSA | re-parsed and re-featurized every epoch |
| ProteinTensor (`.ptt`) | zero-parse read | zero-parse read (lazy, partial) | converted once, then loaded |

---

## Run tests

```bash
pytest tests/ -v
```

166 tests across structure roundtrip, backbone/bonds/MSA/pairs (dense + sparse)/
embeddings/ligands, sequence conversion, A3M parsing, model input adapters (Boltz,
AlphaFold 3, Chai-1, OpenFold), multi-structure dataset, and cloud streaming
(memory:// fsspec - no real cloud account required).
