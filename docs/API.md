# ProteinTensor Python API

Full reference for the `proteintensor` package. For install and a short tour, see
the [README](../README.md).

```python
import proteintensor as pt
```

## Structure

```python
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
```

## MSA

```python
# Read a cached MSA
msa = pt.read_msa("1abc.ptt", source="uniref90")
msa.tokens.shape    # (N_seq, N_res)  int32
msa.profile.shape   # (N_res, 23)    float32

# Cache an MSA (after running JackHMMER / ColabFold)
msa = pt.from_a3m("1abc_uniref90.a3m",
                  tool="jackhammer", tool_version="3.3.2",
                  database="uniref90", database_date="2024-01")
pt.add_msa("1abc.ptt", msa, source="uniref90")
```

## Pair features

```python
pt.compute_and_store_distances("1abc.ptt")       # Ca-Ca distance matrix
pt.compute_and_store_contacts("1abc.ptt", threshold=8.0)

dist = pt.read_pair_feature("1abc.ptt", "distance_matrix")
dist.data.shape     # (N_res, N_res, 1)  float32

# Store arbitrary pair tensors (template features, MSA covariance, ...)
pt.add_pair_feature("1abc.ptt", my_array, name="template_pair",
                    symmetric=False, dtype="float16")
```

### Sparse pair features (radius graph)

`O(N*k)` storage instead of dense `O(N^2)`. Real proteins have local contact
structure, so keeping only pairs within a cutoff shrinks the Ca-Ca distance
matrix 2x (small) to ~76x (3,525-residue enzyme) on disk, losslessly within the
cutoff. See [`benchmarks/SPARSE_PAIRS_RESULTS.md`](../benchmarks/SPARSE_PAIRS_RESULTS.md).

```python
pt.compute_and_store_distances_sparse("1abc.ptt", cutoff=15.0)  # keep pairs <= 15 A
sp = pt.read_pair_feature_sparse("1abc.ptt", "distance_matrix")
sp.nnz, sp.density         # kept entries; fraction of N^2 retained
dense = sp.to_dense()      # rehydrate to [N_res, N_res, 1] for dense-only adapters
```

## PLM embeddings

```python
# Read a cached embedding
emb = pt.read_embedding("1abc.ptt", "esm2_t33_650M_UR50D")
emb.data.shape      # (N_res, 1280)  float32  (upcast from float16 on load)

# Cache ESM2/ESM3 embeddings (after GPU inference)
pt.add_embedding("1abc.ptt", esm_representations,
                 model="esm2_t33_650M_UR50D", layer=-1, dtype="float16",
                 sequence_hash=pt.embedding_sequence_hash(data.sequence_tokens))
```

## Ligands / small molecules

```python
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
```

## Lazy / zero-copy access

```python
positions = pt.mmap_positions("1abc.ptt")       # zarr.Array - no full load
backbone  = pt.mmap_backbone("1abc.ptt")        # [N_res, 4, 3]
msa_lazy  = pt.mmap_msa_tokens("1abc.ptt", "uniref90")  # [N_seq, N_res]
emb_lazy  = pt.mmap_embedding("1abc.ptt", "esm2_t33_650M_UR50D")

# Slice without loading the full tensor
ca_window = backbone[100:164, 1, :]             # 64 Ca positions
top_100   = msa_lazy[:100, :]                   # first 100 MSA sequences
```

## PyTorch / JAX

```python
import torch
data   = pt.read("1abc.ptt")
coords = torch.from_numpy(data.atom_positions)   # (N_atoms, 3)
tokens = torch.from_numpy(data.sequence_tokens)  # (N_res,)

import jax.numpy as jnp
coords = jnp.array(data.atom_positions)
```

## Cloud streaming

```python
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
```

## Multi-structure dataset

```python
from pathlib import Path

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

## Model adapters

```python
from proteintensor import BoltzAdapter, AlphaFold3Adapter, ChaiAdapter, OpenFoldAdapter

# Boltz - end-to-end prediction from a .ptt (verified)
BoltzAdapter("1abc.ptt").predict("boltz_out/", model="boltz2", accelerator="gpu")

# Input generation for other models (.ptt -> native input files)
AlphaFold3Adapter("1abc.ptt").write_input("af3/1abc.json")   # AF3 fold_input JSON
ChaiAdapter("1abc.ptt").write_input("chai/")                 # Chai FASTA (+ A3M)
OpenFoldAdapter("1abc.ptt").write_input("openfold/")         # FASTA + alignments/
```
