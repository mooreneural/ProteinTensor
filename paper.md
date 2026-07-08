---
title: 'ProteinTensor: AI-Native Biomolecular Tensor Storage for Structural Biology ML'
tags:
  - Python
  - structural biology
  - machine learning
  - protein structure
  - tensor format
authors:
  - name: Clayton W. Moore
    orcid: 0009-0001-1033-6320
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 3 June 2026
bibliography: paper.bib
archive_doi: https://doi.org/10.5281/zenodo.21150065
---

# Summary

ProteinTensor (distributed as the `proteintensor` Python package, hosted
at github.com/mooreneural/ProteinTensor) is a Python library and file format
(`.ptt`) for storing
pre-computed tensors required by structural biology machine learning models.
Rather than re-parsing text-based structure files at every training epoch,
ProteinTensor converts each PDB/mmCIF structure once into a Zarr-backed,
LZ4-compressed, memory-mappable directory store, then loads all tensors in
milliseconds on every subsequent run. The format natively encodes atomic
coordinates, backbone geometry, covalent bond graphs, multiple sequence
alignments (MSAs), pairwise distance features, protein language model (PLM)
embeddings, and arbitrary named pair tensors. A multi-structure container
extends the single-structure format to full training datasets compatible
with PyTorch `DataLoader`. Cloud streaming over S3 and GCS is supported
without local downloads via `fsspec` [@zarr].

# Statement of Need

Modern protein structure prediction and design models - AlphaFold2
[@jumper2021alphafold], AlphaFold3 [@abramson2024alphafold3], Boltz
[@wohlwend2024boltz], RoseTTAFold [@baek2021rosettafold], and OpenFold
[@ahdritz2024openfold] - require a sequence of deterministic preprocessing
steps before any forward pass: parsing mmCIF files [@wwpdb2019], extracting
residue and atom arrays, constructing backbone coordinate tensors, computing
covalent bond graphs, searching MSAs with JackHMMER [@eddy2011hmmer] or
ColabFold [@mirdita2022colabfold], and running ESM2 [@lin2023esm2]
inference. Every step produces the same output for a given structure, yet
nearly every training framework repeats all of them from scratch at each
epoch. For a 100,000-structure training run this overhead reaches
approximately 15 hours of CPU time per epoch for full feature assembly.

No existing format fills this niche completely. The mmCIF and PDB formats
[@wwpdb2019] are text-based and require expensive parsing via libraries such
as gemmi [@wojdyr2022gemmi] (7-352 ms per structure depending on size).
General-purpose binary stores such as HDF5 [@hdf5] provide chunked storage
but lack a schema tailored to protein ML tensors, provenance tracking for
MSA and embedding sources, or the lazy access patterns needed for large
training sets. The safetensors format stores flat weight tensors without
structural metadata. ProteinTensor provides a purpose-built, versioned
schema with typed sub-groups for each feature class, per-group provenance
metadata (tool version, database date, sequence SHA-256), lazy memory-mapped
access via Zarr [@zarr], and first-class PyTorch [@paszke2019pytorch] and
JAX integration through NumPy [@harris2020numpy] array interoperability.

# Design and Implementation

The `.ptt` format is a Zarr v2 directory store subdivided into named
sub-groups: `sequence`, `atoms`, `backbone`, `bonds`, `msa/<source>`,
`pairs/<name>`, and `embeddings/<model>`. Each sub-group carries a
`.zattrs` metadata document that records format version and
feature-specific provenance. Multiple MSA sources (e.g., UniRef90,
ColabFold) and multiple embedding models (e.g., ESM2-650M, ESM3) may
coexist in a single file without conflict.

The Python API provides two access modes. The full-load path
(`pt.read()`, `pt.read_backbone()`, `pt.read_msa()`) deserializes
arrays eagerly into NumPy arrays. The lazy path
(`pt.mmap_positions()`, `pt.mmap_backbone()`, `pt.mmap_msa_tokens()`,
`pt.mmap_embedding()`) returns Zarr arrays that support zero-copy
slicing - a 64-residue window or the first 100 MSA sequences can be
extracted without loading the full tensor. A `ProteinDataset` class wraps
multi-structure containers as a PyTorch-compatible dataset with a
`collate()` function that pads variable-length sequences to uniform batch
tensors. Integration with Boltz [@wohlwend2024boltz] is provided through
`BoltzAdapter`, which translates the `.ptt` feature layout into the input
dictionary expected by the Boltz forward pass, enabling end-to-end
structure prediction directly from cached tensors.

# Performance

Benchmarks were conducted on six representative structures spanning 76 to
3,525 residues on an NVIDIA RTX 5080 (CUDA 12.8, Python 3.11), taking the
median of 30 rounds. Per-structure load times for the full tensor set range
from 2.8 to 3.3 ms, compared to 7.2 to 352.4 ms for mmCIF parsing via
gemmi [@wojdyr2022gemmi], yielding speedups of 3x to 108x. Full feature
assembly - sequence tokens, MSA, Ca-Ca distance matrix, and ESM2 embedding
combined - averages approximately 4x faster than the traditional pipeline
across all six structures, measured against a vectorized A3M parser so the
baseline is fair. At dataset scale, loading 100,000 structures per epoch falls
from approximately 3.7 hours (mmCIF parse only) to approximately 5 minutes,
and full feature assembly drops from approximately 15 hours to
approximately 3.6 hours. The largest single gain is MSA caching: generating
MSAs with JackHMMER [@eddy2011hmmer] costs approximately 4,000 CPU-hours
for 100,000 proteins; ProteinTensor reduces all subsequent loads to
approximately 2.2 hours by storing pre-tokenized MSA tensors and per-column
profiles. The disk tradeoff is deliberate - a full-featured `.ptt` file
(8,192-sequence MSA, distance matrix, float16 ESM2 embedding) is on average
23x larger than the source mmCIF, paying disk space once to avoid paying
compute on every training run.

# Acknowledgements

The author thanks the developers of Zarr [@zarr], gemmi [@wojdyr2022gemmi],
and the Boltz team [@wohlwend2024boltz] for the open-source foundations on
which ProteinTensor is built.

# References
