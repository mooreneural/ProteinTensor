from __future__ import annotations
import numpy as np
import zarr
from pathlib import Path

from .schema import ProteinTensorData, BackboneData, BondData
from .msa import MsaData
from .pairs import PairFeature
from .embeddings import EmbeddingData
from .ligands import deserialize_ligands
from .remote import open_store


def read(
    path: str | Path,
    storage_options: dict | None = None,
) -> ProteinTensorData:
    """Load a .ptt file fully into memory.

    Parameters
    ----------
    path             Local path or fsspec URL (s3://, gs://, memory://, ...).
    storage_options  fsspec kwargs forwarded for remote paths (credentials, etc.).
    """
    store = open_store(path, storage_options=storage_options)
    attrs = dict(store.attrs)

    has_atoms      = "atoms" in store
    has_struct     = "structure" in store
    bb_positions   = store["backbone/positions"][:] if "backbone" in store else None
    bb_mask        = store["backbone/mask"][:]      if "backbone" in store else None
    bond_edge_idx  = store["bonds/edge_index"][:]   if "bonds"    in store else None
    bond_edge_type = store["bonds/edge_type"][:]    if "bonds"    in store else None

    return ProteinTensorData(
        sequence_tokens=store["sequence/tokens"][:],
        residue_index=store["sequence/residue_index"][:],
        chain_id=store["sequence/chain_id"][:],
        atom_positions=store["atoms/positions"][:]  if has_atoms  else None,
        atom_mask=store["atoms/mask"][:]            if has_atoms  else None,
        b_factors=store["atoms/b_factors"][:]       if has_atoms  else None,
        residue_atom_start=store["structure/residue_atom_start"][:] if has_struct else None,
        residue_atom_count=store["structure/residue_atom_count"][:] if has_struct else None,
        backbone_positions=bb_positions,
        backbone_mask=bb_mask,
        bond_edge_index=bond_edge_idx,
        bond_edge_type=bond_edge_type,
        ligands=deserialize_ligands(store),
        pdb_id=attrs.get("pdb_id", ""),
        resolution=float(attrs["resolution"]) if attrs.get("resolution") is not None else float("nan"),
        method=attrs.get("method", ""),
        deposition_date=attrs.get("deposition_date", ""),
    )


def mmap_positions(
    path: str | Path,
    storage_options: dict | None = None,
) -> zarr.Array:
    """Return a lazy (zero-copy) view of atom positions without loading the full file."""
    return open_store(path, storage_options=storage_options)["atoms/positions"]


def mmap_tokens(
    path: str | Path,
    storage_options: dict | None = None,
) -> zarr.Array:
    """Return a lazy view of sequence tokens."""
    return open_store(path, storage_options=storage_options)["sequence/tokens"]


def read_backbone(
    path: str | Path,
    storage_options: dict | None = None,
) -> BackboneData:
    """Load only backbone coordinates and sequence - skips all heavy-atom data."""
    store = open_store(path, storage_options=storage_options)
    if "backbone" not in store:
        raise KeyError(f"No backbone group in {path}. Re-convert with proteintensor>=0.2.")
    return BackboneData(
        positions=store["backbone/positions"][:],
        mask=store["backbone/mask"][:],
        sequence_tokens=store["sequence/tokens"][:],
        residue_index=store["sequence/residue_index"][:],
        chain_id=store["sequence/chain_id"][:],
    )


def read_bonds(
    path: str | Path,
    storage_options: dict | None = None,
) -> BondData:
    """Load only the bond graph - skips coordinates, sequence, and backbone."""
    store = open_store(path, storage_options=storage_options)
    if "bonds" not in store:
        raise KeyError(f"No bonds group in {path}. Re-convert with proteintensor>=0.3.")
    return BondData(
        edge_index=store["bonds/edge_index"][:],
        edge_type=store["bonds/edge_type"][:],
        num_atoms=int(store.attrs.get("num_atoms", 0)),
    )


def mmap_backbone(
    path: str | Path,
    storage_options: dict | None = None,
) -> zarr.Array:
    """Return a lazy [N_res, 4, 3] view of backbone positions."""
    store = open_store(path, storage_options=storage_options)
    if "backbone" not in store:
        raise KeyError(f"No backbone group in {path}. Re-convert with proteintensor>=0.2.")
    return store["backbone/positions"]


def list_msas(
    path: str | Path,
    storage_options: dict | None = None,
) -> list[str]:
    """Return the list of MSA source names stored in a .ptt file."""
    store = open_store(path, storage_options=storage_options)
    if "msa" not in store:
        return []
    return list(store["msa"].keys())


def read_msa(
    path: str | Path,
    source: str = "default",
    storage_options: dict | None = None,
) -> MsaData:
    """Load the MSA for one source fully into memory."""
    store = open_store(path, storage_options=storage_options)
    _require_msa_source(store, path, source)
    grp   = store[f"msa/{source}"]
    attrs = dict(grp.attrs)
    return MsaData(
        tokens=grp["tokens"][:],
        deletion_matrix=grp["deletion_matrix"][:],
        profile=grp["profile"][:],
        deletion_mean=grp["deletion_mean"][:],
        sequence_hash=attrs.get("sequence_hash", ""),
        tool=attrs.get("tool", ""),
        tool_version=attrs.get("tool_version", ""),
        database=attrs.get("database", ""),
        database_date=attrs.get("database_date", ""),
        created_at=attrs.get("created_at", 0.0),
    )


def mmap_msa_tokens(
    path: str | Path,
    source: str = "default",
    storage_options: dict | None = None,
) -> zarr.Array:
    """Return a lazy [N_seq, N_res] view of MSA tokens - no full load."""
    store = open_store(path, storage_options=storage_options)
    _require_msa_source(store, path, source)
    return store[f"msa/{source}/tokens"]


def list_pair_features(
    path: str | Path,
    storage_options: dict | None = None,
) -> list[str]:
    """Return the names of all pair feature tensors stored in a .ptt file."""
    store = open_store(path, storage_options=storage_options)
    if "pairs" not in store:
        return []
    return list(store["pairs"].keys())


def read_pair_feature(
    path: str | Path,
    name: str,
    storage_options: dict | None = None,
) -> PairFeature:
    """Load a named pair feature tensor fully into memory.

    Returns a PairFeature whose .data has shape [N_res, N_res, C].
    For single-channel features (C=1) you can index data[:, :, 0] directly.
    """
    store = open_store(path, storage_options=storage_options)
    _require_pair(store, path, name)
    grp   = store[f"pairs/{name}"]
    attrs = dict(grp.attrs)
    return PairFeature(
        data=grp["data"][:],
        name=name,
        channels=int(attrs.get("channels", 1)),
        symmetric=bool(attrs.get("symmetric", False)),
        description=attrs.get("description", ""),
        dtype=attrs.get("dtype", "float32"),
        created_at=float(attrs.get("created_at", 0.0)),
    )


def mmap_pair_feature(
    path: str | Path,
    name: str,
    storage_options: dict | None = None,
) -> zarr.Array:
    """Return a lazy [N_res, N_res, C] view - slice without loading the full tensor."""
    store = open_store(path, storage_options=storage_options)
    _require_pair(store, path, name)
    return store[f"pairs/{name}/data"]


def list_embeddings(
    path: str | Path,
    storage_options: dict | None = None,
) -> list[str]:
    """Return model names of all embeddings stored in a .ptt file."""
    store = open_store(path, storage_options=storage_options)
    if "embeddings" not in store:
        return []
    return list(store["embeddings"].keys())


def read_embedding(
    path: str | Path,
    model: str,
    storage_options: dict | None = None,
) -> EmbeddingData:
    """Load a named PLM embedding fully into memory as float32."""
    store = open_store(path, storage_options=storage_options)
    _require_embedding(store, path, model)
    grp   = store[f"embeddings/{model}"]
    attrs = dict(grp.attrs)
    return EmbeddingData(
        data=grp["data"][:].astype(np.float32),
        model=model,
        layer=int(attrs.get("layer", -1)),
        dim=int(attrs.get("dim", grp["data"].shape[1])),
        dtype=attrs.get("dtype", "float32"),
        sequence_hash=attrs.get("sequence_hash", ""),
        created_at=float(attrs.get("created_at", 0.0)),
    )


def mmap_embedding(
    path: str | Path,
    model: str,
    storage_options: dict | None = None,
) -> zarr.Array:
    """Return a lazy [N_res, D] view of an embedding - no full load."""
    store = open_store(path, storage_options=storage_options)
    _require_embedding(store, path, model)
    return store[f"embeddings/{model}/data"]


def _require_embedding(store: zarr.Group, path, model: str) -> None:
    if "embeddings" not in store or model not in store["embeddings"]:
        available = list(store["embeddings"].keys()) if "embeddings" in store else []
        raise KeyError(
            f"Embedding '{model}' not found in {path}. "
            f"Available: {available or '(none)'}"
        )


def _require_pair(store: zarr.Group, path, name: str) -> None:
    if "pairs" not in store or name not in store["pairs"]:
        available = list(store["pairs"].keys()) if "pairs" in store else []
        raise KeyError(
            f"Pair feature '{name}' not found in {path}. "
            f"Available: {available or '(none)'}"
        )


def _require_msa_source(store: zarr.Group, path, source: str) -> None:
    if "msa" not in store or source not in store["msa"]:
        available = list(store["msa"].keys()) if "msa" in store else []
        raise KeyError(
            f"MSA source '{source}' not found in {path}. "
            f"Available: {available or '(none)'}"
        )
