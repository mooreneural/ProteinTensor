from __future__ import annotations
import numpy as np
import zarr
from pathlib import Path

from .schema import ProteinTensorData, BackboneData, BondData


def read(path: str | Path) -> ProteinTensorData:
    """Load a .ptt file fully into memory."""
    store = zarr.open(str(path), mode="r")
    attrs = dict(store.attrs)

    bb_positions   = store["backbone/positions"][:] if "backbone" in store else None
    bb_mask        = store["backbone/mask"][:]      if "backbone" in store else None
    bond_edge_idx  = store["bonds/edge_index"][:]   if "bonds"    in store else None
    bond_edge_type = store["bonds/edge_type"][:]    if "bonds"    in store else None

    return ProteinTensorData(
        sequence_tokens=store["sequence/tokens"][:],
        residue_index=store["sequence/residue_index"][:],
        chain_id=store["sequence/chain_id"][:],
        atom_positions=store["atoms/positions"][:],
        atom_mask=store["atoms/mask"][:],
        b_factors=store["atoms/b_factors"][:],
        residue_atom_start=store["structure/residue_atom_start"][:],
        residue_atom_count=store["structure/residue_atom_count"][:],
        backbone_positions=bb_positions,
        backbone_mask=bb_mask,
        bond_edge_index=bond_edge_idx,
        bond_edge_type=bond_edge_type,
        pdb_id=attrs.get("pdb_id", ""),
        resolution=float(attrs["resolution"]) if attrs.get("resolution") is not None else float("nan"),
        method=attrs.get("method", ""),
        deposition_date=attrs.get("deposition_date", ""),
    )


def mmap_positions(path: str | Path) -> zarr.Array:
    """Return a lazy (zero-copy) view of atom positions without loading the full file."""
    return zarr.open(str(path), mode="r")["atoms/positions"]


def mmap_tokens(path: str | Path) -> zarr.Array:
    """Return a lazy view of sequence tokens."""
    return zarr.open(str(path), mode="r")["sequence/tokens"]


def read_backbone(path: str | Path) -> BackboneData:
    """Load only backbone coordinates and sequence — skips all heavy-atom data."""
    store = zarr.open(str(path), mode="r")
    if "backbone" not in store:
        raise KeyError(f"No backbone group in {path}. Re-convert with proteintensor>=0.2.")
    return BackboneData(
        positions=store["backbone/positions"][:],
        mask=store["backbone/mask"][:],
        sequence_tokens=store["sequence/tokens"][:],
        residue_index=store["sequence/residue_index"][:],
        chain_id=store["sequence/chain_id"][:],
    )


def read_bonds(path: str | Path) -> BondData:
    """Load only the bond graph — skips coordinates, sequence, and backbone."""
    store = zarr.open(str(path), mode="r")
    if "bonds" not in store:
        raise KeyError(f"No bonds group in {path}. Re-convert with proteintensor>=0.3.")
    return BondData(
        edge_index=store["bonds/edge_index"][:],
        edge_type=store["bonds/edge_type"][:],
        num_atoms=int(store.attrs.get("num_atoms", 0)),
    )


def mmap_backbone(path: str | Path) -> zarr.Array:
    """Return a lazy [N_res, 4, 3] view of backbone positions."""
    store = zarr.open(str(path), mode="r")
    if "backbone" not in store:
        raise KeyError(f"No backbone group in {path}. Re-convert with proteintensor>=0.2.")
    return store["backbone/positions"]
