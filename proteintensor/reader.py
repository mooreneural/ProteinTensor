from __future__ import annotations
import numpy as np
import zarr
from pathlib import Path

from .schema import ProteinTensorData


def read(path: str | Path) -> ProteinTensorData:
    """Load a .ptt file fully into memory."""
    store = zarr.open(str(path), mode="r")
    attrs = dict(store.attrs)

    return ProteinTensorData(
        sequence_tokens=store["sequence/tokens"][:],
        residue_index=store["sequence/residue_index"][:],
        chain_id=store["sequence/chain_id"][:],
        atom_positions=store["atoms/positions"][:],
        atom_mask=store["atoms/mask"][:],
        b_factors=store["atoms/b_factors"][:],
        residue_atom_start=store["structure/residue_atom_start"][:],
        residue_atom_count=store["structure/residue_atom_count"][:],
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
