from __future__ import annotations
import time
import numpy as np
import zarr
from pathlib import Path

from .schema import ProteinTensorData, FORMAT_VERSION


def write(data: ProteinTensorData, path: str | Path, compression: str = "blosc") -> None:
    """Write a ProteinTensorData to a .ptt Zarr directory store."""
    path = Path(path)
    store = zarr.open(str(path), mode="w")
    compressor = _compressor(compression)

    store.attrs.update({
        "format": "ProteinTensor",
        "version": FORMAT_VERSION,
        "pdb_id": data.pdb_id,
        "resolution": float(data.resolution) if data.resolution == data.resolution else None,
        "method": data.method,
        "deposition_date": data.deposition_date,
        "created_at": time.time(),
        "num_residues": int(data.sequence_tokens.shape[0]),
        "num_atoms": int(data.atom_positions.shape[0]),
    })

    seq = store.require_group("sequence")
    _arr(seq, "tokens",        data.sequence_tokens,    "int32",   compressor)
    _arr(seq, "residue_index", data.residue_index,      "int32",   compressor)
    _arr(seq, "chain_id",      data.chain_id,           "S1",      compressor)

    atoms = store.require_group("atoms")
    _arr(atoms, "positions",   data.atom_positions,     "float32", compressor)
    _arr(atoms, "mask",        data.atom_mask,          "bool",    compressor)
    _arr(atoms, "b_factors",   data.b_factors,          "float32", compressor)

    struct = store.require_group("structure")
    _arr(struct, "residue_atom_start", data.residue_atom_start, "int32", compressor)
    _arr(struct, "residue_atom_count", data.residue_atom_count, "int32", compressor)

    if data.backbone_positions is not None and data.backbone_mask is not None:
        bb = store.require_group("backbone")
        _arr(bb, "positions", data.backbone_positions, "float32", compressor)
        _arr(bb, "mask",      data.backbone_mask,      "bool",    compressor)
        store.attrs["has_backbone"] = True

    if data.bond_edge_index is not None and data.bond_edge_type is not None:
        bonds = store.require_group("bonds")
        _arr(bonds, "edge_index", data.bond_edge_index, "int32",  compressor)
        _arr(bonds, "edge_type",  data.bond_edge_type,  "uint8",  compressor)
        store.attrs["num_bonds"] = int(data.bond_edge_index.shape[1])


def _arr(group: zarr.Group, name: str, data: np.ndarray, dtype: str, compressor) -> None:
    group.create_dataset(name, data=data.astype(dtype), compressor=compressor, overwrite=True)


def _compressor(name: str):
    if name == "blosc":
        try:
            from numcodecs import Blosc
            return Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
        except ImportError:
            pass
    return None
