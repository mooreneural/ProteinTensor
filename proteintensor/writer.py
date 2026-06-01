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


def add_msa(
    path: str | Path,
    msa: "MsaData",
    source: str = "default",
    compression: str = "blosc",
    overwrite: bool = False,
) -> None:
    """Append MSA data to an existing .ptt file without touching structure data.

    Parameters
    ----------
    path        Path to an existing .ptt Zarr store.
    msa         MsaData object (from from_a3m() or constructed directly).
    source      Name for this MSA source, e.g. "uniref90", "bfd", "colabfold".
                Multiple sources can coexist; each gets its own sub-group.
    overwrite   If False (default) raise if this source already exists.
    """
    from .msa import MsaData  # local import avoids circular dependency at module load

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run proteintensor convert first.")

    store = zarr.open(str(path), mode="r+")
    msa_root = store.require_group("msa")

    if source in msa_root and not overwrite:
        raise ValueError(
            f"MSA source '{source}' already exists in {path}. "
            "Pass overwrite=True to replace it."
        )

    compressor = _compressor(compression)
    grp = msa_root.require_group(source)

    N_seq, N_res = msa.tokens.shape
    chunk_seq = min(256, N_seq)
    chunk_res = min(256, N_res)

    grp.create_dataset("tokens",          data=msa.tokens,          dtype="int32",
                       chunks=(chunk_seq, chunk_res), compressor=compressor, overwrite=True)
    grp.create_dataset("deletion_matrix", data=msa.deletion_matrix, dtype="float32",
                       chunks=(chunk_seq, chunk_res), compressor=compressor, overwrite=True)
    grp.create_dataset("profile",         data=msa.profile,         dtype="float32",
                       compressor=compressor, overwrite=True)
    grp.create_dataset("deletion_mean",   data=msa.deletion_mean,   dtype="float32",
                       compressor=compressor, overwrite=True)

    grp.attrs.update({
        "num_sequences":   N_seq,
        "num_residues":    N_res,
        "sequence_hash":   msa.sequence_hash,
        "tool":            msa.tool,
        "tool_version":    msa.tool_version,
        "database":        msa.database,
        "database_date":   msa.database_date,
        "created_at":      msa.created_at,
    })

    # Update root-level msa source list
    existing = list(store.attrs.get("msa_sources", []))
    if source not in existing:
        existing.append(source)
    store.attrs["msa_sources"] = existing


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
