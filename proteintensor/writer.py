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
        "num_atoms": int(data.atom_positions.shape[0]) if data.has_structure else 0,
        "has_structure": data.has_structure,
    })

    seq = store.require_group("sequence")
    _arr(seq, "tokens",        data.sequence_tokens,    "int32",   compressor)
    _arr(seq, "residue_index", data.residue_index,      "int32",   compressor)
    _arr(seq, "chain_id",      data.chain_id,           "S1",      compressor)

    # Atom-level and residue->atom mapping are omitted for sequence-only entries.
    if data.has_structure:
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

    if data.ligands:
        from .ligands import serialize_ligands
        serialize_ligands(store, data.ligands, compressor)


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


def add_pair_feature(
    path: str | Path,
    data: np.ndarray,
    name: str,
    *,
    symmetric: bool = False,
    description: str = "",
    dtype: str = "float32",
    compression: str = "blosc",
    overwrite: bool = False,
) -> None:
    """Append a named pairwise feature tensor to an existing .ptt file.

    Parameters
    ----------
    path        Path to an existing .ptt Zarr store.
    data        [N_res, N_res] or [N_res, N_res, C] array.
                Single-channel inputs are automatically expanded to [..., 1].
    name        Feature name, e.g. "distance_matrix", "contacts", "template_pair".
    symmetric   Hint: True if data[i,j] == data[j,i] (stored full, used by readers).
    description Human-readable description stored in metadata.
    dtype       Target dtype for storage (default "float32"). Use "bool" for contacts,
                "float16" to halve memory for large multi-channel features.
    compression Zarr compressor ("blosc" or "none").
    overwrite   Replace existing feature if present (default False).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    if data.ndim != 3 or data.shape[0] != data.shape[1]:
        raise ValueError(
            f"data must be [N, N] or [N, N, C], got shape {data.shape}"
        )

    N, _, C = data.shape
    store = zarr.open(str(path), mode="r+")
    pairs_root = store.require_group("pairs")

    if name in pairs_root and not overwrite:
        raise ValueError(
            f"Pair feature '{name}' already exists in {path}. "
            "Pass overwrite=True to replace it."
        )

    compressor = _compressor(compression)
    chunk = min(128, N)
    grp = pairs_root.require_group(name)
    grp.create_dataset(
        "data",
        data=data.astype(dtype),
        dtype=dtype,
        chunks=(chunk, chunk, C),
        compressor=compressor,
        overwrite=True,
    )
    grp.attrs.update({
        "channels":    C,
        "n_residues":  N,
        "symmetric":   symmetric,
        "description": description,
        "dtype":       dtype,
        "created_at":  time.time(),
    })

    # Keep a root-level index
    existing = list(store.attrs.get("pair_features", []))
    if name not in existing:
        existing.append(name)
    store.attrs["pair_features"] = existing


def add_embedding(
    path: str | Path,
    data: np.ndarray,
    model: str,
    *,
    layer: int = -1,
    dtype: str = "float16",
    sequence_hash: str = "",
    compression: str = "blosc",
    overwrite: bool = False,
) -> None:
    """Append a per-residue PLM embedding to an existing .ptt file.

    Parameters
    ----------
    path            Path to an existing .ptt Zarr store.
    data            float array [N_res, D].
    model           Model identifier, e.g. "esm2_t33_650M_UR50D".
    layer           Source layer index (-1 = final layer, the default).
    dtype           Storage dtype.  "float16" halves disk/memory vs float32.
    sequence_hash   SHA-256 of the input sequence tokens for cache validation.
                    Use proteintensor.embeddings.sequence_hash(tokens) to compute.
    overwrite       Replace existing embedding for this model if present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    if data.ndim != 2:
        raise ValueError(f"data must be [N_res, D], got shape {data.shape}")

    N_res, D = data.shape
    store = zarr.open(str(path), mode="r+")
    emb_root = store.require_group("embeddings")

    if model in emb_root and not overwrite:
        raise ValueError(
            f"Embedding '{model}' already exists in {path}. "
            "Pass overwrite=True to replace it."
        )

    compressor = _compressor(compression)
    grp = emb_root.require_group(model)
    grp.create_dataset(
        "data",
        data=data.astype(dtype),
        dtype=dtype,
        chunks=(min(256, N_res), D),
        compressor=compressor,
        overwrite=True,
    )
    grp.attrs.update({
        "model":          model,
        "layer":          layer,
        "dim":            D,
        "n_residues":     N_res,
        "dtype":          dtype,
        "sequence_hash":  sequence_hash,
        "created_at":     time.time(),
    })

    existing = list(store.attrs.get("embeddings", []))
    if model not in existing:
        existing.append(model)
    store.attrs["embeddings"] = existing


def compute_and_store_distances(
    path: str | Path,
    *,
    overwrite: bool = False,
    compression: str = "blosc",
) -> None:
    """Compute Ca-Ca pairwise distance matrix and store as 'distance_matrix'.

    Requires backbone data (written by convert). Result is float32 [N_res, N_res, 1].
    """
    from .pairs import compute_distance_matrix
    path = Path(path)
    store = zarr.open(str(path), mode="r")
    if "backbone" not in store:
        raise KeyError("No backbone group found. Re-convert with proteintensor>=0.2.")
    bb = store["backbone/positions"][:]
    dist = compute_distance_matrix(bb)
    add_pair_feature(
        path, dist, name="distance_matrix",
        symmetric=True,
        description="Ca-Ca pairwise Euclidean distances in Angstroms",
        dtype="float32",
        compression=compression,
        overwrite=overwrite,
    )


def compute_and_store_contacts(
    path: str | Path,
    *,
    threshold: float = 8.0,
    overwrite: bool = False,
    compression: str = "blosc",
) -> None:
    """Compute binary Ca contact map and store as 'contacts'.

    Contacts are defined as Ca-Ca distance < threshold (default 8.0 A).
    Requires backbone data. Result is bool [N_res, N_res, 1].
    """
    from .pairs import compute_contact_map, compute_distance_matrix
    path = Path(path)
    store = zarr.open(str(path), mode="r")
    if "backbone" not in store:
        raise KeyError("No backbone group found. Re-convert with proteintensor>=0.2.")
    bb   = store["backbone/positions"][:]
    dist = compute_distance_matrix(bb)
    contacts = compute_contact_map(dist, threshold=threshold)
    add_pair_feature(
        path, contacts, name="contacts",
        symmetric=True,
        description=f"Ca-Ca contacts: distance < {threshold} A",
        dtype="bool",
        compression=compression,
        overwrite=overwrite,
    )


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
