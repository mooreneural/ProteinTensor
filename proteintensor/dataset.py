"""
Multi-structure dataset container for ProteinTensor.

Stores N protein structures in a single Zarr directory store under
structures/<key>/. Each sub-group has the same layout as a standalone .ptt file,
so all existing reader helpers work on sliced sub-groups.

API
---
  create_dataset       initialize an empty dataset store
  add_to_dataset       copy one .ptt into the dataset
  ProteinDataset       PyTorch-compatible Dataset for reading

Usage
-----
  create_dataset("training.ptt")
  for p in Path("ptt_files").glob("*.ptt"):
      add_to_dataset("training.ptt", p)

  ds = ProteinDataset("training.ptt")
  data = ds[0]          # ProteinTensorData
  data = ds["1ABC"]     # look up by PDB ID (case-insensitive)

  # PyTorch DataLoader (pass collate_fn to handle variable-length sequences)
  from torch.utils.data import DataLoader
  loader = DataLoader(ds, batch_size=8, collate_fn=ProteinDataset.collate)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np
import zarr

from .schema import ProteinTensorData, FORMAT_VERSION
from .ligands import deserialize_ligands
from .remote import open_store, _is_url


def create_dataset(path: str | Path, overwrite: bool = False) -> None:
    """Initialize an empty multi-structure dataset store at path.

    Parameters
    ----------
    path        Location for the new Zarr directory store.
    overwrite   If True, delete and recreate an existing store.
    """
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Dataset already exists: {path}. Use overwrite=True to replace."
            )
        import shutil
        shutil.rmtree(path)

    store = zarr.open(str(path), mode="w")
    store.attrs.update({
        "format": "proteintensor-dataset",
        "version": FORMAT_VERSION,
        "created_at": time.time(),
        "num_structures": 0,
    })
    store.require_group("structures")


def add_to_dataset(
    dataset_path: str | Path,
    ptt_path: str | Path,
    *,
    key: str | None = None,
) -> str:
    """Copy a .ptt structure into a dataset store.

    The entire source Zarr tree (sequence, atoms, backbone, bonds, msa,
    pairs, embeddings) is copied verbatim under structures/<key>/.

    Parameters
    ----------
    dataset_path  Path to an existing dataset created with create_dataset().
    ptt_path      Single-structure .ptt file to add.
    key           Storage key. Auto-assigned as a zero-padded integer if None.

    Returns
    -------
    The key under which the structure was stored.
    """
    dst = zarr.open(str(dataset_path), mode="a")
    src = zarr.open(str(ptt_path), mode="r")

    if dst.attrs.get("format") != "proteintensor-dataset":
        raise ValueError(
            f"{dataset_path} is not a ProteinTensor dataset. "
            "Create one with create_dataset() first."
        )

    n = int(dst.attrs.get("num_structures", 0))
    if key is None:
        key = f"{n:06d}"

    structs = dst["structures"]
    if key in structs:
        raise KeyError(f"Key {key!r} already exists in the dataset.")

    zarr.copy(src, structs, name=key)
    dst.attrs["num_structures"] = n + 1
    return key


class ProteinDataset:
    """Multi-structure dataset implementing the PyTorch Dataset protocol.

    Parameters
    ----------
    path    Path to a dataset created with create_dataset() + add_to_dataset().

    Examples
    --------
    ds = ProteinDataset("training.ptt")
    data = ds[0]          # first structure as ProteinTensorData
    data = ds["1ABC"]     # look up by PDB ID (case-insensitive)
    len(ds)               # number of structures

    loader = DataLoader(ds, batch_size=8, collate_fn=ProteinDataset.collate)

    # Remote dataset (S3, GCS, etc.)
    ds = ProteinDataset("s3://my-bucket/training.ptt")
    ds = ProteinDataset("s3://my-bucket/training.ptt",
                        storage_options={"key": "...", "secret": "..."})
    """

    def __init__(
        self,
        path: str | Path,
        storage_options: dict | None = None,
    ) -> None:
        self.path = path if _is_url(str(path)) else Path(path)
        if not _is_url(str(path)) and not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        self._store = open_store(path, storage_options=storage_options)
        if self._store.attrs.get("format") != "proteintensor-dataset":
            raise ValueError(
                f"{path} is not a ProteinTensor dataset. "
                "Use create_dataset() to create one."
            )
        self._keys: list[str] = sorted(self._store["structures"].keys())
        self._pdb_to_key: dict[str, str] = {}
        for k in self._keys:
            pdb_id = str(
                self._store[f"structures/{k}"].attrs.get("pdb_id", "")
            ).upper()
            if pdb_id:
                self._pdb_to_key[pdb_id] = k

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int | str) -> ProteinTensorData:
        """Return one structure.

        idx : int   zero-based index (negative indices supported)
        idx : str   PDB ID (case-insensitive) or raw storage key
        """
        if isinstance(idx, int):
            if idx < 0:
                idx = len(self._keys) + idx
            if not (0 <= idx < len(self._keys)):
                raise IndexError(
                    f"Index {idx} out of range for dataset of size {len(self._keys)}"
                )
            key = self._keys[idx]
        else:
            upper = idx.upper()
            if upper in self._pdb_to_key:
                key = self._pdb_to_key[upper]
            elif idx in self._keys:
                key = idx
            else:
                sample = list(self._pdb_to_key.keys())[:5]
                raise KeyError(
                    f"Structure {idx!r} not found. "
                    f"Sample IDs: {sample}{'...' if len(self._pdb_to_key) > 5 else ''}"
                )
        return _read_group(self._store[f"structures/{key}"])

    def __iter__(self) -> Iterator[ProteinTensorData]:
        for k in self._keys:
            yield _read_group(self._store[f"structures/{k}"])

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def keys(self) -> list[str]:
        """Return all storage keys in sorted order."""
        return list(self._keys)

    def pdb_ids(self) -> list[str]:
        """Return PDB IDs for all structures (empty string where unknown)."""
        return [
            str(self._store[f"structures/{k}"].attrs.get("pdb_id", ""))
            for k in self._keys
        ]

    # ------------------------------------------------------------------
    # Collate for DataLoader
    # ------------------------------------------------------------------

    @staticmethod
    def collate(samples: list[ProteinTensorData]) -> dict[str, np.ndarray]:
        """Collate a list of ProteinTensorData into padded batch arrays.

        Sequences and atom arrays are padded to the max length in the batch.
        Sequence-only entries (has_structure == False) contribute zero atoms:
        their atom rows stay zero/False, so structure and sequence-only entries
        can be batched together. Use the ``has_structure`` mask to tell them apart.

        Returns a dict with keys:
          sequence_tokens    [B, max_res]        int32
          residue_index      [B, max_res]        int32,   -1 for padding
          chain_id           [B, max_res]        S1,      b'' for padding
          atom_positions     [B, max_atoms, 3]   float32, 0 for padding
          atom_mask          [B, max_atoms]      bool,    False for padding
          b_factors          [B, max_atoms]      float32, 0 for padding
          backbone_positions [B, max_res, 4, 3]  float32  (if all samples have it)
          backbone_mask      [B, max_res, 4]     bool     (if all samples have it)
          padding_mask       [B, max_res]        bool,    True = real residue
          n_residues         [B]                 int32
          n_atoms            [B]                 int32,   0 for sequence-only
          has_structure      [B]                 bool,    False for sequence-only
        """
        from .schema import AA_UNK

        B         = len(samples)
        max_res   = max(s.sequence_tokens.shape[0] for s in samples)
        n_res     = np.array([s.sequence_tokens.shape[0] for s in samples], dtype=np.int32)
        n_atoms   = np.array([s.atom_positions.shape[0] if s.has_structure else 0
                              for s in samples], dtype=np.int32)
        has_struct = np.array([s.has_structure for s in samples], dtype=bool)
        max_atoms = int(n_atoms.max()) if B else 0

        seq_tok   = np.full((B, max_res),        AA_UNK, dtype=np.int32)
        res_idx   = np.full((B, max_res),        -1,     dtype=np.int32)
        chain_id  = np.full((B, max_res),        b"",    dtype="S1")
        atom_pos  = np.zeros((B, max_atoms, 3),          dtype=np.float32)
        atom_mask = np.zeros((B, max_atoms),             dtype=bool)
        b_fac     = np.zeros((B, max_atoms),             dtype=np.float32)
        pad_mask  = np.zeros((B, max_res),               dtype=bool)

        for i, s in enumerate(samples):
            nr = n_res[i]
            seq_tok[i,  :nr]   = s.sequence_tokens
            res_idx[i,  :nr]   = s.residue_index
            chain_id[i, :nr]   = s.chain_id
            pad_mask[i, :nr]   = True
            if s.has_structure:
                na = n_atoms[i]
                atom_pos[i,  :na]  = s.atom_positions
                atom_mask[i, :na]  = s.atom_mask
                b_fac[i,     :na]  = s.b_factors

        batch: dict[str, np.ndarray] = {
            "sequence_tokens": seq_tok,
            "residue_index":   res_idx,
            "chain_id":        chain_id,
            "atom_positions":  atom_pos,
            "atom_mask":       atom_mask,
            "b_factors":       b_fac,
            "padding_mask":    pad_mask,
            "n_residues":      n_res,
            "n_atoms":         n_atoms,
            "has_structure":   has_struct,
        }

        if all(s.backbone_positions is not None for s in samples):
            bb_pos  = np.zeros((B, max_res, 4, 3), dtype=np.float32)
            bb_mask = np.zeros((B, max_res, 4),    dtype=bool)
            for i, s in enumerate(samples):
                nr = n_res[i]
                bb_pos[i,  :nr] = s.backbone_positions  # type: ignore[index]
                bb_mask[i, :nr] = s.backbone_mask        # type: ignore[index]
            batch["backbone_positions"] = bb_pos
            batch["backbone_mask"]      = bb_mask

        return batch


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _read_group(grp: zarr.Group) -> ProteinTensorData:
    """Read a ProteinTensorData from a sub-group with the same layout as a .ptt root."""
    attrs = dict(grp.attrs)

    has_atoms      = "atoms" in grp
    has_struct     = "structure" in grp
    bb_positions   = grp["backbone/positions"][:] if "backbone" in grp else None
    bb_mask        = grp["backbone/mask"][:]      if "backbone" in grp else None
    bond_edge_idx  = grp["bonds/edge_index"][:]   if "bonds"    in grp else None
    bond_edge_type = grp["bonds/edge_type"][:]    if "bonds"    in grp else None

    return ProteinTensorData(
        sequence_tokens=grp["sequence/tokens"][:],
        residue_index=grp["sequence/residue_index"][:],
        chain_id=grp["sequence/chain_id"][:],
        atom_positions=grp["atoms/positions"][:]  if has_atoms  else None,
        atom_mask=grp["atoms/mask"][:]            if has_atoms  else None,
        b_factors=grp["atoms/b_factors"][:]       if has_atoms  else None,
        residue_atom_start=grp["structure/residue_atom_start"][:] if has_struct else None,
        residue_atom_count=grp["structure/residue_atom_count"][:] if has_struct else None,
        backbone_positions=bb_positions,
        backbone_mask=bb_mask,
        bond_edge_index=bond_edge_idx,
        bond_edge_type=bond_edge_type,
        ligands=deserialize_ligands(grp),
        pdb_id=attrs.get("pdb_id", ""),
        resolution=float(attrs["resolution"]) if attrs.get("resolution") is not None else float("nan"),
        method=attrs.get("method", ""),
        deposition_date=attrs.get("deposition_date", ""),
    )
