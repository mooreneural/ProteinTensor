"""
Pairwise feature storage for ProteinTensor.

Pair tensors are stored under pairs/<name>/ inside an existing .ptt file.
Multiple named features coexist (distance_matrix, contacts, template_pair, …).

Standard computed features
--------------------------
  compute_distance_matrix  Ca-Ca pairwise distances [N_res, N_res] float32
  compute_contact_map      binary contacts at a distance threshold [N_res, N_res] bool

Generic API
-----------
  add_pair_feature         store any [N_res, N_res] or [N_res, N_res, C] array
  read_pair_feature        load a named pair tensor
  list_pair_features       list stored names
  mmap_pair_feature        lazy zarr.Array - slice without full load

Size note
---------
A single float32 channel for a 3000-residue protein is ~36 MB.  Multi-channel
features (e.g. C=64 template features) can reach several GB.  Use float16 or
chunked lazy access (mmap_pair_feature) for large C.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class PairFeature:
    """A named pairwise feature tensor loaded from a .ptt file."""
    data:        np.ndarray   # [N_res, N_res, C]  (C=1 for scalar features)
    name:        str
    channels:    int          # C
    symmetric:   bool         # True if data[i,j] == data[j,i]
    description: str
    dtype:       str
    created_at:  float

    @property
    def num_residues(self) -> int:
        return int(self.data.shape[0])


@dataclass
class SparsePairFeature:
    """A sparse (COO) pairwise feature loaded from a .ptt file.

    Only the kept (i, j) entries are stored. For symmetric features only the
    upper triangle (i <= j) is stored and mirrored on densify, roughly halving
    the entry count. Everything else is implicitly ``fill_value``.
    """
    indices:     np.ndarray   # int32 [2, nnz]   (row, col)
    values:      np.ndarray   # [nnz, C]
    n_residues:  int
    channels:    int
    symmetric:   bool
    fill_value:  float
    mode:        str          # "radius" | "threshold" | "nonzero" | "mask"
    cutoff:      float        # radius/threshold used (nan if not applicable)
    dtype:       str
    description: str
    created_at:  float

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[1])

    @property
    def density(self) -> float:
        n = self.n_residues
        return self.nnz / float(n * n) if n else 0.0

    def to_dense(self) -> np.ndarray:
        """Rehydrate to a dense [N_res, N_res, C] array (the edge shim)."""
        n, c = self.n_residues, self.channels
        dense = np.full((n, n, c), self.fill_value, dtype=self.dtype)
        i, j = self.indices
        dense[i, j] = self.values
        if self.symmetric:
            dense[j, i] = self.values
        return dense


# ---------------------------------------------------------------------------
# Sparse (COO) encoding helpers
# ---------------------------------------------------------------------------

def _as_3d(data: np.ndarray) -> np.ndarray:
    return data[:, :, None] if data.ndim == 2 else data


def radius_mask(distance_matrix: np.ndarray, cutoff: float) -> np.ndarray:
    """Keep-mask for pairs within a distance cutoff (radius graph)."""
    return distance_matrix <= cutoff


def threshold_mask(data: np.ndarray, threshold: float) -> np.ndarray:
    """Keep-mask where any channel's magnitude is >= threshold."""
    d = _as_3d(data)
    return np.any(np.abs(d) >= threshold, axis=2)


def nonzero_mask(data: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Keep-mask where any channel differs from fill_value."""
    d = _as_3d(data)
    return np.any(d != fill_value, axis=2)


def sparsify(
    data: np.ndarray,
    keep_mask: np.ndarray,
    *,
    symmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a dense [N,N] or [N,N,C] array to COO (indices, values).

    For symmetric features only upper-triangle (i <= j) entries are kept.
    Returns indices [2, nnz] int32 and values [nnz, C] (C from data).
    """
    d = _as_3d(data)
    n = d.shape[0]
    mask = keep_mask
    if symmetric:
        mask = mask & np.triu(np.ones((n, n), dtype=bool))
    rows, cols = np.nonzero(mask)
    indices = np.stack([rows, cols]).astype(np.int32)
    values = d[rows, cols]
    return indices, values


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def compute_distance_matrix(backbone_positions: np.ndarray) -> np.ndarray:
    """Compute Cα-Cα pairwise Euclidean distances.

    Parameters
    ----------
    backbone_positions : float32 [N_res, 4, 3]   N / CA / C / O

    Returns
    -------
    dist : float32 [N_res, N_res]   Angstroms, symmetric
    """
    ca = backbone_positions[:, 1, :].astype(np.float64)   # [N_res, 3]
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    sq = np.sum(ca ** 2, axis=1)
    sq_dist = sq[:, None] + sq[None, :] - 2.0 * (ca @ ca.T)
    np.maximum(sq_dist, 0.0, out=sq_dist)   # numerical floor
    return np.sqrt(sq_dist).astype(np.float32)


def compute_contact_map(
    distance_matrix: np.ndarray,
    threshold: float = 8.0,
) -> np.ndarray:
    """Boolean contact map: True where Cα-Cα distance < threshold Angstroms.

    Parameters
    ----------
    distance_matrix : float32 [N_res, N_res]
    threshold       : contact distance cutoff in Angstroms (default 8.0)

    Returns
    -------
    contacts : bool [N_res, N_res]
    """
    return (distance_matrix < threshold)
