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
  mmap_pair_feature        lazy zarr.Array — slice without full load

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
