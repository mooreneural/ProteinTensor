"""Tests for sparse (COO) pair feature storage."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import proteintensor as pt
from proteintensor.pairs import compute_distance_matrix, sparsify, radius_mask


def _ptt_with_backbone(tmp: str, n_res: int = 40) -> tuple[Path, "object"]:
    from proteintensor.schema import ProteinTensorData, N_BACKBONE
    rng = np.random.default_rng(0)
    n_atoms = n_res * 4
    data = ProteinTensorData(
        sequence_tokens=rng.integers(0, 19, n_res, dtype=np.int32),
        residue_index=np.arange(n_res, dtype=np.int32),
        chain_id=np.array([b"A"] * n_res, dtype="S1"),
        atom_positions=rng.standard_normal((n_atoms, 3)).astype(np.float32),
        atom_mask=np.ones(n_atoms, dtype=bool),
        b_factors=np.zeros(n_atoms, dtype=np.float32),
        residue_atom_start=np.arange(0, n_atoms, 4, dtype=np.int32),
        residue_atom_count=np.full(n_res, 4, dtype=np.int32),
        backbone_positions=(rng.standard_normal((n_res, N_BACKBONE, 3)) * 10).astype(np.float32),
        backbone_mask=np.ones((n_res, N_BACKBONE), dtype=bool),
        pdb_id="TEST",
    )
    p = Path(tmp) / "t.ptt"
    pt.write(data, p)
    return p, data


# --------------------------------------------------------------------------
# core encode/decode
# --------------------------------------------------------------------------

def test_sparsify_densify_identity_on_kept_entries():
    rng = np.random.default_rng(1)
    dense = rng.standard_normal((20, 20, 2)).astype(np.float32)
    mask = rng.random((20, 20)) < 0.2
    idx, val = sparsify(dense, mask, symmetric=False)
    assert idx.shape == (2, int(mask.sum()))
    # reconstruct
    out = np.zeros_like(dense)
    out[idx[0], idx[1]] = val
    np.testing.assert_array_equal(out, np.where(mask[:, :, None], dense, 0))


def test_symmetric_stores_upper_triangle_only():
    n = 30
    dense = np.ones((n, n, 1), dtype=np.float32)
    full_mask = np.ones((n, n), dtype=bool)
    idx, _ = sparsify(dense, full_mask, symmetric=True)
    assert idx.shape[1] == n * (n + 1) // 2   # upper triangle incl diagonal
    # every stored index has row <= col
    assert (idx[0] <= idx[1]).all()


# --------------------------------------------------------------------------
# store / read round-trip
# --------------------------------------------------------------------------

def test_add_read_sparse_threshold(tmp_path):
    p, _ = _ptt_with_backbone(str(tmp_path))
    rng = np.random.default_rng(2)
    data = rng.standard_normal((40, 40, 1)).astype(np.float32)
    data[np.abs(data) < 1.5] = 0.0   # make it sparse

    nnz = pt.add_pair_feature_sparse(p, data, "feat", mode="threshold", threshold=0.5)
    sp = pt.read_pair_feature_sparse(p, "feat")
    assert sp.nnz == nnz
    assert sp.channels == 1
    assert 0.0 < sp.density < 1.0
    dense = sp.to_dense()
    np.testing.assert_array_equal(dense, np.where(np.abs(data) >= 0.5, data, 0.0))


def test_distances_sparse_matches_dense_within_cutoff(tmp_path):
    p, data = _ptt_with_backbone(str(tmp_path))
    cutoff = 15.0
    pt.compute_and_store_distances_sparse(p, cutoff=cutoff)

    sp = pt.read_pair_feature_sparse(p, "distance_matrix")
    assert sp.symmetric is True
    full = compute_distance_matrix(data.backbone_positions)
    dense = sp.to_dense()[:, :, 0]
    keep = full <= cutoff
    np.testing.assert_allclose(dense[keep], full[keep], atol=1e-4)
    assert (dense[~keep] == 0).all()


def test_contacts_sparse_matches_dense(tmp_path):
    p, data = _ptt_with_backbone(str(tmp_path))
    pt.compute_and_store_contacts_sparse(p, threshold=8.0)
    sp = pt.read_pair_feature_sparse(p, "contacts")
    dense = sp.to_dense()[:, :, 0].astype(bool)
    full = compute_distance_matrix(data.backbone_positions) < 8.0
    np.testing.assert_array_equal(dense, full)


def test_sparse_distances_smaller_on_disk_than_dense(tmp_path):
    # The clear win: a float distance matrix (poorly compressible dense) with a
    # tight radius cutoff. (Boolean contact maps compress so well dense that COO
    # index overhead can lose at small N - that nuance is shown in the benchmark.)
    p, data = _ptt_with_backbone(str(tmp_path), n_res=120)
    pt.compute_and_store_distances(p)                     # dense float32 [N,N,1]
    pt.compute_and_store_distances_sparse(p, cutoff=8.0)  # sparse radius graph

    def _dir_bytes(d: Path) -> int:
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())

    dense_bytes  = _dir_bytes(p / "pairs" / "distance_matrix")
    sparse_bytes = _dir_bytes(p / "pairs_sparse" / "distance_matrix")
    assert sparse_bytes < dense_bytes


def test_list_and_overwrite(tmp_path):
    p, _ = _ptt_with_backbone(str(tmp_path))
    pt.compute_and_store_distances_sparse(p, cutoff=12.0)
    assert "distance_matrix" in pt.list_pair_features_sparse(p)
    with pytest.raises(ValueError):
        pt.compute_and_store_distances_sparse(p, cutoff=12.0)
    pt.compute_and_store_distances_sparse(p, cutoff=20.0, overwrite=True)  # no raise
