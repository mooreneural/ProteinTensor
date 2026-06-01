from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from proteintensor.pairs import (
    PairFeature, compute_distance_matrix, compute_contact_map,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dummy_ptt(tmp: str, n_res: int = 20) -> Path:
    from proteintensor.schema import ProteinTensorData, N_BACKBONE
    from proteintensor import write
    rng = np.random.default_rng(1)
    n_atoms = n_res * 4
    # Place Ca atoms on a line so distances are predictable
    bb = np.zeros((n_res, N_BACKBONE, 3), dtype=np.float32)
    for i in range(n_res):
        bb[i, 1, 0] = float(i) * 4.0   # Ca separated by 4 A
    data = ProteinTensorData(
        sequence_tokens=rng.integers(0, 20, n_res, dtype=np.int32),
        residue_index=np.arange(n_res, dtype=np.int32),
        chain_id=np.array([b"A"] * n_res, dtype="S1"),
        atom_positions=rng.standard_normal((n_atoms, 3)).astype(np.float32),
        atom_mask=np.ones(n_atoms, dtype=bool),
        b_factors=np.zeros(n_atoms, dtype=np.float32),
        residue_atom_start=np.arange(0, n_atoms, 4, dtype=np.int32),
        residue_atom_count=np.full(n_res, 4, dtype=np.int32),
        backbone_positions=bb,
        backbone_mask=np.ones((n_res, N_BACKBONE), dtype=bool),
    )
    p = Path(tmp) / "test.ptt"
    write(data, p)
    return p


# ---------------------------------------------------------------------------
# compute_distance_matrix
# ---------------------------------------------------------------------------

def test_distance_matrix_shape():
    bb = np.zeros((10, 4, 3), dtype=np.float32)
    dist = compute_distance_matrix(bb)
    assert dist.shape == (10, 10)
    assert dist.dtype == np.float32


def test_distance_matrix_diagonal_zero():
    rng = np.random.default_rng(0)
    bb = rng.standard_normal((12, 4, 3)).astype(np.float32)
    dist = compute_distance_matrix(bb)
    np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-5)


def test_distance_matrix_symmetric():
    rng = np.random.default_rng(0)
    bb = rng.standard_normal((15, 4, 3)).astype(np.float32)
    dist = compute_distance_matrix(bb)
    np.testing.assert_allclose(dist, dist.T, atol=1e-5)


def test_distance_matrix_known_values():
    """Ca atoms on a line at 4 A spacing -> dist[i,j] = 4*|i-j|."""
    n = 5
    bb = np.zeros((n, 4, 3), dtype=np.float32)
    for i in range(n):
        bb[i, 1, 0] = float(i) * 4.0
    dist = compute_distance_matrix(bb)
    for i in range(n):
        for j in range(n):
            np.testing.assert_allclose(dist[i, j], abs(i - j) * 4.0, atol=1e-4)


# ---------------------------------------------------------------------------
# compute_contact_map
# ---------------------------------------------------------------------------

def test_contact_map_shape_and_dtype():
    dist = np.eye(10, dtype=np.float32) * 0.0   # all zeros -> all contacts
    contacts = compute_contact_map(dist, threshold=8.0)
    assert contacts.shape == (10, 10)
    assert contacts.dtype == bool


def test_contact_map_threshold():
    """With Ca on a line at 4 A spacing and threshold=9 A, only neighbors touch."""
    n = 6
    bb = np.zeros((n, 4, 3), dtype=np.float32)
    for i in range(n):
        bb[i, 1, 0] = float(i) * 4.0
    dist     = compute_distance_matrix(bb)
    contacts = compute_contact_map(dist, threshold=9.0)
    # Adjacent pairs: dist=4 < 9 -> True
    for i in range(n - 1):
        assert contacts[i, i + 1]
    # Next-nearest: dist=8 < 9 -> True
    for i in range(n - 2):
        assert contacts[i, i + 2]
    # Two apart: dist=12 >= 9 -> False
    for i in range(n - 3):
        assert not contacts[i, i + 3]


def test_contact_map_symmetric():
    rng = np.random.default_rng(0)
    bb = rng.standard_normal((10, 4, 3)).astype(np.float32)
    dist     = compute_distance_matrix(bb)
    contacts = compute_contact_map(dist)
    np.testing.assert_array_equal(contacts, contacts.T)


# ---------------------------------------------------------------------------
# add_pair_feature / read_pair_feature
# ---------------------------------------------------------------------------

def test_add_and_read_pair_feature():
    from proteintensor import add_pair_feature, read_pair_feature
    rng = np.random.default_rng(0)
    feat = rng.standard_normal((10, 10, 3)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=10)
        add_pair_feature(ptt, feat, name="template_pair",
                         symmetric=False, description="test", dtype="float32")
        pf = read_pair_feature(ptt, "template_pair")

    assert isinstance(pf, PairFeature)
    assert pf.data.shape   == (10, 10, 3)
    assert pf.channels     == 3
    assert pf.symmetric    == False
    assert pf.description  == "test"
    np.testing.assert_array_almost_equal(feat, pf.data, decimal=5)


def test_add_pair_feature_2d_input_expands():
    """2D [N, N] input must be stored as [N, N, 1]."""
    from proteintensor import add_pair_feature, read_pair_feature
    arr = np.ones((8, 8), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=8)
        add_pair_feature(ptt, arr, name="flat", dtype="float32")
        pf = read_pair_feature(ptt, "flat")
    assert pf.data.shape == (8, 8, 1)


def test_add_pair_feature_does_not_overwrite_structure():
    from proteintensor import add_pair_feature, read
    arr = np.eye(10, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=10)
        before = read(ptt)
        add_pair_feature(ptt, arr, name="eye")
        after  = read(ptt)
    np.testing.assert_array_equal(before.sequence_tokens, after.sequence_tokens)
    np.testing.assert_array_equal(before.atom_positions,  after.atom_positions)


def test_add_pair_feature_overwrite_guard():
    from proteintensor import add_pair_feature
    arr = np.ones((6, 6), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=6)
        add_pair_feature(ptt, arr, name="feat")
        with pytest.raises(ValueError, match="already exists"):
            add_pair_feature(ptt, arr, name="feat", overwrite=False)
        add_pair_feature(ptt, arr * 2, name="feat", overwrite=True)
        from proteintensor import read_pair_feature
        pf = read_pair_feature(ptt, "feat")
    np.testing.assert_array_almost_equal(pf.data[:, :, 0], arr * 2)


def test_read_pair_feature_missing_raises():
    from proteintensor import read_pair_feature
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        with pytest.raises(KeyError, match="distance_matrix"):
            read_pair_feature(ptt, "distance_matrix")


# ---------------------------------------------------------------------------
# list_pair_features / mmap_pair_feature
# ---------------------------------------------------------------------------

def test_list_pair_features_empty():
    from proteintensor import list_pair_features
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        assert list_pair_features(ptt) == []


def test_list_pair_features_multiple():
    from proteintensor import add_pair_feature, list_pair_features
    arr = np.ones((5, 5), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=5)
        add_pair_feature(ptt, arr, name="alpha")
        add_pair_feature(ptt, arr, name="beta")
        names = list_pair_features(ptt)
    assert set(names) == {"alpha", "beta"}


def test_mmap_pair_feature_is_lazy():
    from proteintensor import add_pair_feature, mmap_pair_feature
    arr = np.ones((8, 8, 2), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=8)
        add_pair_feature(ptt, arr, name="feat")
        lazy = mmap_pair_feature(ptt, "feat")
        assert isinstance(lazy, zarr.Array)
        assert lazy.shape == (8, 8, 2)


# ---------------------------------------------------------------------------
# compute_and_store_distances / compute_and_store_contacts
# ---------------------------------------------------------------------------

def test_compute_and_store_distances():
    from proteintensor import compute_and_store_distances, read_pair_feature
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=10)
        compute_and_store_distances(ptt)
        pf = read_pair_feature(ptt, "distance_matrix")

    assert pf.data.shape     == (10, 10, 1)
    assert pf.symmetric      == True
    assert pf.dtype          == "float32"
    # Diagonal must be zero
    np.testing.assert_allclose(pf.data[:, :, 0].diagonal(), 0.0, atol=1e-4)


def test_compute_and_store_contacts_known_values():
    """Ca line at 4 A spacing, threshold=9 A: only i/i+1 and i/i+2 should be True."""
    from proteintensor import compute_and_store_contacts, read_pair_feature
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=6)
        compute_and_store_contacts(ptt, threshold=9.0)
        pf = read_pair_feature(ptt, "contacts")

    c = pf.data[:, :, 0]
    assert c[0, 1] == True    # 4 A < 9 A
    assert c[0, 2] == True    # 8 A < 9 A
    assert c[0, 3] == False   # 12 A >= 9 A


def test_contacts_dtype_is_bool():
    from proteintensor import compute_and_store_contacts, read_pair_feature
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        compute_and_store_contacts(ptt)
        pf = read_pair_feature(ptt, "contacts")
    assert pf.data.dtype == bool


def test_convenience_wrappers_overwrite_guard():
    from proteintensor import compute_and_store_distances
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        compute_and_store_distances(ptt)
        with pytest.raises(ValueError, match="already exists"):
            compute_and_store_distances(ptt, overwrite=False)
        compute_and_store_distances(ptt, overwrite=True)   # no raise
