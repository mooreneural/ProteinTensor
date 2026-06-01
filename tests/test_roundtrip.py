from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _dummy(n_res: int = 12, atoms_per_res: int = 4, with_backbone: bool = True):
    from proteintensor.schema import ProteinTensorData, N_BACKBONE
    n_atoms = n_res * atoms_per_res
    rng = np.random.default_rng(0)

    bb_pos  = rng.standard_normal((n_res, N_BACKBONE, 3)).astype(np.float32) if with_backbone else None
    bb_mask = np.ones((n_res, N_BACKBONE), dtype=bool) if with_backbone else None

    return ProteinTensorData(
        sequence_tokens=rng.integers(0, 20, n_res, dtype=np.int32),
        residue_index=np.arange(n_res, dtype=np.int32),
        chain_id=np.array([b"A"] * n_res, dtype="S1"),
        atom_positions=rng.standard_normal((n_atoms, 3)).astype(np.float32),
        atom_mask=np.ones(n_atoms, dtype=bool),
        b_factors=rng.uniform(0, 100, n_atoms).astype(np.float32),
        residue_atom_start=np.arange(0, n_atoms, atoms_per_res, dtype=np.int32),
        residue_atom_count=np.full(n_res, atoms_per_res, dtype=np.int32),
        backbone_positions=bb_pos,
        backbone_mask=bb_mask,
        pdb_id="TEST",
        resolution=2.0,
        method="X-RAY DIFFRACTION",
        deposition_date="2024-01-01",
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_roundtrip_arrays():
    from proteintensor import read, write
    data = _dummy()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        loaded = read(p)

    np.testing.assert_array_equal(data.sequence_tokens, loaded.sequence_tokens)
    np.testing.assert_array_equal(data.residue_index,   loaded.residue_index)
    np.testing.assert_array_almost_equal(data.atom_positions, loaded.atom_positions, decimal=5)
    np.testing.assert_array_equal(data.atom_mask,       loaded.atom_mask)
    np.testing.assert_array_equal(data.residue_atom_start, loaded.residue_atom_start)
    np.testing.assert_array_equal(data.residue_atom_count, loaded.residue_atom_count)


def test_roundtrip_metadata():
    from proteintensor import read, write
    data = _dummy()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        loaded = read(p)

    assert loaded.pdb_id == "TEST"
    assert loaded.resolution == pytest.approx(2.0)
    assert loaded.method == "X-RAY DIFFRACTION"
    assert loaded.deposition_date == "2024-01-01"


def test_zarr_attrs():
    from proteintensor import write
    data = _dummy(n_res=8, atoms_per_res=3)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        store = zarr.open(str(p), mode="r")
        assert store.attrs["format"] == "ProteinTensor"
        assert store.attrs["num_residues"] == 8
        assert store.attrs["num_atoms"] == 24


def test_mmap_positions_is_lazy():
    """mmap_positions must return a zarr.Array, not a numpy array."""
    from proteintensor import write, mmap_positions
    data = _dummy()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        arr = mmap_positions(p)
        assert isinstance(arr, zarr.Array), "Expected lazy zarr.Array, got numpy array"
        assert arr.shape == data.atom_positions.shape


def test_mmap_tokens_is_lazy():
    from proteintensor import write, mmap_tokens
    data = _dummy()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        arr = mmap_tokens(p)
        assert isinstance(arr, zarr.Array)
        assert arr.shape == (data.sequence_tokens.shape[0],)


def test_no_blosc_falls_back_cleanly(monkeypatch):
    """writer must not crash when numcodecs.Blosc is unavailable."""
    import proteintensor.writer as wmod
    original = wmod._compressor

    def _no_blosc(name):
        if name == "blosc":
            return None
        return original(name)

    monkeypatch.setattr(wmod, "_compressor", _no_blosc)
    from proteintensor import read, write
    data = _dummy()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        loaded = read(p)
    np.testing.assert_array_equal(data.sequence_tokens, loaded.sequence_tokens)


def test_backbone_roundtrip():
    from proteintensor import read, write
    from proteintensor.schema import N_BACKBONE
    data = _dummy(n_res=10)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        loaded = read(p)

    assert loaded.backbone_positions is not None
    assert loaded.backbone_mask is not None
    assert loaded.backbone_positions.shape == (10, N_BACKBONE, 3)
    assert loaded.backbone_mask.shape == (10, N_BACKBONE)
    np.testing.assert_array_almost_equal(data.backbone_positions, loaded.backbone_positions, decimal=5)
    np.testing.assert_array_equal(data.backbone_mask, loaded.backbone_mask)


def test_read_backbone_only():
    """read_backbone() must return only backbone+sequence without loading heavy atoms."""
    from proteintensor import read_backbone, write
    from proteintensor.schema import N_BACKBONE, BackboneData
    data = _dummy(n_res=8)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        bb = read_backbone(p)

    assert isinstance(bb, BackboneData)
    assert bb.positions.shape == (8, N_BACKBONE, 3)
    assert bb.mask.shape == (8, N_BACKBONE)
    assert bb.sequence_tokens.shape == (8,)
    np.testing.assert_array_equal(bb.sequence_tokens, data.sequence_tokens)


def test_mmap_backbone_is_lazy():
    from proteintensor import write, mmap_backbone
    from proteintensor.schema import N_BACKBONE
    data = _dummy()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        arr = mmap_backbone(p)
        assert isinstance(arr, zarr.Array)
        assert arr.shape == (data.backbone_positions.shape[0], N_BACKBONE, 3)


def test_backbone_missing_atoms_mask():
    """Residues with missing backbone atoms must have mask=False at those positions."""
    from proteintensor import read, write
    from proteintensor.schema import ProteinTensorData, N_BACKBONE
    rng = np.random.default_rng(42)
    n_res = 6
    bb_pos  = rng.standard_normal((n_res, N_BACKBONE, 3)).astype(np.float32)
    bb_mask = np.ones((n_res, N_BACKBONE), dtype=bool)
    bb_mask[2, 1] = False  # residue 2 missing CA
    bb_mask[4, 3] = False  # residue 4 missing O
    bb_pos[bb_mask == False] = 0.0

    data = _dummy(n_res=n_res)
    data.backbone_positions = bb_pos
    data.backbone_mask = bb_mask

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        loaded = read(p)

    assert not loaded.backbone_mask[2, 1]
    assert not loaded.backbone_mask[4, 3]
    assert loaded.backbone_mask.sum() == n_res * N_BACKBONE - 2


def test_no_backbone_field_is_none():
    """Files written without backbone data should round-trip backbone_positions=None."""
    from proteintensor import read, write
    data = _dummy(with_backbone=False)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        loaded = read(p)
    assert loaded.backbone_positions is None
    assert loaded.backbone_mask is None


def test_read_backbone_raises_on_missing_group():
    from proteintensor import write, read_backbone
    data = _dummy(with_backbone=False)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        with pytest.raises(KeyError, match="backbone"):
            read_backbone(p)


def test_bonds_roundtrip():
    from proteintensor import read, write
    from proteintensor.bonds import BOND_PEPTIDE, BOND_SINGLE
    rng = np.random.default_rng(7)
    n_edges = 40
    data = _dummy()
    data.bond_edge_index = rng.integers(0, data.atom_positions.shape[0], (2, n_edges), dtype=np.int32)
    data.bond_edge_type  = np.full(n_edges, BOND_SINGLE, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        loaded = read(p)

    assert loaded.bond_edge_index is not None
    assert loaded.bond_edge_type  is not None
    np.testing.assert_array_equal(data.bond_edge_index, loaded.bond_edge_index)
    np.testing.assert_array_equal(data.bond_edge_type,  loaded.bond_edge_type)


def test_read_bonds_only():
    from proteintensor import read_bonds, write
    from proteintensor.schema import BondData
    from proteintensor.bonds import BOND_AROMATIC
    rng = np.random.default_rng(8)
    n_edges = 20
    data = _dummy()
    data.bond_edge_index = rng.integers(0, data.atom_positions.shape[0], (2, n_edges), dtype=np.int32)
    data.bond_edge_type  = np.full(n_edges, BOND_AROMATIC, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        bonds = read_bonds(p)

    assert isinstance(bonds, BondData)
    assert bonds.edge_index.shape == (2, n_edges)
    assert bonds.edge_type.shape  == (n_edges,)


def test_bonds_build_peptide_connectivity():
    """ALA-ALA-ALA chain must have 2 peptide bonds at C->N junctions."""
    from proteintensor.bonds import build, BOND_PEPTIDE
    # Minimal atom layout: each ALA has N(0), CA(1), C(2), O(3), CB(4)
    # Residue 0: atoms 0-4, Residue 1: atoms 5-9, Residue 2: atoms 10-14
    maps = [
        {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4},
        {"N": 5, "CA": 6, "C": 7, "O": 8, "CB": 9},
        {"N": 10, "CA": 11, "C": 12, "O": 13, "CB": 14},
    ]
    resnames  = ["ALA", "ALA", "ALA"]
    chain_ids = [b"A", b"A", b"A"]
    positions = np.zeros((15, 3), dtype=np.float32)

    edge_index, edge_type = build(maps, resnames, chain_ids, positions)

    # Find peptide edges
    peptide_mask = edge_type == BOND_PEPTIDE
    peptide_edges = set(map(tuple, edge_index[:, peptide_mask].T.tolist()))

    assert (2, 5)  in peptide_edges, "Missing C0->N1 peptide bond"
    assert (5, 2)  in peptide_edges, "Missing N1->C0 reverse"
    assert (7, 10) in peptide_edges, "Missing C1->N2 peptide bond"
    assert (10, 7) in peptide_edges, "Missing N2->C1 reverse"


def test_bonds_no_peptide_across_chains():
    """Peptide bonds must not be added between different chains."""
    from proteintensor.bonds import build, BOND_PEPTIDE
    maps = [
        {"N": 0, "CA": 1, "C": 2, "O": 3},
        {"N": 4, "CA": 5, "C": 6, "O": 7},
    ]
    resnames  = ["GLY", "GLY"]
    chain_ids = [b"A", b"B"]   # different chains
    positions = np.zeros((8, 3), dtype=np.float32)

    edge_index, edge_type = build(maps, resnames, chain_ids, positions)
    assert (BOND_PEPTIDE not in edge_type), "Peptide bond incorrectly added across chains"


def test_bonds_disulfide_detection():
    """Two CYS SG atoms within 2.05 A must produce BOND_DISULFIDE edges."""
    from proteintensor.bonds import build, BOND_DISULFIDE
    # CYS-GLY-CYS, SG atoms at indices 4 and 9
    maps = [
        {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "SG": 5},
        {"N": 6, "CA": 7, "C": 8, "O": 9},
        {"N": 10, "CA": 11, "C": 12, "O": 13, "CB": 14, "SG": 15},
    ]
    resnames  = ["CYS", "GLY", "CYS"]
    chain_ids = [b"A", b"A", b"A"]
    positions = np.zeros((16, 3), dtype=np.float32)
    positions[5]  = [0.0, 0.0, 0.0]
    positions[15] = [2.0, 0.0, 0.0]  # 2.0 Å apart -> disulfide

    edge_index, edge_type = build(maps, resnames, chain_ids, positions)
    ds_mask = edge_type == BOND_DISULFIDE
    assert ds_mask.sum() == 2, "Expected 2 disulfide edges (bidirectional)"
    ds_edges = set(map(tuple, edge_index[:, ds_mask].T.tolist()))
    assert (5, 15) in ds_edges and (15, 5) in ds_edges


def test_bonds_no_disulfide_far_sg():
    """SG atoms > 2.5 Å apart must NOT be connected."""
    from proteintensor.bonds import build, BOND_DISULFIDE
    maps = [
        {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "SG": 5},
        {"N": 6, "CA": 7, "C": 8, "O": 9, "CB": 10, "SG": 11},
    ]
    resnames  = ["CYS", "CYS"]
    chain_ids = [b"A", b"A"]
    positions = np.zeros((12, 3), dtype=np.float32)
    positions[5]  = [0.0, 0.0, 0.0]
    positions[11] = [5.0, 0.0, 0.0]  # 5.0 Å apart -> no disulfide

    edge_index, edge_type = build(maps, resnames, chain_ids, positions)
    assert BOND_DISULFIDE not in edge_type


def test_bonds_missing_field_is_none():
    from proteintensor import read, write
    data = _dummy()
    # Don't set bond fields (they default to None)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        loaded = read(p)
    assert loaded.bond_edge_index is None
    assert loaded.bond_edge_type  is None


def test_read_bonds_raises_on_missing_group():
    from proteintensor import read_bonds, write
    data = _dummy()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        with pytest.raises(KeyError, match="bonds"):
            read_bonds(p)


def test_nan_resolution_survives_roundtrip():
    from proteintensor import read, write
    data = _dummy()
    data.resolution = float("nan")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        loaded = read(p)
    assert loaded.resolution != loaded.resolution  # NaN != NaN
