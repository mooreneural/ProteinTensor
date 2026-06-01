from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _dummy(n_res: int = 12, atoms_per_res: int = 4):
    from proteintensor.schema import ProteinTensorData
    n_atoms = n_res * atoms_per_res
    rng = np.random.default_rng(0)
    return ProteinTensorData(
        sequence_tokens=rng.integers(0, 20, n_res, dtype=np.int32),
        residue_index=np.arange(n_res, dtype=np.int32),
        chain_id=np.array([b"A"] * n_res, dtype="S1"),
        atom_positions=rng.standard_normal((n_atoms, 3)).astype(np.float32),
        atom_mask=np.ones(n_atoms, dtype=bool),
        b_factors=rng.uniform(0, 100, n_atoms).astype(np.float32),
        residue_atom_start=np.arange(0, n_atoms, atoms_per_res, dtype=np.int32),
        residue_atom_count=np.full(n_res, atoms_per_res, dtype=np.int32),
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


def test_nan_resolution_survives_roundtrip():
    from proteintensor import read, write
    data = _dummy()
    data.resolution = float("nan")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.ptt"
        write(data, p)
        loaded = read(p)
    assert loaded.resolution != loaded.resolution  # NaN != NaN
