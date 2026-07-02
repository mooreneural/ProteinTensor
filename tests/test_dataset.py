"""Tests for the multi-structure ProteinDataset container."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_ptt(tmp: str, n_res: int, pdb_id: str = "TEST") -> Path:
    from proteintensor.schema import ProteinTensorData, N_BACKBONE
    from proteintensor import write

    rng = np.random.default_rng(abs(hash(pdb_id)) % (2**32))
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
        backbone_positions=rng.standard_normal((n_res, N_BACKBONE, 3)).astype(np.float32),
        backbone_mask=np.ones((n_res, N_BACKBONE), dtype=bool),
        pdb_id=pdb_id,
        resolution=2.0,
    )
    p = Path(tmp) / f"{pdb_id}.ptt"
    write(data, p)
    return p


# ---------------------------------------------------------------------------
# create_dataset
# ---------------------------------------------------------------------------

def test_create_dataset_empty():
    from proteintensor import create_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        ds = ProteinDataset(ds_path)
    assert len(ds) == 0
    assert ds.keys() == []


def test_create_dataset_overwrite():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        ptt = _make_ptt(tmp, 20, "AAAA")
        create_dataset(ds_path)
        add_to_dataset(ds_path, ptt)
        assert len(ProteinDataset(ds_path)) == 1

        create_dataset(ds_path, overwrite=True)
        assert len(ProteinDataset(ds_path)) == 0


def test_create_dataset_no_overwrite_raises():
    from proteintensor import create_dataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        with pytest.raises(FileExistsError):
            create_dataset(ds_path)


# ---------------------------------------------------------------------------
# add_to_dataset
# ---------------------------------------------------------------------------

def test_add_single_structure():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        ptt = _make_ptt(tmp, 30, "1ABC")
        create_dataset(ds_path)
        key = add_to_dataset(ds_path, ptt)
        ds = ProteinDataset(ds_path)
    assert len(ds) == 1
    assert key == "000000"


def test_add_multiple_structures():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        sizes = [20, 50, 100]
        ids   = ["1ABC", "2DEF", "3GHI"]
        for n, pid in zip(sizes, ids):
            add_to_dataset(ds_path, _make_ptt(tmp, n, pid))
        ds = ProteinDataset(ds_path)
        assert len(ds) == 3
        assert ds.pdb_ids() == ids


def test_add_duplicate_key_raises():
    from proteintensor import create_dataset, add_to_dataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        ptt = _make_ptt(tmp, 20, "1ABC")
        add_to_dataset(ds_path, ptt, key="mykey")
        with pytest.raises(KeyError):
            add_to_dataset(ds_path, ptt, key="mykey")


def test_add_preserves_structure_data():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset, read
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        ptt = _make_ptt(tmp, 40, "1UBQ")
        create_dataset(ds_path)
        add_to_dataset(ds_path, ptt)

        original = read(ptt)
        ds = ProteinDataset(ds_path)
        loaded = ds[0]

    np.testing.assert_array_equal(original.sequence_tokens, loaded.sequence_tokens)
    np.testing.assert_array_almost_equal(original.atom_positions, loaded.atom_positions)
    assert loaded.pdb_id == "1UBQ"


# ---------------------------------------------------------------------------
# ProteinDataset - indexing
# ---------------------------------------------------------------------------

def test_index_by_integer():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        for n, pid in [(20, "1ABC"), (30, "2DEF"), (40, "3GHI")]:
            add_to_dataset(ds_path, _make_ptt(tmp, n, pid))
        ds = ProteinDataset(ds_path)
        assert ds[0].pdb_id == "1ABC"
        assert ds[1].pdb_id == "2DEF"
        assert ds[2].pdb_id == "3GHI"
        assert ds[-1].pdb_id == "3GHI"


def test_index_out_of_range():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        add_to_dataset(ds_path, _make_ptt(tmp, 20, "1ABC"))
        ds = ProteinDataset(ds_path)
    with pytest.raises(IndexError):
        _ = ds[99]


def test_index_by_pdb_id():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        for n, pid in [(20, "1ABC"), (30, "2DEF")]:
            add_to_dataset(ds_path, _make_ptt(tmp, n, pid))
        ds = ProteinDataset(ds_path)
        assert ds["1ABC"].sequence_tokens.shape[0] == 20
        assert ds["2def"].sequence_tokens.shape[0] == 30  # case-insensitive


def test_index_unknown_pdb_raises():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        add_to_dataset(ds_path, _make_ptt(tmp, 20, "1ABC"))
        ds = ProteinDataset(ds_path)
    with pytest.raises(KeyError):
        _ = ds["XXXX"]


# ---------------------------------------------------------------------------
# ProteinDataset - iteration / len
# ---------------------------------------------------------------------------

def test_iter():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        sizes = [20, 30, 40]
        for n, pid in zip(sizes, ["1A", "2B", "3C"]):
            add_to_dataset(ds_path, _make_ptt(tmp, n, pid))
        ds = ProteinDataset(ds_path)
        actual = [s.sequence_tokens.shape[0] for s in ds]
    assert actual == sizes


def test_len_empty():
    from proteintensor import create_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        ds = ProteinDataset(ds_path)
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# ProteinDataset - collate
# ---------------------------------------------------------------------------

def test_collate_basic():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        sizes = [20, 30, 25]
        for n, pid in zip(sizes, ["1A", "2B", "3C"]):
            add_to_dataset(ds_path, _make_ptt(tmp, n, pid))
        ds = ProteinDataset(ds_path)
        batch = ProteinDataset.collate([ds[i] for i in range(3)])

    assert batch["sequence_tokens"].shape == (3, 30)
    assert batch["atom_positions"].shape  == (3, 30 * 4, 3)
    assert batch["padding_mask"].shape    == (3, 30)
    assert batch["n_residues"].tolist()   == [20, 30, 25]
    np.testing.assert_array_equal(batch["padding_mask"][0], [True]*20 + [False]*10)
    np.testing.assert_array_equal(batch["padding_mask"][1], [True]*30)


def test_collate_includes_backbone():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        for n, pid in [(15, "1A"), (20, "2B")]:
            add_to_dataset(ds_path, _make_ptt(tmp, n, pid))
        ds = ProteinDataset(ds_path)
        batch = ProteinDataset.collate([ds[0], ds[1]])

    assert "backbone_positions" in batch
    assert batch["backbone_positions"].shape == (2, 20, 4, 3)
    assert batch["backbone_mask"].shape      == (2, 20, 4)


def test_collate_padding_values():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    from proteintensor.schema import AA_UNK
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        for n, pid in [(10, "1A"), (20, "2B")]:
            add_to_dataset(ds_path, _make_ptt(tmp, n, pid))
        ds = ProteinDataset(ds_path)
        batch = ProteinDataset.collate([ds[0], ds[1]])

    # Padded positions in short sequence should be AA_UNK
    assert (batch["sequence_tokens"][0, 10:] == AA_UNK).all()
    # Padded atom mask should be False
    assert not batch["atom_mask"][0, 40:].any()


# ---------------------------------------------------------------------------
# open non-dataset raises
# ---------------------------------------------------------------------------

def test_open_single_ptt_as_dataset_raises():
    from proteintensor import ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, 20, "1ABC")
        with pytest.raises(ValueError, match="not a ProteinTensor dataset"):
            ProteinDataset(ptt)


# ---------------------------------------------------------------------------
# sequence-only entries in a dataset
# ---------------------------------------------------------------------------

def _make_seq_ptt(tmp: str, seq: str, pdb_id: str) -> Path:
    from proteintensor import from_sequence, write
    p = Path(tmp) / f"{pdb_id}.ptt"
    write(from_sequence(seq, pdb_id=pdb_id), p)
    return p


def test_dataset_reads_sequence_only_entry():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        add_to_dataset(ds_path, _make_seq_ptt(tmp, "MKTAYIAKQR", "SEQ1"))

        ds = ProteinDataset(ds_path)
        assert len(ds) == 1
        data = ds["SEQ1"]
        assert data.has_structure is False
        assert data.atom_positions is None
        assert data.residue_atom_start is None
        assert data.sequence_tokens.shape[0] == 10


def test_collate_all_sequence_only():
    from proteintensor import ProteinDataset, from_sequence
    samples = [from_sequence("MKTAYIAKQR"), from_sequence("QRLLGKPFSAED")]
    batch = ProteinDataset.collate(samples)
    assert batch["sequence_tokens"].shape == (2, 12)   # padded to longest
    assert batch["atom_positions"].shape == (2, 0, 3)  # no atoms in the batch
    assert batch["n_atoms"].tolist() == [0, 0]
    assert batch["has_structure"].tolist() == [False, False]
    # padding mask marks real residues per row
    assert batch["padding_mask"][0].sum() == 10
    assert batch["padding_mask"][1].sum() == 12


def test_collate_mixed_structure_and_sequence_only():
    from proteintensor import ProteinDataset, from_sequence, read
    with tempfile.TemporaryDirectory() as tmp:
        struct = read(_make_ptt(tmp, 15, "STRC"))     # has structure
        seqonly = from_sequence("MKTAYIAKQRQISFV", pdb_id="SEQ2")  # 15 res, no structure

        batch = ProteinDataset.collate([struct, seqonly])
        assert batch["has_structure"].tolist() == [True, False]
        assert batch["n_atoms"].tolist() == [15 * 4, 0]
        # structure entry keeps its atoms; sequence-only entry is all-padding
        assert batch["atom_mask"][0, :60].all()
        assert not batch["atom_mask"][1].any()
        # both contribute real residues
        assert batch["padding_mask"][0].sum() == 15
        assert batch["padding_mask"][1].sum() == 15
        # backbone block omitted because not all samples have backbone
        assert "backbone_positions" not in batch
