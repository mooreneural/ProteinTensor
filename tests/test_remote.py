"""Tests for cloud/remote storage support via fsspec memory:// filesystem."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

fsspec = pytest.importorskip("fsspec", reason="fsspec not installed")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_ptt(tmp: str, n_res: int = 20, pdb_id: str = "TEST") -> Path:
    from proteintensor.schema import ProteinTensorData, N_BACKBONE
    from proteintensor import write

    rng = np.random.default_rng(42)
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


def _local_to_memory(local_path: Path, memory_url: str) -> None:
    """Copy a local Zarr store into an fsspec memory:// store."""
    local_store = zarr.DirectoryStore(str(local_path))
    mem_mapper  = fsspec.get_mapper(memory_url)
    zarr.copy_store(local_store, mem_mapper)


# ---------------------------------------------------------------------------
# remote.open_store
# ---------------------------------------------------------------------------

def test_open_store_local_path():
    """open_store() with a local path behaves like zarr.open()."""
    from proteintensor.remote import open_store
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp)
        store = open_store(ptt)
        assert "sequence" in store


def test_open_store_memory_url():
    """open_store() accepts fsspec memory:// URLs."""
    from proteintensor.remote import open_store
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp)
        _local_to_memory(ptt, "memory://test_open_store/test.ptt")
        store = open_store("memory://test_open_store/test.ptt")
    assert "sequence" in store


def test_is_url():
    from proteintensor.remote import _is_url
    assert _is_url("s3://bucket/path")
    assert _is_url("gs://bucket/path")
    assert _is_url("memory://path")
    assert not _is_url("/local/path.ptt")
    assert not _is_url("relative/path.ptt")


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------

def test_consolidate_local_writes_zmetadata():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp)
        from proteintensor import consolidate
        consolidate(ptt)
        # .zmetadata should now exist inside the store directory
        meta = ptt / ".zmetadata"
        assert meta.exists(), ".zmetadata not written"


def test_consolidate_memory_url():
    from proteintensor import consolidate
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp)
        _local_to_memory(ptt, "memory://test_consolidate/test.ptt")
        consolidate("memory://test_consolidate/test.ptt")
        # open_consolidated should now succeed
        mapper = fsspec.get_mapper("memory://test_consolidate/test.ptt")
        store = zarr.open_consolidated(mapper, mode="r")
        assert "sequence" in store


def test_consolidate_then_open_store_uses_consolidated():
    """open_store() after consolidate() returns a ConsolidatedMetadataStore."""
    from proteintensor.remote import open_store
    from proteintensor import consolidate
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp)
        _local_to_memory(ptt, "memory://test_consol2/test.ptt")
        consolidate("memory://test_consol2/test.ptt")
        store = open_store("memory://test_consol2/test.ptt")
    # The store type should be a consolidated metadata store
    assert isinstance(store.store, zarr.storage.ConsolidatedMetadataStore)


# ---------------------------------------------------------------------------
# read() from memory URL
# ---------------------------------------------------------------------------

def test_read_from_memory_url():
    from proteintensor import read, write
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, n_res=30, pdb_id="1ABC")
        original = read(ptt)
        _local_to_memory(ptt, "memory://test_read/1ABC.ptt")
        remote = read("memory://test_read/1ABC.ptt")

    np.testing.assert_array_equal(original.sequence_tokens, remote.sequence_tokens)
    np.testing.assert_array_almost_equal(original.atom_positions, remote.atom_positions)
    assert remote.pdb_id == "1ABC"
    assert remote.sequence_tokens.shape[0] == 30


def test_read_backbone_from_memory_url():
    from proteintensor import read_backbone
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, n_res=25)
        _local_to_memory(ptt, "memory://test_bb/test.ptt")
        bb = read_backbone("memory://test_bb/test.ptt")
    assert bb.positions.shape == (25, 4, 3)
    assert bb.sequence_tokens.shape == (25,)


def test_read_bonds_from_memory_url():
    from proteintensor.schema import ProteinTensorData, N_BACKBONE
    from proteintensor import write, read_bonds
    from proteintensor.converters.mmcif import from_mmcif
    with tempfile.TemporaryDirectory() as tmp:
        # Use a ptt with bonds written by a converter - fake it by checking
        # a ptt without bonds raises correctly
        ptt = _make_ptt(tmp, n_res=15)
        _local_to_memory(ptt, "memory://test_bonds/test.ptt")
        with pytest.raises(KeyError, match="No bonds group"):
            read_bonds("memory://test_bonds/test.ptt")


def test_list_msas_from_memory_url():
    from proteintensor import list_msas
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp)
        _local_to_memory(ptt, "memory://test_list_msa/test.ptt")
        sources = list_msas("memory://test_list_msa/test.ptt")
    assert sources == []


def test_list_pair_features_from_memory_url():
    from proteintensor import list_pair_features
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp)
        _local_to_memory(ptt, "memory://test_list_pairs/test.ptt")
        features = list_pair_features("memory://test_list_pairs/test.ptt")
    assert features == []


def test_list_embeddings_from_memory_url():
    from proteintensor import list_embeddings
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp)
        _local_to_memory(ptt, "memory://test_list_emb/test.ptt")
        embs = list_embeddings("memory://test_list_emb/test.ptt")
    assert embs == []


# ---------------------------------------------------------------------------
# mmap from memory URL
# ---------------------------------------------------------------------------

def test_mmap_positions_from_memory_url():
    from proteintensor import mmap_positions
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, n_res=20)
        _local_to_memory(ptt, "memory://test_mmap_pos/test.ptt")
        arr = mmap_positions("memory://test_mmap_pos/test.ptt")
    assert arr.shape == (80, 3)
    # Slice without full load
    chunk = arr[:4]
    assert chunk.shape == (4, 3)


def test_mmap_backbone_from_memory_url():
    from proteintensor import mmap_backbone
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, n_res=20)
        _local_to_memory(ptt, "memory://test_mmap_bb/test.ptt")
        arr = mmap_backbone("memory://test_mmap_bb/test.ptt")
    assert arr.shape == (20, 4, 3)


# ---------------------------------------------------------------------------
# ProteinDataset from memory URL
# ---------------------------------------------------------------------------

def test_protein_dataset_from_memory_url():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        for n, pid in [(20, "1ABC"), (30, "2DEF")]:
            add_to_dataset(ds_path, _make_ptt(tmp, n, pid))

        # Copy dataset to memory
        _local_to_memory(ds_path, "memory://test_ds/ds.ptt")

        ds = ProteinDataset("memory://test_ds/ds.ptt")
        assert len(ds) == 2

        with tempfile.TemporaryDirectory() as tmp2:
            # Read structures from remote dataset
            d0 = ds[0]
            d1 = ds[1]

    assert d0.sequence_tokens.shape[0] == 20
    assert d1.sequence_tokens.shape[0] == 30


def test_protein_dataset_pdb_lookup_from_memory_url():
    from proteintensor import create_dataset, add_to_dataset, ProteinDataset
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = Path(tmp) / "ds.ptt"
        create_dataset(ds_path)
        add_to_dataset(ds_path, _make_ptt(tmp, 25, "1UBQ"))
        _local_to_memory(ds_path, "memory://test_ds_pdb/ds.ptt")
        ds = ProteinDataset("memory://test_ds_pdb/ds.ptt")
        data = ds["1UBQ"]
    assert data.pdb_id == "1UBQ"
    assert data.sequence_tokens.shape[0] == 25


# ---------------------------------------------------------------------------
# missing fsspec raises helpful error
# ---------------------------------------------------------------------------

def test_require_fsspec_raises_on_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "fsspec":
            raise ImportError("mocked missing fsspec")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    from proteintensor.remote import _require_fsspec
    with pytest.raises(ImportError, match="proteintensor\\[cloud\\]"):
        _require_fsspec()
