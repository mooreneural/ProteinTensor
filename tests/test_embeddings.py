from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from proteintensor.embeddings import EmbeddingData, KNOWN_DIMS, sequence_hash


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dummy_ptt(tmp: str, n_res: int = 30) -> Path:
    from proteintensor.schema import ProteinTensorData, N_BACKBONE
    from proteintensor import write
    rng = np.random.default_rng(2)
    n_atoms = n_res * 4
    data = ProteinTensorData(
        sequence_tokens=rng.integers(0, 20, n_res, dtype=np.int32),
        residue_index=np.arange(n_res, dtype=np.int32),
        chain_id=np.array([b"A"] * n_res, dtype="S1"),
        atom_positions=rng.standard_normal((n_atoms, 3)).astype(np.float32),
        atom_mask=np.ones(n_atoms, dtype=bool),
        b_factors=np.zeros(n_atoms, dtype=np.float32),
        residue_atom_start=np.arange(0, n_atoms, 4, dtype=np.int32),
        residue_atom_count=np.full(n_res, 4, dtype=np.int32),
        backbone_positions=rng.standard_normal((n_res, N_BACKBONE, 3)).astype(np.float32),
        backbone_mask=np.ones((n_res, N_BACKBONE), dtype=bool),
    )
    p = Path(tmp) / "test.ptt"
    write(data, p)
    return p


def _mock_embedding(n_res: int, dim: int = 1280) -> np.ndarray:
    return np.random.default_rng(3).standard_normal((n_res, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# sequence_hash
# ---------------------------------------------------------------------------

def test_sequence_hash_deterministic():
    tokens = np.arange(20, dtype=np.int32)
    assert sequence_hash(tokens) == sequence_hash(tokens)


def test_sequence_hash_differs_on_change():
    t1 = np.arange(20, dtype=np.int32)
    t2 = t1.copy(); t2[5] = 99
    assert sequence_hash(t1) != sequence_hash(t2)


def test_sequence_hash_is_sha256_hex():
    h = sequence_hash(np.zeros(10, dtype=np.int32))
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# add_embedding / read_embedding
# ---------------------------------------------------------------------------

def test_add_and_read_embedding_float32():
    from proteintensor import add_embedding, read_embedding
    emb = _mock_embedding(30, dim=1280)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=30)
        add_embedding(ptt, emb, model="esm2_t33_650M_UR50D",
                      dtype="float32", sequence_hash="abc")
        loaded = read_embedding(ptt, "esm2_t33_650M_UR50D")

    assert isinstance(loaded, EmbeddingData)
    assert loaded.data.shape    == (30, 1280)
    assert loaded.data.dtype    == np.float32
    assert loaded.model         == "esm2_t33_650M_UR50D"
    assert loaded.dim           == 1280
    assert loaded.sequence_hash == "abc"
    np.testing.assert_array_almost_equal(emb, loaded.data, decimal=4)


def test_add_and_read_embedding_float16_storage():
    """Stored as float16, read_embedding returns float32 (upcast on load)."""
    from proteintensor import add_embedding, read_embedding, mmap_embedding
    emb = _mock_embedding(20, dim=480)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=20)
        add_embedding(ptt, emb, model="esm2_t12_35M_UR50D", dtype="float16")

        loaded = read_embedding(ptt, "esm2_t12_35M_UR50D")
        assert loaded.data.dtype == np.float32   # always upcast on read
        assert loaded.dtype      == "float16"    # stored dtype preserved in attrs

        lazy = mmap_embedding(ptt, "esm2_t12_35M_UR50D")
        assert lazy.dtype == np.float16          # lazy keeps storage dtype


def test_float16_size_is_half_of_float32():
    """float16 stored arrays must be ~2x smaller than float32."""
    from proteintensor import add_embedding
    emb = _mock_embedding(50, dim=1280)
    with tempfile.TemporaryDirectory() as tmp:
        ptt16 = _dummy_ptt(tmp + "/f16", n_res=50)
        ptt32 = _dummy_ptt(tmp + "/f32", n_res=50)
        import os; os.makedirs(tmp + "/f16", exist_ok=True); os.makedirs(tmp + "/f32", exist_ok=True)
        add_embedding(ptt16, emb, model="esm2_t33_650M_UR50D", dtype="float16")
        add_embedding(ptt32, emb, model="esm2_t33_650M_UR50D", dtype="float32")

        def _emb_bytes(p):
            ep = p / "embeddings" / "esm2_t33_650M_UR50D"
            return sum(f.stat().st_size for f in ep.rglob("*") if f.is_file())

        sz16 = _emb_bytes(ptt16)
        sz32 = _emb_bytes(ptt32)
    assert sz16 < sz32, "float16 should be smaller than float32"
    assert sz32 / sz16 > 1.5, f"Expected ~2x ratio, got {sz32/sz16:.2f}x"


def test_add_embedding_does_not_overwrite_structure():
    from proteintensor import add_embedding, read
    emb = _mock_embedding(30)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=30)
        before = read(ptt)
        add_embedding(ptt, emb, model="esm2_t33_650M_UR50D")
        after  = read(ptt)
    np.testing.assert_array_equal(before.sequence_tokens, after.sequence_tokens)
    np.testing.assert_array_equal(before.atom_positions,  after.atom_positions)


def test_add_embedding_overwrite_guard():
    from proteintensor import add_embedding, read_embedding
    emb = _mock_embedding(10, dim=320)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=10)
        add_embedding(ptt, emb, model="esm2_t6_8M_UR50D", dtype="float32")
        with pytest.raises(ValueError, match="already exists"):
            add_embedding(ptt, emb, model="esm2_t6_8M_UR50D", overwrite=False)
        add_embedding(ptt, emb * 2, model="esm2_t6_8M_UR50D", dtype="float32", overwrite=True)
        loaded = read_embedding(ptt, "esm2_t6_8M_UR50D")
    np.testing.assert_array_almost_equal(loaded.data, emb * 2, decimal=4)


def test_read_embedding_missing_raises():
    from proteintensor import read_embedding
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        with pytest.raises(KeyError, match="esm2_t33_650M_UR50D"):
            read_embedding(ptt, "esm2_t33_650M_UR50D")


# ---------------------------------------------------------------------------
# list_embeddings / mmap_embedding
# ---------------------------------------------------------------------------

def test_list_embeddings_empty():
    from proteintensor import list_embeddings
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        assert list_embeddings(ptt) == []


def test_list_embeddings_multiple():
    from proteintensor import add_embedding, list_embeddings
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=20)
        add_embedding(ptt, _mock_embedding(20, 320),  model="esm2_t6_8M_UR50D")
        add_embedding(ptt, _mock_embedding(20, 1280), model="esm2_t33_650M_UR50D")
        names = list_embeddings(ptt)
    assert set(names) == {"esm2_t6_8M_UR50D", "esm2_t33_650M_UR50D"}


def test_mmap_embedding_is_lazy():
    from proteintensor import add_embedding, mmap_embedding
    emb = _mock_embedding(25, dim=640)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=25)
        add_embedding(ptt, emb, model="esm2_t30_150M_UR50D")
        lazy = mmap_embedding(ptt, "esm2_t30_150M_UR50D")
    assert isinstance(lazy, zarr.Array)
    assert lazy.shape == (25, 640)


def test_mmap_embedding_partial_load():
    """Slicing a lazy embedding must not load all rows."""
    from proteintensor import add_embedding, mmap_embedding
    emb = _mock_embedding(50, dim=480)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=50)
        add_embedding(ptt, emb, model="esm2_t12_35M_UR50D", dtype="float32")
        lazy = mmap_embedding(ptt, "esm2_t12_35M_UR50D")
        row10 = lazy[10, :]
    np.testing.assert_array_almost_equal(row10, emb[10], decimal=4)


# ---------------------------------------------------------------------------
# KNOWN_DIMS
# ---------------------------------------------------------------------------

def test_known_dims_coverage():
    for model, dim in KNOWN_DIMS.items():
        assert dim > 0
        assert "esm" in model.lower()


def test_layer_metadata_roundtrip():
    from proteintensor import add_embedding, read_embedding
    emb = _mock_embedding(15, dim=1280)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=15)
        add_embedding(ptt, emb, model="esm2_t33_650M_UR50D", layer=33)
        loaded = read_embedding(ptt, "esm2_t33_650M_UR50D")
    assert loaded.layer == 33
