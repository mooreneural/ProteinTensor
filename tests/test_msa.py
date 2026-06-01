from __future__ import annotations
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from proteintensor.msa import (
    MsaData, compute_profile, from_a3m,
    MSA_GAP, MSA_MASK, MSA_VOCAB_SIZE, CHAR_TO_TOKEN,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dummy_ptt(tmp: str, n_res: int = 20) -> Path:
    """Write a minimal .ptt file (no MSA) and return its path."""
    from proteintensor.schema import ProteinTensorData, N_BACKBONE
    from proteintensor import write
    rng = np.random.default_rng(0)
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


def _dummy_msa(n_seq: int = 50, n_res: int = 20) -> MsaData:
    rng = np.random.default_rng(42)
    tokens = rng.integers(0, 20, (n_seq, n_res), dtype=np.int32)
    tokens[rng.random((n_seq, n_res)) < 0.1] = MSA_GAP
    del_matrix = rng.uniform(0, 3, (n_seq, n_res)).astype(np.float32)
    profile, deletion_mean = compute_profile(tokens)
    return MsaData(
        tokens=tokens,
        deletion_matrix=del_matrix,
        profile=profile,
        deletion_mean=deletion_mean,
        sequence_hash="abc123",
        tool="jackhammer",
        tool_version="3.3.2",
        database="uniref90",
        database_date="2022-01",
        created_at=time.time(),
    )


# ---------------------------------------------------------------------------
# compute_profile
# ---------------------------------------------------------------------------

def test_profile_shape():
    msa = _dummy_msa(n_seq=10, n_res=15)
    assert msa.profile.shape == (15, MSA_VOCAB_SIZE)
    assert msa.deletion_mean.shape == (15,)


def test_profile_sums_to_one():
    msa = _dummy_msa(n_seq=100, n_res=20)
    row_sums = msa.profile.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(20), atol=1e-5)


def test_profile_gap_column_matches_deletion_mean():
    msa = _dummy_msa()
    np.testing.assert_allclose(
        msa.profile[:, MSA_GAP], msa.deletion_mean, atol=1e-6
    )


# ---------------------------------------------------------------------------
# add_msa / read_msa / list_msas
# ---------------------------------------------------------------------------

def test_add_and_read_msa():
    from proteintensor import add_msa, read_msa
    msa = _dummy_msa(n_seq=30, n_res=20)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=20)
        add_msa(ptt, msa, source="uniref90")
        loaded = read_msa(ptt, source="uniref90")

    assert loaded.tokens.shape     == (30, 20)
    assert loaded.profile.shape    == (20, MSA_VOCAB_SIZE)
    np.testing.assert_array_equal(msa.tokens, loaded.tokens)
    np.testing.assert_array_almost_equal(msa.deletion_matrix, loaded.deletion_matrix)


def test_msa_provenance_roundtrip():
    from proteintensor import add_msa, read_msa
    msa = _dummy_msa()
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        add_msa(ptt, msa, source="bfd")
        loaded = read_msa(ptt, source="bfd")

    assert loaded.tool          == "jackhammer"
    assert loaded.tool_version  == "3.3.2"
    assert loaded.database      == "uniref90"
    assert loaded.database_date == "2022-01"
    assert loaded.sequence_hash == "abc123"


def test_list_msas_empty():
    from proteintensor import list_msas
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        assert list_msas(ptt) == []


def test_list_msas_multiple_sources():
    from proteintensor import add_msa, list_msas
    msa = _dummy_msa()
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        add_msa(ptt, msa, source="uniref90")
        add_msa(ptt, msa, source="bfd")
        sources = list_msas(ptt)

    assert set(sources) == {"uniref90", "bfd"}


def test_add_msa_does_not_overwrite_structure():
    """Adding MSA must not touch the existing structure data."""
    from proteintensor import add_msa, read
    msa = _dummy_msa(n_res=20)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp, n_res=20)
        before = read(ptt)
        add_msa(ptt, msa, source="uniref90")
        after  = read(ptt)

    np.testing.assert_array_equal(before.sequence_tokens, after.sequence_tokens)
    np.testing.assert_array_equal(before.atom_positions,  after.atom_positions)


def test_add_msa_overwrite_flag():
    from proteintensor import add_msa, read_msa
    msa1 = _dummy_msa(n_seq=10, n_res=20)
    msa2 = _dummy_msa(n_seq=25, n_res=20)
    msa2.sequence_hash = "different_hash"

    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        add_msa(ptt, msa1, source="uniref90")

        with pytest.raises(ValueError, match="already exists"):
            add_msa(ptt, msa2, source="uniref90", overwrite=False)

        add_msa(ptt, msa2, source="uniref90", overwrite=True)
        loaded = read_msa(ptt, source="uniref90")

    assert loaded.tokens.shape[0] == 25
    assert loaded.sequence_hash == "different_hash"


def test_read_msa_missing_source_error():
    from proteintensor import read_msa
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        with pytest.raises(KeyError, match="uniref90"):
            read_msa(ptt, source="uniref90")


def test_mmap_msa_tokens_is_lazy():
    import zarr
    from proteintensor import add_msa, mmap_msa_tokens
    msa = _dummy_msa(n_seq=20, n_res=20)
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _dummy_ptt(tmp)
        add_msa(ptt, msa, source="uniref90")
        arr = mmap_msa_tokens(ptt, source="uniref90")
        assert isinstance(arr, zarr.Array)
        assert arr.shape == (20, 20)


# ---------------------------------------------------------------------------
# A3M parser
# ---------------------------------------------------------------------------

_A3M_SIMPLE = """\
>query
ACDEFGHIKL
>seq1
ACDEFGHIKl
>seq2
-CDEFGHIKl
>seq3
ACDEFGHiikKL
"""

def test_from_a3m_shape():
    with tempfile.NamedTemporaryFile(suffix=".a3m", mode="w", delete=False) as f:
        f.write(_A3M_SIMPLE)
        fname = f.name
    msa = from_a3m(fname, tool="jackhammer", database="uniref90")
    assert msa.tokens.shape      == (4, 10)
    assert msa.deletion_matrix.shape == (4, 10)
    assert msa.profile.shape     == (10, MSA_VOCAB_SIZE)


def test_from_a3m_query_row_no_gaps():
    with tempfile.NamedTemporaryFile(suffix=".a3m", mode="w", delete=False) as f:
        f.write(_A3M_SIMPLE)
        fname = f.name
    msa = from_a3m(fname)
    # Query row (row 0) has no gaps
    assert MSA_GAP not in msa.tokens[0]


def test_from_a3m_gap_token():
    with tempfile.NamedTemporaryFile(suffix=".a3m", mode="w", delete=False) as f:
        f.write(_A3M_SIMPLE)
        fname = f.name
    msa = from_a3m(fname)
    # seq2 starts with '-' -> MSA_GAP at position 0
    assert msa.tokens[2, 0] == MSA_GAP


def test_from_a3m_insertion_deletion_count():
    """seq3 'ACDEFGHiikKL': 'iik' = 3 insertions before aligned col 7 (K)."""
    with tempfile.NamedTemporaryFile(suffix=".a3m", mode="w", delete=False) as f:
        f.write(_A3M_SIMPLE)
        fname = f.name
    msa = from_a3m(fname)
    assert msa.deletion_matrix[3, 7] == pytest.approx(3.0)


def test_from_a3m_provenance():
    with tempfile.NamedTemporaryFile(suffix=".a3m", mode="w", delete=False) as f:
        f.write(_A3M_SIMPLE)
        fname = f.name
    msa = from_a3m(fname, tool="hhblits", tool_version="3.3.0",
                   database="bfd", database_date="2021-06")
    assert msa.tool          == "hhblits"
    assert msa.tool_version  == "3.3.0"
    assert msa.database      == "bfd"
    assert msa.database_date == "2021-06"
    assert len(msa.sequence_hash) == 64   # SHA-256 hex digest


def test_char_to_token_coverage():
    """Standard AA one-letter codes must map to correct token indices."""
    from proteintensor.schema import AA_VOCAB
    from proteintensor.msa import _1TO3
    for letter, three in _1TO3.items():
        assert CHAR_TO_TOKEN[letter] == AA_VOCAB[three]
    assert CHAR_TO_TOKEN["-"] == MSA_GAP
