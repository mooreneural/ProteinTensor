import numpy as np
import pytest

import proteintensor as pt
from proteintensor.schema import (
    AA_VOCAB, AA_UNK, AA_1LETTER, ONE_LETTER_TO_TOKEN,
    sequence_to_tokens, tokens_to_sequence,
)
from proteintensor.converters.sequence import from_sequence, from_fasta, parse_fasta


UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


# --------------------------------------------------------------------------
# vocab consistency (guards the AA_1LETTER bug that was fixed)
# --------------------------------------------------------------------------

def test_aa_1letter_matches_vocab_order():
    # AA_1LETTER[token] must be the 1-letter code for that token.
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    for three, token in AA_VOCAB.items():
        if three == "UNK":
            assert AA_1LETTER[token] == "X"
        else:
            assert AA_1LETTER[token] == three_to_one[three]


def test_token_roundtrip_is_identity():
    toks = sequence_to_tokens(UBIQUITIN)
    assert tokens_to_sequence(toks) == UBIQUITIN


# --------------------------------------------------------------------------
# from_sequence
# --------------------------------------------------------------------------

def test_from_sequence_basic():
    data = from_sequence(UBIQUITIN, pdb_id="UBQ", chain_id="A")
    assert data.num_residues == len(UBIQUITIN)
    assert data.has_structure is False
    assert data.atom_positions is None
    assert data.backbone_positions is None
    assert data.sequence_tokens.dtype == np.int32
    assert data.residue_index[0] == 1
    assert data.residue_index[-1] == len(UBIQUITIN)
    assert set(data.chain_id.tolist()) == {b"A"}


def test_from_sequence_whitespace_ignored():
    a = from_sequence("MKT AYI\nAKQR")
    b = from_sequence("MKTAYIAKQR")
    np.testing.assert_array_equal(a.sequence_tokens, b.sequence_tokens)


def test_from_sequence_unknown_chars_map_to_unk():
    # B, Z, J, O, X, and gap chars are not standard residues -> UNK
    data = from_sequence("ABXZ-J")
    assert data.sequence_tokens[0] == AA_VOCAB["ALA"]
    assert (data.sequence_tokens[1:] == AA_UNK).all()


def test_from_sequence_empty_raises():
    with pytest.raises(ValueError):
        from_sequence("   \n  ")


def test_from_sequence_residue_start():
    data = from_sequence("MKT", residue_start=100)
    np.testing.assert_array_equal(data.residue_index, [100, 101, 102])


# --------------------------------------------------------------------------
# round-trip through .ptt (write -> read) for a sequence-only entry
# --------------------------------------------------------------------------

def test_sequence_only_roundtrip(tmp_path):
    data = from_sequence(UBIQUITIN, pdb_id="UBQ")
    ptt = tmp_path / "ubq_seq.ptt"
    pt.write(data, str(ptt))

    loaded = pt.read(str(ptt))
    assert loaded.has_structure is False
    assert loaded.atom_positions is None
    assert loaded.residue_atom_start is None
    np.testing.assert_array_equal(loaded.sequence_tokens, data.sequence_tokens)
    np.testing.assert_array_equal(loaded.residue_index, data.residue_index)
    assert loaded.pdb_id == "UBQ"


def test_sequence_only_has_structure_flag_in_store(tmp_path):
    import zarr
    ptt = tmp_path / "seq.ptt"
    pt.write(from_sequence("MKTAYIAKQR"), str(ptt))
    store = zarr.open(str(ptt), mode="r")
    assert store.attrs["has_structure"] is False
    assert store.attrs["num_atoms"] == 0
    assert "atoms" not in store
    assert "structure" not in store
    assert "sequence" in store


def test_read_backbone_on_sequence_only_raises(tmp_path):
    ptt = tmp_path / "seq.ptt"
    pt.write(from_sequence("MKTAYIAKQR"), str(ptt))
    with pytest.raises(KeyError):
        pt.read_backbone(str(ptt))


# --------------------------------------------------------------------------
# FASTA parsing
# --------------------------------------------------------------------------

def test_parse_fasta_multi():
    text = ">chainA\nMKTA\nYIAK\n>chainB\nQRLL\n"
    recs = parse_fasta(text)
    assert recs == [("chainA", "MKTAYIAK"), ("chainB", "QRLL")]


def test_from_fasta_single(tmp_path):
    fasta = tmp_path / "ubq.fasta"
    fasta.write_text(f">UBQ\n{UBIQUITIN}\n")
    data = from_fasta(fasta)
    assert data.num_residues == len(UBIQUITIN)
    assert data.pdb_id == "UBQ"
    assert set(data.chain_id.tolist()) == {b"A"}


def test_from_fasta_multichain(tmp_path):
    fasta = tmp_path / "complex.fasta"
    fasta.write_text(">a\nMKTAY\n>b\nQRLLG\n")
    data = from_fasta(fasta)
    assert data.num_residues == 10
    # two chains, A and B, residue numbering restarts per chain
    assert set(data.chain_id.tolist()) == {b"A", b"B"}
    np.testing.assert_array_equal(data.residue_index, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])


def test_from_fasta_empty_raises(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("\n\n")
    with pytest.raises(ValueError):
        from_fasta(fasta)
