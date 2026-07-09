"""Tests for nucleic-acid (DNA/RNA) support in the converter and format."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import proteintensor as pt
from proteintensor.schema import (
    ProteinTensorData, NUC_VOCAB, MOL_PROTEIN, MOL_DNA, MOL_RNA,
)


def test_nuc_vocab_extends_past_amino_acids():
    # Nucleotide tokens live above the 21 amino-acid tokens (0-20).
    assert min(NUC_VOCAB.values()) == 21
    assert NUC_VOCAB["DA"] != NUC_VOCAB["A"]      # DNA vs RNA adenine distinct
    assert (MOL_PROTEIN, MOL_DNA, MOL_RNA) == (0, 1, 2)


def test_molecule_type_roundtrip(tmp_path):
    # A mixed DNA/RNA sequence-only entry round-trips losslessly.
    data = ProteinTensorData(
        sequence_tokens=np.array([25, 26, 27, 28, 21, 22], np.int32),  # A C G U DA DC
        residue_index=np.arange(6, dtype=np.int32),
        chain_id=np.array([b"A"] * 6, dtype="S1"),
        molecule_type=np.array([MOL_RNA] * 4 + [MOL_DNA] * 2, dtype=np.uint8),
    )
    p = Path(tmp_path) / "nuc.ptt"
    pt.write(data, p)
    r = pt.read(p)
    np.testing.assert_array_equal(r.molecule_type, data.molecule_type)
    np.testing.assert_array_equal(r.sequence_tokens, data.sequence_tokens)


def test_protein_only_has_no_molecule_type(tmp_path):
    p = Path(tmp_path) / "seq.ptt"
    pt.write(pt.from_sequence("MKTAYIAK", pdb_id="X"), p)   # protein
    assert pt.read(p).molecule_type is None


@pytest.mark.skipif(not Path("7LYJ.cif").exists(), reason="7LYJ.cif not present")
def test_rna_structure_parses():
    # 7LYJ is a pure-RNA structure that the protein-only parser used to reject.
    data = pt.from_mmcif("7LYJ.cif")
    assert data.num_residues > 0
    assert data.molecule_type is not None
    assert (data.molecule_type == MOL_RNA).all()
    # RNA residues tokenize into the nucleotide range (>= 21), not the AA range
    assert data.sequence_tokens.min() >= 21
