"""Tests for the .ptt -> model input adapters (AlphaFold 3, Chai-1, OpenFold).

These validate the emitted input FORMAT (parse it back, check the schema). The
models are not bundled, so end-to-end prediction is not tested here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import proteintensor as pt
from proteintensor import AlphaFold3Adapter, ChaiAdapter, OpenFoldAdapter

UBIQ = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")


def _ptt_with_msa(tmp, seq=UBIQ, pdb_id="UBQ"):
    from proteintensor.msa import MsaData, compute_profile, MSA_GAP
    p = Path(tmp) / f"{pdb_id}.ptt"
    data = pt.from_sequence(seq, pdb_id=pdb_id)
    pt.write(data, p)

    n = len(seq)
    depth = 8
    rng = np.random.default_rng(0)
    tokens = np.tile(data.sequence_tokens, (depth, 1)).astype(np.int32)
    tokens[rng.random((depth, n)) < 0.1] = MSA_GAP
    tokens[0] = data.sequence_tokens          # query row = exact sequence
    prof, dmean = compute_profile(tokens)
    msa = MsaData(tokens, np.zeros((depth, n), np.float32), prof, dmean,
                  "hash", "tool", "v", "db", "date")
    pt.add_msa(str(p), msa, source="default")
    return p


# --------------------------------------------------------------------------
# AlphaFold 3
# --------------------------------------------------------------------------

def test_af3_input_schema(tmp_path):
    out = AlphaFold3Adapter(_ptt_with_msa(tmp_path)).write_input(tmp_path / "af3.json")
    d = json.loads(out.read_text())
    assert d["dialect"] == "alphafold3"
    assert d["version"] in (1, 2, 3, 4)
    assert d["modelSeeds"]
    prot = d["sequences"][0]["protein"]
    assert prot["id"] == "A"
    assert prot["sequence"] == UBIQ
    assert prot["unpairedMsa"].startswith(">query")   # cached MSA embedded


def test_af3_no_msa_omits_unpaired(tmp_path):
    p = Path(tmp_path) / "seq.ptt"
    pt.write(pt.from_sequence(UBIQ, pdb_id="X"), p)   # no MSA cached
    d = json.loads(AlphaFold3Adapter(p).write_input(tmp_path / "x.json").read_text())
    assert "unpairedMsa" not in d["sequences"][0]["protein"]


# --------------------------------------------------------------------------
# Chai-1
# --------------------------------------------------------------------------

def test_chai_fasta_and_msa(tmp_path):
    fasta = ChaiAdapter(_ptt_with_msa(tmp_path)).write_input(tmp_path / "chai")
    text = fasta.read_text()
    assert ">protein|name=UBQ_A" in text
    assert UBIQ in text
    assert (fasta.parent / "msa" / "UBQ_A.a3m").exists()


# --------------------------------------------------------------------------
# OpenFold
# --------------------------------------------------------------------------

def test_openfold_fasta_and_alignments(tmp_path):
    fasta = OpenFoldAdapter(_ptt_with_msa(tmp_path)).write_input(tmp_path / "of")
    text = fasta.read_text()
    assert ">UBQ_A" in text and UBIQ in text
    a3m = fasta.parent / "alignments" / "UBQ_A" / "uniref90_hits.a3m"
    assert a3m.exists() and a3m.read_text().startswith(">query")


# --------------------------------------------------------------------------
# multi-chain + ligands + honest predict() guard
# --------------------------------------------------------------------------

def test_multichain_fasta(tmp_path):
    p = Path(tmp_path) / "cplx.ptt"
    pt.write(pt.from_fasta(_write_fasta(tmp_path)), p)
    chains = ChaiAdapter(p).chains()
    assert set(chains) == {"A", "B"}


def _write_fasta(tmp):
    f = Path(tmp) / "in.fasta"
    f.write_text(">a\nMKTAYIAKQR\n>b\nQRLLGKPFSAED\n")
    return f


def test_chai_and_af3_emit_ligand(tmp_path):
    pytest.importorskip("rdkit")
    p = _ptt_with_msa(tmp_path)
    pt.add_ligand(str(p), pt.from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="AIN"))

    assert ">ligand|name=AIN" in ChaiAdapter(p).write_input(tmp_path / "c2").read_text()

    d = json.loads(AlphaFold3Adapter(p).write_input(tmp_path / "a2.json").read_text())
    ligs = [s["ligand"] for s in d["sequences"] if "ligand" in s]
    assert ligs and ligs[0]["smiles"]


@pytest.mark.parametrize("cls", [AlphaFold3Adapter, ChaiAdapter, OpenFoldAdapter])
def test_predict_not_bundled(tmp_path, cls):
    adapter = cls(_ptt_with_msa(tmp_path))
    with pytest.raises(NotImplementedError):
        adapter.predict()
