"""
Tests for the Boltz adapter.

write_input() tests run without Boltz installed.
The predict() integration test is skipped when boltz is not present or
when no GPU is available (CI-safe).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_ptt(tmp: str, n_res_per_chain: dict[str, int], with_msa: bool = False) -> Path:
    """Write a multi-chain .ptt with optional mock MSA."""
    from proteintensor.schema import ProteinTensorData, N_BACKBONE
    from proteintensor import write, add_msa
    from proteintensor.msa import MsaData, compute_profile, MSA_GAP
    import hashlib, time as t

    rng = np.random.default_rng(99)
    chains = sorted(n_res_per_chain.keys())
    total_res = sum(n_res_per_chain.values())
    n_atoms = total_res * 4

    toks  = rng.integers(0, 19, total_res, dtype=np.int32)
    chain_labels = np.array(
        [c.encode() for c, n in sorted(n_res_per_chain.items()) for _ in range(n)],
        dtype="S1",
    )
    bb = rng.standard_normal((total_res, N_BACKBONE, 3)).astype(np.float32)

    data = ProteinTensorData(
        sequence_tokens=toks,
        residue_index=np.arange(total_res, dtype=np.int32),
        chain_id=chain_labels,
        atom_positions=rng.standard_normal((n_atoms, 3)).astype(np.float32),
        atom_mask=np.ones(n_atoms, dtype=bool),
        b_factors=np.zeros(n_atoms, dtype=np.float32),
        residue_atom_start=np.arange(0, n_atoms, 4, dtype=np.int32),
        residue_atom_count=np.full(total_res, 4, dtype=np.int32),
        backbone_positions=bb,
        backbone_mask=np.ones((total_res, N_BACKBONE), dtype=bool),
        pdb_id="TEST",
    )
    p = Path(tmp) / "test.ptt"
    write(data, p)

    if with_msa:
        n_seq = 64
        msa_toks = rng.integers(0, 20, (n_seq, total_res), dtype=np.int32)
        msa_toks[rng.random((n_seq, total_res)) < 0.1] = MSA_GAP
        del_mat  = rng.uniform(0, 1, (n_seq, total_res)).astype(np.float32)
        prof, dm = compute_profile(msa_toks)
        msa = MsaData(msa_toks, del_mat, prof, dm,
                      hashlib.sha256(b"test").hexdigest(),
                      "jackhammer", "3.3.2", "uniref90", "2024-01", t.time())
        add_msa(p, msa, source="uniref90")

    return p


# ---------------------------------------------------------------------------
# chains()
# ---------------------------------------------------------------------------

def test_chains_single_chain():
    from proteintensor import BoltzAdapter
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, {"A": 20})
        chains = BoltzAdapter(ptt).chains()
    assert set(chains.keys()) == {"A"}
    assert len(chains["A"]) == 20
    assert all(c in "ACDEFGHIKLMNPQRSTVWYX" for c in chains["A"])


def test_chains_multi_chain():
    from proteintensor import BoltzAdapter
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, {"A": 15, "B": 10})
        chains = BoltzAdapter(ptt).chains()
    assert set(chains.keys()) == {"A", "B"}
    assert len(chains["A"]) == 15
    assert len(chains["B"]) == 10


# ---------------------------------------------------------------------------
# write_input() - YAML structure
# ---------------------------------------------------------------------------

def test_write_input_creates_yaml():
    from proteintensor import BoltzAdapter
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, {"A": 20})
        out = Path(tmp) / "boltz"
        yaml_path = BoltzAdapter(ptt).write_input(out)
        assert yaml_path.exists()
        assert yaml_path.suffix == ".yaml"


def test_write_input_yaml_structure():
    from proteintensor import BoltzAdapter
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, {"A": 10, "B": 8})
        out = Path(tmp) / "boltz"
        yaml_path = BoltzAdapter(ptt).write_input(out)
        doc = yaml.safe_load(yaml_path.read_text())

    assert "sequences" in doc
    assert doc["version"] == 1
    assert len(doc["sequences"]) == 2
    # Each entry must have id + sequence
    for entry in doc["sequences"]:
        assert "protein" in entry
        p = entry["protein"]
        assert "id" in p and "sequence" in p


def test_write_input_sequence_length_matches():
    from proteintensor import BoltzAdapter
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, {"A": 12, "B": 7})
        out = Path(tmp) / "boltz"
        yaml_path = BoltzAdapter(ptt).write_input(out)
        doc  = yaml.safe_load(yaml_path.read_text())
        seqs = {e["protein"]["id"]: e["protein"]["sequence"] for e in doc["sequences"]}

    assert len(seqs["A"]) == 12
    assert len(seqs["B"]) == 7


def test_write_input_no_msa_writes_single_seq_a3m():
    """Without MSA data, write_input must write a single-sequence A3M so
    Boltz never needs --use_msa_server."""
    from proteintensor import BoltzAdapter
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, {"A": 15}, with_msa=False)
        out = Path(tmp) / "boltz"
        yaml_path = BoltzAdapter(ptt).write_input(out)
        doc = yaml.safe_load(yaml_path.read_text())

        # msa key must be present (single-sequence A3M)
        for entry in doc["sequences"]:
            assert "msa" in entry["protein"]

        # A3M must exist and contain exactly the query
        a3m = (out / "msa" / "A.a3m").read_text()
        lines = [l for l in a3m.splitlines() if l]
        assert lines[0] == ">query"
        assert len(lines[1]) == 15   # one row = query sequence


def test_write_input_with_msa_creates_a3m():
    """When MSA is present, write_input must create one A3M per chain."""
    from proteintensor import BoltzAdapter
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, {"A": 10, "B": 8}, with_msa=True)
        out = Path(tmp) / "boltz"
        yaml_path = BoltzAdapter(ptt).write_input(out, msa_source="uniref90")
        doc = yaml.safe_load(yaml_path.read_text())

        msa_dir = out / "msa"
        assert msa_dir.exists()
        assert (msa_dir / "A.a3m").exists()
        assert (msa_dir / "B.a3m").exists()

        for entry in doc["sequences"]:
            assert "msa" in entry["protein"]


def test_write_input_a3m_content():
    """A3M files must start with >query and have N_res columns."""
    from proteintensor import BoltzAdapter
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, {"A": 12}, with_msa=True)
        out = Path(tmp) / "boltz"
        BoltzAdapter(ptt).write_input(out, msa_source="uniref90")
        a3m = (out / "msa" / "A.a3m").read_text()

    lines = [l for l in a3m.splitlines() if l]
    assert lines[0] == ">query"
    query_seq = lines[1]
    assert len(query_seq) == 12
    # Subsequent sequence lines must have same length
    seq_lines = [lines[i] for i in range(1, len(lines), 2)]
    for s in seq_lines:
        assert len(s) == 12


def test_write_input_max_msa_seqs_respected():
    """max_msa_seqs must cap the number of sequences in the A3M."""
    from proteintensor import BoltzAdapter
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, {"A": 10}, with_msa=True)  # 64 seqs in mock MSA
        out = Path(tmp) / "boltz"
        BoltzAdapter(ptt).write_input(out, msa_source="uniref90", max_msa_seqs=10)
        a3m = (out / "msa" / "A.a3m").read_text()

    headers = [l for l in a3m.splitlines() if l.startswith(">")]
    assert len(headers) <= 10


def test_write_input_idempotent():
    """Calling write_input twice with the same dir overwrites cleanly."""
    from proteintensor import BoltzAdapter
    with tempfile.TemporaryDirectory() as tmp:
        ptt = _make_ptt(tmp, {"A": 10})
        out = Path(tmp) / "boltz"
        adapter = BoltzAdapter(ptt)
        p1 = adapter.write_input(out)
        p2 = adapter.write_input(out)
    assert p1 == p2


# ---------------------------------------------------------------------------
# Real-protein integration (uses downloaded .cif files)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("1UBQ.cif").exists(),
    reason="1UBQ.cif not present",
)
def test_write_input_real_ubiquitin():
    """Write Boltz input for a real protein and verify YAML correctness."""
    from proteintensor.converters import from_mmcif
    from proteintensor import write, BoltzAdapter

    with tempfile.TemporaryDirectory() as tmp:
        data = from_mmcif(Path("1UBQ.cif"))
        ptt  = Path(tmp) / "1UBQ.ptt"
        write(data, ptt)

        out       = Path(tmp) / "boltz"
        yaml_path = BoltzAdapter(ptt).write_input(out)
        doc       = yaml.safe_load(yaml_path.read_text())

    assert len(doc["sequences"]) >= 1
    seq = doc["sequences"][0]["protein"]["sequence"]
    assert len(seq) == 76                         # ubiquitin is 76 residues
    assert all(c in "ACDEFGHIKLMNPQRSTVWYX" for c in seq)


@pytest.mark.skipif(
    not Path("4HHB.cif").exists(),
    reason="4HHB.cif not present",
)
def test_write_input_real_hemoglobin_chains():
    """Hemoglobin (4HHB) has 4 chains: A, B, C, D."""
    from proteintensor.converters import from_mmcif
    from proteintensor import write, BoltzAdapter

    with tempfile.TemporaryDirectory() as tmp:
        data = from_mmcif(Path("4HHB.cif"))
        ptt  = Path(tmp) / "4HHB.ptt"
        write(data, ptt)

        out = Path(tmp) / "boltz"
        doc = yaml.safe_load(BoltzAdapter(ptt).write_input(out).read_text())

    chain_ids = {e["protein"]["id"] for e in doc["sequences"]}
    assert chain_ids == {"A", "B", "C", "D"}
