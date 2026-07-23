"""Tests for the Nesso-1 input adapter (`.ptt` -> Nesso prediction YAML)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import proteintensor as pt
from proteintensor import NessoAdapter


def _ptt_with_ligand(tmp) -> Path:
    p = Path(tmp) / "x.ptt"
    pt.write(pt.from_sequence("MKTAYIAKQR", pdb_id="X", chain_id="A"), p)
    lig = pt.from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="AIN")   # aspirin
    pt.add_ligand(str(p), lig)
    return p


def test_write_input_matches_nesso_schema(tmp_path):
    out = NessoAdapter(str(_ptt_with_ligand(tmp_path))).write_input(tmp_path / "n.yaml")
    assert out.exists()
    data = yaml.safe_load(out.read_text())

    proteins = [s["protein"] for s in data["sequences"] if "protein" in s]
    assert proteins[0]["id"] == "A"
    assert proteins[0]["sequence"] == "MKTAYIAKQR"

    ligands = [s["ligand"] for s in data["sequences"] if "ligand" in s]
    assert len(ligands) == 1
    assert ligands[0]["id"] != "A"          # ligand id must not collide with a chain
    assert ligands[0]["smiles"]

    # affinity property targets the ligand
    assert data["properties"][0]["affinity"]["binder"] == ligands[0]["id"]


def test_no_ligand_no_properties(tmp_path):
    p = Path(tmp_path) / "nolig.ptt"
    pt.write(pt.from_sequence("MKTAYIAKQR"), p)
    data = yaml.safe_load(NessoAdapter(str(p)).write_input(tmp_path / "n.yaml").read_text())
    assert "properties" not in data
    assert all("ligand" not in s for s in data["sequences"])


def test_affinity_flag_off(tmp_path):
    out = NessoAdapter(str(_ptt_with_ligand(tmp_path))).write_input(
        tmp_path / "n.yaml", affinity=False)
    data = yaml.safe_load(out.read_text())
    assert "properties" not in data
    # ligand still present, just no affinity target
    assert any("ligand" in s for s in data["sequences"])


def test_chains_helper(tmp_path):
    assert NessoAdapter(str(_ptt_with_ligand(tmp_path))).chains() == {"A": "MKTAYIAKQR"}


def test_predict_raises(tmp_path):
    with pytest.raises(NotImplementedError):
        NessoAdapter(str(_ptt_with_ligand(tmp_path))).predict()


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        NessoAdapter("does_not_exist.ptt")
