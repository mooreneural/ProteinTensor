"""Tests for small-molecule / ligand support."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import proteintensor as pt
from proteintensor.schema import LigandData


def _synthetic_ligand(name: str = "TST") -> LigandData:
    return LigandData(
        name=name,
        elements=np.array([b"C", b"N", b"O", b"Mg"], dtype="S2"),
        positions=np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1, 1, 1]], dtype=np.float32),
        b_factors=np.array([10, 20, 30, 40], dtype=np.float32),
        chain_id="A",
        res_num=42,
    )


# ---------------------------------------------------------------------------
# storage round-trip
# ---------------------------------------------------------------------------

def test_ligand_roundtrip(tmp_path):
    data = pt.from_sequence("MKTAYIAK")
    data.ligands = [_synthetic_ligand("LIG"), _synthetic_ligand("GDP")]
    ptt = tmp_path / "x.ptt"
    pt.write(data, str(ptt))

    loaded = pt.read(str(ptt))
    assert len(loaded.ligands) == 2
    l0 = loaded.ligands[0]
    assert l0.name == "LIG"
    assert l0.num_atoms == 4
    assert l0.chain_id == "A"
    assert l0.res_num == 42
    np.testing.assert_array_equal(l0.positions, _synthetic_ligand().positions)
    assert [e.decode() for e in l0.elements] == ["C", "N", "O", "Mg"]


def test_no_ligands_by_default(tmp_path):
    import zarr
    ptt = tmp_path / "x.ptt"
    pt.write(pt.from_sequence("MKT"), str(ptt))
    store = zarr.open(str(ptt), mode="r")
    assert "ligands" not in store
    assert pt.read(str(ptt)).ligands == []


def test_list_and_add_ligand(tmp_path):
    ptt = tmp_path / "x.ptt"
    pt.write(pt.from_sequence("MKT"), str(ptt))
    assert pt.read_ligands(str(ptt)) == []
    assert pt.list_ligands(str(ptt)) == []

    assert pt.add_ligand(str(ptt), _synthetic_ligand("NEW")) == 0
    assert pt.add_ligand(str(ptt), _synthetic_ligand("TWO")) == 1
    assert pt.list_ligands(str(ptt)) == ["NEW", "TWO"]
    ligs = pt.read_ligands(str(ptt))
    assert [l.name for l in ligs] == ["NEW", "TWO"]


# ---------------------------------------------------------------------------
# SMILES input (RDKit)
# ---------------------------------------------------------------------------

def test_from_smiles_aspirin():
    pytest.importorskip("rdkit")
    lig = pt.from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="AIN")
    assert lig.name == "AIN"
    assert lig.num_atoms == 13                    # aspirin has 13 heavy atoms
    assert lig.positions.shape == (13, 3)
    assert lig.smiles                             # canonical SMILES recorded
    assert not any(e == b"H" for e in lig.elements)  # heavy atoms only


def test_from_smiles_invalid():
    pytest.importorskip("rdkit")
    with pytest.raises(ValueError):
        pt.from_smiles("this-is-not-smiles!!!")


def test_from_smiles_roundtrips_into_ptt(tmp_path):
    pytest.importorskip("rdkit")
    ptt = tmp_path / "x.ptt"
    pt.write(pt.from_sequence("MKT"), str(ptt))
    pt.add_ligand(str(ptt), pt.from_smiles("c1ccccc1", name="BNZ"))  # benzene
    ligs = pt.read_ligands(str(ptt))
    assert ligs[0].name == "BNZ"
    assert ligs[0].num_atoms == 6


# ---------------------------------------------------------------------------
# extraction from a real structure
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Path("3HTB.cif").exists(), reason="3HTB.cif not present")
def test_extract_ligands_from_mmcif(tmp_path):
    from proteintensor.converters.mmcif import from_mmcif

    data = from_mmcif("3HTB.cif", include_ligands=True)
    names = [l.name for l in data.ligands]
    assert "JZ4" in names                 # the drug ligand in 3HTB
    assert data.num_residues > 0          # protein still parsed
    assert all(l.name != "HOH" for l in data.ligands)  # water excluded

    jz4 = next(l for l in data.ligands if l.name == "JZ4")
    assert jz4.num_atoms == 10

    ptt = tmp_path / "3htb.ptt"
    pt.write(data, str(ptt))
    assert "JZ4" in pt.list_ligands(str(ptt))


@pytest.mark.skipif(not Path("6OIM.cif").exists(), reason="6OIM.cif not present")
def test_include_ligands_defaults_off():
    from proteintensor.converters.mmcif import from_mmcif
    data = from_mmcif("6OIM.cif")     # include_ligands defaults False
    assert data.ligands == []
