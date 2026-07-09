"""Tests for the pocket-centric ligand schema: bond graphs, atom chemistry,
protein-ligand interaction edges, and binding-site residues."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import proteintensor as pt
from proteintensor.schema import ProteinTensorData, LigandData


# --------------------------------------------------------------------------
# from_smiles bond graph + atom chemistry (RDKit)
# --------------------------------------------------------------------------

def test_from_smiles_has_bond_graph():
    pytest.importorskip("rdkit")
    lig = pt.from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="AIN")   # aspirin
    assert lig.has_bonds
    assert lig.bond_index.shape[0] == 2
    assert lig.bond_index.shape[1] == lig.bond_order.shape[0]
    # bidirectional: every edge appears both ways
    assert lig.bond_index.shape[1] % 2 == 0
    # aspirin has an aromatic benzene ring -> some aromatic atoms and bonds
    assert lig.is_aromatic.any()
    assert (lig.bond_order == 4).any()          # aromatic bonds
    assert (lig.bond_order == 2).any()          # C=O double bonds
    assert lig.formal_charge.shape[0] == lig.num_atoms


def test_from_smiles_bonds_reference_valid_atoms():
    pytest.importorskip("rdkit")
    lig = pt.from_smiles("c1ccccc1", name="BEN")   # benzene
    assert (lig.bond_index >= 0).all()
    assert (lig.bond_index < lig.num_atoms).all()


def test_ligand_bond_graph_roundtrip(tmp_path):
    pytest.importorskip("rdkit")
    p = Path(tmp_path) / "seq.ptt"
    pt.write(pt.from_sequence("MKTAYIAK", pdb_id="X"), p)
    lig = pt.from_smiles("CC(=O)Oc1ccccc1C(=O)O", name="AIN")
    pt.add_ligand(str(p), lig)

    loaded = pt.read_ligands(str(p))[0]
    assert loaded.has_bonds
    np.testing.assert_array_equal(loaded.bond_index, lig.bond_index)
    np.testing.assert_array_equal(loaded.bond_order, lig.bond_order)
    np.testing.assert_array_equal(loaded.is_aromatic, lig.is_aromatic)


# --------------------------------------------------------------------------
# protein-ligand interactions + binding site (self-contained geometry)
# --------------------------------------------------------------------------

def _protein_with_ligand(tmp) -> Path:
    n_res, apr = 5, 4
    n_atoms = n_res * apr
    pos = np.array([[r * 10.0, a * 1.0, 0.0] for r in range(n_res) for a in range(apr)],
                   dtype=np.float32)   # residues spaced 10 A along x
    data = ProteinTensorData(
        sequence_tokens=np.zeros(n_res, np.int32),
        residue_index=np.arange(n_res, dtype=np.int32),
        chain_id=np.array([b"A"] * n_res, dtype="S1"),
        atom_positions=pos,
        atom_mask=np.ones(n_atoms, bool),
        b_factors=np.zeros(n_atoms, np.float32),
        residue_atom_start=np.arange(0, n_atoms, apr, dtype=np.int32),
        residue_atom_count=np.full(n_res, apr, np.int32),
        pdb_id="TEST",
    )
    p = Path(tmp) / "complex.ptt"
    pt.write(data, p)
    # ligand sitting right next to residue 0 (near x=0), far from the rest
    lig = LigandData(
        name="LIG",
        elements=np.array([b"C", b"O"], dtype="S2"),
        positions=np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]], dtype=np.float32),
        b_factors=np.zeros(2, np.float32),
    )
    pt.add_ligand(str(p), lig)
    return p


def test_pocket_binding_site_and_interactions(tmp_path):
    p = _protein_with_ligand(tmp_path)
    mask = pt.compute_and_store_pocket(str(p), cutoff=5.0)

    assert mask.dtype == bool and mask.shape == (5,)
    assert mask[0] and not mask[1:].any()        # only residue 0 is in the pocket

    reloaded = pt.read_binding_site(str(p))
    np.testing.assert_array_equal(reloaded, mask)

    inter = pt.read_interactions(str(p))
    assert len(inter) == 1
    edges = inter[0]["edges"]                     # [2, N]  (ligand_atom, residue)
    assert edges.shape[0] == 2
    assert (edges[1] == 0).all()                  # every contact is to residue 0
    assert (inter[0]["dist"] < 5.0).all()


def test_read_binding_site_none_when_not_computed(tmp_path):
    p = _protein_with_ligand(tmp_path)
    assert pt.read_binding_site(str(p)) is None   # pocket not computed yet
    assert pt.read_interactions(str(p)) == []


def test_pocket_requires_structure(tmp_path):
    p = Path(tmp_path) / "seqonly.ptt"
    pt.write(pt.from_sequence("MKTAYIAK", pdb_id="X"), p)   # no protein atoms
    pt.add_ligand(str(p), LigandData(
        name="L", elements=np.array([b"C"], "S2"),
        positions=np.zeros((1, 3), np.float32), b_factors=np.zeros(1, np.float32)))
    with pytest.raises(KeyError):
        pt.compute_and_store_pocket(str(p))
