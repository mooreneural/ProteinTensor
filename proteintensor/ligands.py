"""
Small-molecule / ligand support for ProteinTensor.

Ligands (drugs, cofactors, ions, and any other non-polymer, non-water groups)
are stored under ``ligands/<index>/`` inside a .ptt store, one sub-group per
ligand instance:

    ligands/
      000000/
        .zattrs            name (CCD code), chain_id, res_num, smiles, num_atoms
        elements   [N]     S2      element symbols ("C", "N", "Mg", ...)
        positions  [N, 3]  float32 Angstrom coordinates
        b_factors  [N]     float32

The CCD code (``name``) is the reliable chemical identity; downstream tools such
as Boltz resolve the canonical bond graph from it. SMILES is stored only when
explicitly known - it is never inferred from coordinates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from .schema import LigandData
from .remote import open_store


# ---------------------------------------------------------------------------
# Extraction from a parsed structure
# ---------------------------------------------------------------------------

def extract_from_gemmi(structure) -> list[LigandData]:
    """Pull non-polymer, non-water residues from a gemmi Structure as ligands.

    Standard amino acids and nucleic acids (polymer residues) and water are
    excluded; everything else (drugs, cofactors, ions) becomes a LigandData.
    Only the first model is used.
    """
    import gemmi

    ligands: list[LigandData] = []
    model = structure[0]
    for chain in model:
        for res in chain:
            info = gemmi.find_tabulated_residue(res.name)
            if info is not None and (info.is_amino_acid() or info.is_nucleic_acid()):
                continue
            if res.is_water():
                continue

            elements: list[str] = []
            positions: list[list[float]] = []
            bfactors: list[float] = []
            for atom in res:
                elements.append(atom.element.name)
                positions.append([atom.pos.x, atom.pos.y, atom.pos.z])
                bfactors.append(float(atom.b_iso))
            if not elements:
                continue

            ligands.append(LigandData(
                name=res.name,
                elements=np.array(elements, dtype="S2"),
                positions=np.array(positions, dtype=np.float32).reshape(-1, 3),
                b_factors=np.array(bfactors, dtype=np.float32),
                chain_id=chain.name,
                res_num=int(res.seqid.num),
            ))
    return ligands


# ---------------------------------------------------------------------------
# SMILES input (RDKit) - the drug-screening direction
# ---------------------------------------------------------------------------

def from_smiles(smiles: str, name: str = "LIG", *, seed: int = 42) -> LigandData:
    """Build a LigandData from a SMILES string with an RDKit-generated 3D pose.

    Requires rdkit (pip install rdkit). Hydrogens are added for embedding then
    stripped, so the stored atoms are heavy atoms with a valid conformer.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError("rdkit is required for from_smiles: pip install rdkit") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    canonical = Chem.MolToSmiles(mol)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"3D embedding failed for SMILES: {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)

    conf = mol.GetConformer()
    elements = np.array([a.GetSymbol() for a in mol.GetAtoms()], dtype="S2")
    positions = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
    return LigandData(
        name=name,
        elements=elements,
        positions=positions,
        b_factors=np.zeros(len(elements), dtype=np.float32),
        smiles=canonical,
    )


# ---------------------------------------------------------------------------
# (De)serialization - used by writer.write() and reader.read()
# ---------------------------------------------------------------------------

def serialize_ligands(store: zarr.Group, ligands: list[LigandData], compressor) -> None:
    """Write a list of ligands under store['ligands/']."""
    root = store.require_group("ligands")
    names: list[str] = []
    for i, lig in enumerate(ligands):
        grp = root.require_group(f"{i:06d}")
        grp.create_dataset("elements",  data=lig.elements,                  dtype="S2",
                           compressor=compressor, overwrite=True)
        grp.create_dataset("positions", data=lig.positions.astype("float32"),
                           compressor=compressor, overwrite=True)
        grp.create_dataset("b_factors", data=lig.b_factors.astype("float32"),
                           compressor=compressor, overwrite=True)
        grp.attrs.update({
            "name":      lig.name,
            "chain_id":  lig.chain_id,
            "res_num":   int(lig.res_num),
            "smiles":    lig.smiles,
            "num_atoms": int(lig.num_atoms),
        })
        names.append(lig.name)
    store.attrs["ligands"] = names
    store.attrs["num_ligands"] = len(ligands)


def deserialize_ligands(grp: zarr.Group) -> list[LigandData]:
    """Read all ligands from a group's 'ligands/' sub-group (empty if none)."""
    if "ligands" not in grp:
        return []
    root = grp["ligands"]
    out: list[LigandData] = []
    for key in sorted(root.keys()):
        g = root[key]
        a = dict(g.attrs)
        out.append(LigandData(
            name=a.get("name", ""),
            elements=g["elements"][:],
            positions=g["positions"][:],
            b_factors=g["b_factors"][:],
            chain_id=a.get("chain_id", ""),
            res_num=int(a.get("res_num", 0)),
            smiles=a.get("smiles", ""),
        ))
    return out


# ---------------------------------------------------------------------------
# Public read / append API
# ---------------------------------------------------------------------------

def read_ligands(path: str | Path, storage_options: dict | None = None) -> list[LigandData]:
    """Load all ligands stored in a .ptt file."""
    return deserialize_ligands(open_store(path, storage_options=storage_options))


def list_ligands(path: str | Path, storage_options: dict | None = None) -> list[str]:
    """Return the CCD name of every ligand stored in a .ptt file."""
    store = open_store(path, storage_options=storage_options)
    if "ligands" not in store:
        return []
    return [str(store[f"ligands/{k}"].attrs.get("name", "")) for k in sorted(store["ligands"].keys())]


def add_ligand(path: str | Path, ligand: LigandData, compression: str = "blosc") -> int:
    """Append a single ligand to an existing .ptt file. Returns its index."""
    from .writer import _compressor

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    store = zarr.open(str(path), mode="r+")
    root = store.require_group("ligands")
    idx = len(list(root.keys()))
    compressor = _compressor(compression)

    grp = root.require_group(f"{idx:06d}")
    grp.create_dataset("elements",  data=ligand.elements,                  dtype="S2",
                       compressor=compressor, overwrite=True)
    grp.create_dataset("positions", data=ligand.positions.astype("float32"),
                       compressor=compressor, overwrite=True)
    grp.create_dataset("b_factors", data=ligand.b_factors.astype("float32"),
                       compressor=compressor, overwrite=True)
    grp.attrs.update({
        "name":      ligand.name,
        "chain_id":  ligand.chain_id,
        "res_num":   int(ligand.res_num),
        "smiles":    ligand.smiles,
        "num_atoms": int(ligand.num_atoms),
    })

    names = list(store.attrs.get("ligands", []))
    names.append(ligand.name)
    store.attrs["ligands"] = names
    store.attrs["num_ligands"] = len(names)
    return idx
