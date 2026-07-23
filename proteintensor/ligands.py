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

    # Exact bond graph + atom chemistry from RDKit (bidirectional edges).
    _ORDER = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
    bi, bj, orders = [], [], []
    for b in mol.GetBonds():
        a, z = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        o = 4 if b.GetIsAromatic() else _ORDER.get(b.GetBondType(), 1)
        bi += [a, z]; bj += [z, a]; orders += [o, o]
    bond_index = np.array([bi, bj], dtype=np.int32) if bi else np.zeros((2, 0), np.int32)
    bond_order = np.array(orders, dtype=np.uint8)
    formal_charge = np.array([a.GetFormalCharge() for a in mol.GetAtoms()], dtype=np.int8)
    is_aromatic = np.array([a.GetIsAromatic() for a in mol.GetAtoms()], dtype=bool)

    return LigandData(
        name=name,
        elements=elements,
        positions=positions,
        b_factors=np.zeros(len(elements), dtype=np.float32),
        smiles=canonical,
        bond_index=bond_index,
        bond_order=bond_order,
        formal_charge=formal_charge,
        is_aromatic=is_aromatic,
    )


# ---------------------------------------------------------------------------
# (De)serialization - used by writer.write() and reader.read()
# ---------------------------------------------------------------------------

def _write_ligand_group(grp: zarr.Group, lig: LigandData, compressor) -> None:
    """Write one ligand's arrays + metadata into an (already created) group."""
    grp.create_dataset("elements",  data=lig.elements,                  dtype="S2",
                       compressor=compressor, overwrite=True)
    grp.create_dataset("positions", data=lig.positions.astype("float32"),
                       compressor=compressor, overwrite=True)
    grp.create_dataset("b_factors", data=lig.b_factors.astype("float32"),
                       compressor=compressor, overwrite=True)
    if lig.bond_index is not None:
        grp.create_dataset("bond_index", data=lig.bond_index.astype("int32"),
                           dtype="int32", compressor=compressor, overwrite=True)
        grp.create_dataset("bond_order", data=lig.bond_order.astype("uint8"),
                           dtype="uint8", compressor=compressor, overwrite=True)
    if lig.formal_charge is not None:
        grp.create_dataset("formal_charge", data=lig.formal_charge.astype("int8"),
                           dtype="int8", compressor=compressor, overwrite=True)
    if lig.is_aromatic is not None:
        grp.create_dataset("is_aromatic", data=lig.is_aromatic.astype(bool),
                           dtype="bool", compressor=compressor, overwrite=True)
    grp.attrs.update({
        "name":      lig.name,
        "chain_id":  lig.chain_id,
        "res_num":   int(lig.res_num),
        "smiles":    lig.smiles,
        "num_atoms": int(lig.num_atoms),
        "has_bonds": lig.bond_index is not None,
    })


def serialize_ligands(store: zarr.Group, ligands: list[LigandData], compressor) -> None:
    """Write a list of ligands under store['ligands/']."""
    root = store.require_group("ligands")
    for i, lig in enumerate(ligands):
        _write_ligand_group(root.require_group(f"{i:06d}"), lig, compressor)
    store.attrs["ligands"] = [lig.name for lig in ligands]
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
            bond_index=g["bond_index"][:] if "bond_index" in g else None,
            bond_order=g["bond_order"][:] if "bond_order" in g else None,
            formal_charge=g["formal_charge"][:] if "formal_charge" in g else None,
            is_aromatic=g["is_aromatic"][:] if "is_aromatic" in g else None,
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
    _write_ligand_group(root.require_group(f"{idx:06d}"), ligand, _compressor(compression))

    names = list(store.attrs.get("ligands", []))
    names.append(ligand.name)
    store.attrs["ligands"] = names
    store.attrs["num_ligands"] = len(names)
    return idx


def resolve_ligand_smiles(
    path: str | Path,
    *,
    allow_network: bool = True,
    cache_dir: str | None = None,
) -> int:
    """Fill in SMILES for stored ligands that lack one, from their CCD code.

    Structure-extracted (CCD) ligands store elements + coordinates but no SMILES,
    so they don't flow into ligand-consuming adapters (Chai / AF3 / Nesso). This
    resolves each ligand's 3-letter code to canonical SMILES via the RCSB
    Chemical Component Dictionary (cached), writing it back to the `.ptt`.

    Requires network on first lookup of a code (opt-in). Returns the number of
    ligands newly resolved. SMILES are never guessed - only fetched from RCSB.
    """
    from .ccd import ccd_to_smiles

    path = Path(path)
    store = zarr.open(str(path), mode="r+")
    if "ligands" not in store:
        return 0
    resolved = 0
    for key in sorted(store["ligands"].keys()):
        grp = store[f"ligands/{key}"]
        if grp.attrs.get("smiles"):
            continue
        code = str(grp.attrs.get("name", "")).strip()
        smi = ccd_to_smiles(code, allow_network=allow_network, cache_dir=cache_dir)
        if smi:
            grp.attrs["smiles"] = smi
            resolved += 1
    return resolved


# ---------------------------------------------------------------------------
# Pocket: protein-ligand interactions + binding-site residues
# ---------------------------------------------------------------------------

def compute_and_store_pocket(
    path: str | Path,
    *,
    cutoff: float = 5.0,
    compression: str = "blosc",
) -> np.ndarray:
    """Compute protein-ligand interactions and binding-site residues.

    For every stored ligand, finds (ligand atom, protein residue) contacts within
    ``cutoff`` Angstroms (all-atom), storing per-ligand interaction edges under
    ``ligands/<i>/`` and a union binding-site residue mask under ``pocket/``.
    Requires a .ptt with both protein structure and ligands. Returns the mask.
    """
    from .writer import _compressor

    path = Path(path)
    store = zarr.open(str(path), mode="r+")
    if "ligands" not in store:
        raise KeyError("No ligands in this .ptt.")
    if "atoms" not in store or "structure" not in store:
        raise KeyError("Pocket needs protein structure (atoms + residue mapping).")

    prot = store["atoms/positions"][:]                      # [N_prot, 3]
    res_start = store["structure/residue_atom_start"][:]
    res_count = store["structure/residue_atom_count"][:]
    n_res = int(res_start.shape[0])

    atom_res = np.empty(prot.shape[0], dtype=np.int32)      # protein atom -> residue
    for r in range(n_res):
        atom_res[res_start[r]:res_start[r] + res_count[r]] = r

    binding = np.zeros(n_res, dtype=bool)
    compressor = _compressor(compression)
    root = store["ligands"]
    for key in sorted(root.keys()):
        g = root[key]
        lig = g["positions"][:]                             # [N_lig, 3]
        d = np.sqrt(((lig[:, None, :] - prot[None, :, :]) ** 2).sum(-1))  # [N_lig, N_prot]
        close = d < cutoff
        binding[atom_res[close.any(axis=0)]] = True

        li, pj = np.nonzero(close)
        if li.size:
            rj = atom_res[pj]
            dij = d[li, pj]
            keys = li.astype(np.int64) * n_res + rj
            order = np.argsort(dij, kind="stable")          # min distance first
            _, first = np.unique(keys[order], return_index=True)
            sel = order[first]
            edges = np.stack([li[sel], rj[sel]]).astype(np.int32)   # [2, N_edge]
            edist = dij[sel].astype(np.float32)
        else:
            edges = np.zeros((2, 0), dtype=np.int32)
            edist = np.zeros(0, dtype=np.float32)

        g.create_dataset("interaction_edges", data=edges, dtype="int32",
                         compressor=compressor, overwrite=True)
        g.create_dataset("interaction_dist", data=edist, dtype="float32",
                         compressor=compressor, overwrite=True)
        g.attrs["interaction_cutoff"] = float(cutoff)

    pocket = store.require_group("pocket")
    pocket.create_dataset("binding_site", data=binding, dtype="bool",
                          compressor=compressor, overwrite=True)
    pocket.attrs["cutoff"] = float(cutoff)
    pocket.attrs["num_binding_residues"] = int(binding.sum())
    return binding


def read_binding_site(path: str | Path, storage_options: dict | None = None) -> np.ndarray | None:
    """Return the [N_res] bool binding-site mask, or None if not computed."""
    store = open_store(path, storage_options=storage_options)
    if "pocket" not in store:
        return None
    return store["pocket/binding_site"][:]


def read_interactions(path: str | Path, storage_options: dict | None = None) -> list[dict]:
    """Return per-ligand interaction edges: [{name, edges [2,N], dist [N]}, ...]."""
    store = open_store(path, storage_options=storage_options)
    if "ligands" not in store:
        return []
    out: list[dict] = []
    root = store["ligands"]
    for key in sorted(root.keys()):
        g = root[key]
        if "interaction_edges" not in g:
            continue
        out.append({
            "name": g.attrs.get("name", ""),
            "edges": g["interaction_edges"][:],   # [2, N]  (ligand_atom, residue_index)
            "dist":  g["interaction_dist"][:],    # [N]     Angstroms
        })
    return out
