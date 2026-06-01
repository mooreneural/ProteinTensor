"""
Bond type constants and covalent bond tables for the 20 standard amino acids.

Graph convention: bidirectional — every bond is stored as both (u->v) and (v->u),
matching PyTorch Geometric / DGL expectations.
"""
from __future__ import annotations
import numpy as np

# Bond type encoding (stored as uint8)
BOND_SINGLE    = 1
BOND_DOUBLE    = 2
BOND_TRIPLE    = 3
BOND_AROMATIC  = 4
BOND_PEPTIDE   = 5  # inter-residue C(i) - N(i+1)
BOND_DISULFIDE = 6  # CYS SG - SG, distance < 2.5 Å

BOND_TYPE_NAMES = {
    1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE",
    4: "AROMATIC", 5: "PEPTIDE", 6: "DISULFIDE",
}

# Covalent bond tables for each residue type.
# Each entry: (atom1_name, atom2_name, bond_type)
# Backbone bonds (N-CA, CA-C, C=O) are prepended automatically in build().
_BB = [
    ("N",  "CA", BOND_SINGLE),
    ("CA", "C",  BOND_SINGLE),
    ("C",  "O",  BOND_DOUBLE),
]

RESIDUE_BONDS: dict[str, list[tuple[str, str, int]]] = {
    "ALA": _BB + [("CA", "CB", BOND_SINGLE)],
    "ARG": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1), ("CG", "CD", 1),
        ("CD", "NE", 1), ("NE", "CZ", 1),
        ("CZ", "NH1", 1), ("CZ", "NH2", 1),
    ],
    "ASN": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1),
        ("CG", "OD1", 2), ("CG", "ND2", 1),
    ],
    "ASP": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1),
        ("CG", "OD1", 1), ("CG", "OD2", 2),
    ],
    "CYS": _BB + [("CA", "CB", 1), ("CB", "SG", 1)],
    "GLN": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1), ("CG", "CD", 1),
        ("CD", "OE1", 2), ("CD", "NE2", 1),
    ],
    "GLU": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1), ("CG", "CD", 1),
        ("CD", "OE1", 1), ("CD", "OE2", 2),
    ],
    "GLY": _BB,
    "HIS": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1),
        ("CG", "ND1", 4), ("CG", "CD2", 4),
        ("ND1", "CE1", 4), ("CE1", "NE2", 4), ("NE2", "CD2", 4),
    ],
    "ILE": _BB + [
        ("CA", "CB", 1), ("CB", "CG1", 1), ("CB", "CG2", 1),
        ("CG1", "CD1", 1),
    ],
    "LEU": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1),
        ("CG", "CD1", 1), ("CG", "CD2", 1),
    ],
    "LYS": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1), ("CG", "CD", 1),
        ("CD", "CE", 1), ("CE", "NZ", 1),
    ],
    "MET": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1),
        ("CG", "SD", 1), ("SD", "CE", 1),
    ],
    "PHE": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1),
        ("CG", "CD1", 4), ("CG", "CD2", 4),
        ("CD1", "CE1", 4), ("CD2", "CE2", 4),
        ("CE1", "CZ", 4), ("CE2", "CZ", 4),
    ],
    "PRO": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1),
        ("CG", "CD", 1), ("CD", "N", 1),   # pyrrolidine ring closure
    ],
    "SER": _BB + [("CA", "CB", 1), ("CB", "OG", 1)],
    "THR": _BB + [("CA", "CB", 1), ("CB", "OG1", 1), ("CB", "CG2", 1)],
    "TRP": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1),
        # pyrrole ring
        ("CG", "CD1", 4), ("CD1", "NE1", 4), ("NE1", "CE2", 4),
        ("CE2", "CD2", 4), ("CD2", "CG", 4),
        # benzene ring fused at CD2-CE2
        ("CD2", "CE3", 4), ("CE3", "CZ3", 4), ("CZ3", "CH2", 4),
        ("CH2", "CZ2", 4), ("CZ2", "CE2", 4),
    ],
    "TYR": _BB + [
        ("CA", "CB", 1), ("CB", "CG", 1),
        ("CG", "CD1", 4), ("CG", "CD2", 4),
        ("CD1", "CE1", 4), ("CD2", "CE2", 4),
        ("CE1", "CZ", 4), ("CE2", "CZ", 4),
        ("CZ", "OH", 1),
    ],
    "VAL": _BB + [
        ("CA", "CB", 1), ("CB", "CG1", 1), ("CB", "CG2", 1),
    ],
    "UNK": _BB,  # fallback: backbone only
}


def build(
    residue_atom_maps: list[dict[str, int]],
    resnames: list[str],
    chain_ids: list[bytes],
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a bidirectional covalent bond graph for all polymer residues.

    Returns
    -------
    edge_index : int32 [2, N_edges]   (src, dst) pairs — both directions stored
    edge_type  : uint8 [N_edges]
    """
    srcs: list[int] = []
    dsts: list[int] = []
    types: list[int] = []

    def _add(u: int, v: int, t: int) -> None:
        srcs.append(u); dsts.append(v); types.append(t)
        srcs.append(v); dsts.append(u); types.append(t)

    n_res = len(resnames)

    # Intra-residue bonds (backbone + sidechain)
    for ri, (amap, resname) in enumerate(zip(residue_atom_maps, resnames)):
        bond_list = RESIDUE_BONDS.get(resname.upper(), RESIDUE_BONDS["UNK"])
        for a1, a2, btype in bond_list:
            if a1 in amap and a2 in amap:
                _add(amap[a1], amap[a2], btype)

    # Inter-residue peptide bonds: C(i) -> N(i+1) within the same chain
    for i in range(n_res - 1):
        if chain_ids[i] != chain_ids[i + 1]:
            continue
        c_idx = residue_atom_maps[i].get("C")
        n_idx = residue_atom_maps[i + 1].get("N")
        if c_idx is not None and n_idx is not None:
            _add(c_idx, n_idx, BOND_PEPTIDE)

    # Disulfide bonds: SG pairs within 2.5 Å
    sg_pairs = [
        (ri, amap["SG"])
        for ri, amap in enumerate(residue_atom_maps)
        if "SG" in amap
    ]
    for i in range(len(sg_pairs)):
        for j in range(i + 1, len(sg_pairs)):
            ri, si = sg_pairs[i]
            rj, sj = sg_pairs[j]
            if np.linalg.norm(positions[si] - positions[sj]) < 2.5:
                _add(si, sj, BOND_DISULFIDE)

    if not srcs:
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_type  = np.empty(0, dtype=np.uint8)
    else:
        edge_index = np.array([srcs, dsts], dtype=np.int32)
        edge_type  = np.array(types, dtype=np.uint8)

    return edge_index, edge_type
