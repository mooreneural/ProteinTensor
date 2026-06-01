from __future__ import annotations
from dataclasses import dataclass
import numpy as np

FORMAT_VERSION = "0.2"

AA_VOCAB: dict[str, int] = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
    "UNK": 20,
}
AA_UNK = 20
AA_VOCAB_SIZE = 21

# Single-letter equivalents for display
AA_1LETTER = "ARNDCQEGHILKMFPSTWYXU"

# Canonical backbone atom order (AlphaFold / OpenFold convention)
BACKBONE_ATOMS = ["N", "CA", "C", "O"]
N_BACKBONE = len(BACKBONE_ATOMS)


@dataclass
class BackboneData:
    """Lightweight view returned by read_backbone() — sequence + backbone only."""
    positions: np.ndarray        # float32 [N_res, 4, 3]  N / CA / C / O
    mask: np.ndarray             # bool    [N_res, 4]
    sequence_tokens: np.ndarray  # int32   [N_res]
    residue_index: np.ndarray    # int32   [N_res]
    chain_id: np.ndarray         # S1      [N_res]


@dataclass
class ProteinTensorData:
    # Sequence-level — shape [N_res]
    sequence_tokens: np.ndarray      # int32   residue vocab indices
    residue_index: np.ndarray        # int32   PDB sequence numbers
    chain_id: np.ndarray             # S1      single-char chain labels

    # Atom-level — shapes [N_atoms] or [N_atoms, 3]
    atom_positions: np.ndarray       # float32 [N_atoms, 3]  Angstroms
    atom_mask: np.ndarray            # bool    [N_atoms]
    b_factors: np.ndarray            # float32 [N_atoms]     B-factor / pLDDT

    # Residue->atom mapping — shape [N_res]
    residue_atom_start: np.ndarray   # int32   first atom index for each residue
    residue_atom_count: np.ndarray   # int32   number of atoms per residue

    # Backbone dense layout — shapes [N_res, 4, 3] and [N_res, 4]
    # Atom order: N=0, CA=1, C=2, O=3  (missing atoms have mask=False, coords=0)
    backbone_positions: np.ndarray | None = None  # float32 [N_res, 4, 3]
    backbone_mask: np.ndarray | None = None       # bool    [N_res, 4]

    # Structure metadata
    pdb_id: str = ""
    resolution: float = float("nan")
    method: str = ""
    deposition_date: str = ""
