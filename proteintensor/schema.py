from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

FORMAT_VERSION = "0.7"

AA_VOCAB: dict[str, int] = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
    "UNK": 20,
}
AA_UNK = 20
AA_VOCAB_SIZE = 21

# Single-letter codes indexed by token: position i is the 1-letter code for token i.
# Tokens 0-19 are the standard amino acids in AA_VOCAB order; token 20 (UNK) -> "X".
AA_1LETTER = "ARNDCQEGHILKMFPSTWYVX"

# Inverse map for sequence input. Any character absent here resolves to AA_UNK,
# which also covers ambiguity codes (B, Z, J, O) and gaps (-, .).
ONE_LETTER_TO_TOKEN: dict[str, int] = {c: i for i, c in enumerate(AA_1LETTER)}


def sequence_to_tokens(sequence: str) -> np.ndarray:
    """Map a 1-letter amino-acid string to an int32 token array (unknown -> UNK)."""
    cleaned = "".join(sequence.split()).upper()
    return np.array(
        [ONE_LETTER_TO_TOKEN.get(c, AA_UNK) for c in cleaned], dtype=np.int32
    )


def tokens_to_sequence(tokens: np.ndarray) -> str:
    """Map an int32 token array back to a 1-letter amino-acid string."""
    return "".join(AA_1LETTER[int(t)] if 0 <= int(t) < AA_VOCAB_SIZE else "X" for t in tokens)

# Canonical backbone atom order (AlphaFold / OpenFold convention)
BACKBONE_ATOMS = ["N", "CA", "C", "O"]
N_BACKBONE = len(BACKBONE_ATOMS)


@dataclass
class BondData:
    """Covalent bond graph returned by read_bonds()."""
    edge_index: np.ndarray   # int32 [2, N_edges]  (src, dst) - bidirectional
    edge_type:  np.ndarray   # uint8 [N_edges]
    num_atoms:  int


@dataclass
class BackboneData:
    """Lightweight view returned by read_backbone() - sequence + backbone only."""
    positions: np.ndarray        # float32 [N_res, 4, 3]  N / CA / C / O
    mask: np.ndarray             # bool    [N_res, 4]
    sequence_tokens: np.ndarray  # int32   [N_res]
    residue_index: np.ndarray    # int32   [N_res]
    chain_id: np.ndarray         # S1      [N_res]


@dataclass
class LigandData:
    """A small-molecule / non-polymer ligand (drug, cofactor, ion, etc.).

    The CCD code (``name``) is the reliable chemical identity - downstream tools
    such as Boltz resolve the canonical bond graph from it. Coordinates are the
    observed pose from the source structure. ``smiles`` is populated only when
    explicitly provided (e.g. via from_smiles); it is never inferred from
    coordinates, which would be error-prone.
    """
    name: str                 # CCD / residue code, e.g. "STI", "JZ4", "GDP"
    elements: np.ndarray      # S2      [N_atoms]  element symbols ("C", "N", "Mg")
    positions: np.ndarray     # float32 [N_atoms, 3]  Angstrom coordinates
    b_factors: np.ndarray     # float32 [N_atoms]
    chain_id: str = ""        # source chain label
    res_num: int = 0          # source residue number
    smiles: str = ""          # canonical SMILES if known (never inferred from coords)

    @property
    def num_atoms(self) -> int:
        return int(self.positions.shape[0])


@dataclass
class ProteinTensorData:
    # Sequence-level - shape [N_res]
    sequence_tokens: np.ndarray      # int32   residue vocab indices
    residue_index: np.ndarray        # int32   PDB sequence numbers
    chain_id: np.ndarray             # S1      single-char chain labels

    # Atom-level - shapes [N_atoms] or [N_atoms, 3].
    # None for sequence-only entries (from_sequence / from_fasta) that carry no structure.
    atom_positions: np.ndarray | None = None   # float32 [N_atoms, 3]  Angstroms
    atom_mask: np.ndarray | None = None        # bool    [N_atoms]
    b_factors: np.ndarray | None = None        # float32 [N_atoms]     B-factor / pLDDT

    # Residue->atom mapping - shape [N_res] (None for sequence-only entries)
    residue_atom_start: np.ndarray | None = None   # int32   first atom index for each residue
    residue_atom_count: np.ndarray | None = None   # int32   number of atoms per residue

    # Backbone dense layout - shapes [N_res, 4, 3] and [N_res, 4]
    # Atom order: N=0, CA=1, C=2, O=3  (missing atoms have mask=False, coords=0)
    backbone_positions: np.ndarray | None = None  # float32 [N_res, 4, 3]
    backbone_mask: np.ndarray | None = None       # bool    [N_res, 4]

    # Covalent bond graph - bidirectional edges referencing atom_positions indices
    bond_edge_index: np.ndarray | None = None  # int32 [2, N_edges]
    bond_edge_type:  np.ndarray | None = None  # uint8 [N_edges]

    # Small-molecule / non-polymer ligands (drugs, cofactors, ions). Empty by
    # default; populated by from_mmcif(include_ligands=True) or add_ligand().
    ligands: list["LigandData"] = field(default_factory=list)

    # Structure metadata
    pdb_id: str = ""
    resolution: float = float("nan")
    method: str = ""
    deposition_date: str = ""

    @property
    def has_structure(self) -> bool:
        """True if 3D coordinates are present; False for sequence-only entries."""
        return self.atom_positions is not None

    @property
    def num_residues(self) -> int:
        return int(self.sequence_tokens.shape[0])
