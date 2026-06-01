"""
MSA (Multiple Sequence Alignment) feature storage for ProteinTensor.

Design
------
MSAs are stored as sub-groups under msa/<source>/ inside an existing .ptt file,
so they can be added after initial conversion without touching structure data.
Multiple sources (uniref90, bfd, colabfold, etc.) coexist in the same file.

Provenance is tracked per-source so stale caches can be detected when the
sequence or database changes.

Token vocabulary (extends AA_VOCAB)
------------------------------------
0-19   Standard amino acids  (matches AA_VOCAB)
20     UNK / non-standard
21     GAP  (alignment gap '-')
22     MASK (reserved for masked MSA training objectives)
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schema import AA_VOCAB, AA_UNK

# MSA-specific token constants
MSA_GAP        = 21
MSA_MASK       = 22
MSA_VOCAB_SIZE = 23   # 0-22 inclusive

# Single-letter amino acid -> token (consistent with AA_VOCAB)
_1TO3: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

CHAR_TO_TOKEN: dict[str, int] = {
    letter: AA_VOCAB[three] for letter, three in _1TO3.items()
}
CHAR_TO_TOKEN.update({
    "-": MSA_GAP,
    "X": AA_UNK, "B": AA_UNK, "Z": AA_UNK,
    "J": AA_UNK, "U": AA_UNK, "O": AA_UNK,
})


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MsaData:
    """MSA features for one protein from one database search."""

    # Core alignment arrays
    tokens:          np.ndarray   # int32   [N_seq, N_res]  row 0 = query
    deletion_matrix: np.ndarray   # float32 [N_seq, N_res]  insertions before each column
    profile:         np.ndarray   # float32 [N_res, MSA_VOCAB_SIZE]  per-position frequencies
    deletion_mean:   np.ndarray   # float32 [N_res]  mean deletions per position

    # Provenance - required for cache invalidation
    sequence_hash: str   # SHA-256 of the query sequence (row 0, gaps stripped)
    tool:          str   # "jackhammer" | "hhblits" | "mmseqs2" | "colabfold"
    tool_version:  str
    database:      str   # "uniref90" | "bfd" | "mgnify" | "colabfold_db" | …
    database_date: str   # e.g. "2022-01"  (empty string if unknown)
    created_at:    float = field(default_factory=time.time)

    @property
    def num_sequences(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def num_residues(self) -> int:
        return int(self.tokens.shape[1])


# ---------------------------------------------------------------------------
# Profile computation
# ---------------------------------------------------------------------------

def compute_profile(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-position residue frequencies and mean deletion depth.

    Parameters
    ----------
    tokens : int32 [N_seq, N_res]

    Returns
    -------
    profile       : float32 [N_res, MSA_VOCAB_SIZE]
    deletion_mean : float32 [N_res]
    """
    N_seq, N_res = tokens.shape
    profile = np.zeros((N_res, MSA_VOCAB_SIZE), dtype=np.float32)
    for tok in range(MSA_VOCAB_SIZE):
        profile[:, tok] = (tokens == tok).mean(axis=0)
    deletion_mean = profile[:, MSA_GAP].copy()
    return profile, deletion_mean


# ---------------------------------------------------------------------------
# A3M parser
# ---------------------------------------------------------------------------

def from_a3m(
    path: str | Path,
    *,
    tool: str = "unknown",
    tool_version: str = "unknown",
    database: str = "unknown",
    database_date: str = "",
) -> MsaData:
    """Parse an A3M format MSA file into a MsaData object.

    A3M conventions
    ---------------
    - Uppercase letters / '-' : aligned columns (one per query position)
    - Lowercase letters / '.' : insertions in this sequence (not in the matrix)
    - Row 0 is treated as the query sequence.
    """
    path = Path(path)
    raw_seqs = _read_a3m_sequences(path.read_text(encoding="utf-8", errors="replace"))

    if not raw_seqs:
        raise ValueError(f"No sequences found in {path}")

    aligned_seqs, del_rows = zip(*(_parse_a3m_row(s) for s in raw_seqs))
    query_len = len(aligned_seqs[0])

    N_seq = len(aligned_seqs)
    tokens = np.full((N_seq, query_len), MSA_GAP, dtype=np.int32)
    deletion_matrix = np.zeros((N_seq, query_len), dtype=np.float32)

    for i, (seq, dels) in enumerate(zip(aligned_seqs, del_rows)):
        n = min(len(seq), query_len)
        for j in range(n):
            tokens[i, j] = CHAR_TO_TOKEN.get(seq[j], AA_UNK)
        deletion_matrix[i, :len(dels)] = dels[:query_len]

    profile, deletion_mean = compute_profile(tokens)

    # Hash the query sequence (gaps stripped) for provenance
    query_seq = "".join(c for c in aligned_seqs[0] if c != "-")
    seq_hash = hashlib.sha256(query_seq.encode()).hexdigest()

    return MsaData(
        tokens=tokens,
        deletion_matrix=deletion_matrix,
        profile=profile,
        deletion_mean=deletion_mean,
        sequence_hash=seq_hash,
        tool=tool,
        tool_version=tool_version,
        database=database,
        database_date=database_date,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_a3m_sequences(text: str) -> list[str]:
    """Extract raw sequence strings from A3M text, ignoring headers."""
    seqs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                seqs.append("".join(current))
                current = []
        else:
            current.append(line)
    if current:
        seqs.append("".join(current))
    return seqs


def _parse_a3m_row(seq: str) -> tuple[str, np.ndarray]:
    """Split one A3M row into aligned characters and per-column deletion counts.

    Lowercase letters and '.' are insertions - they increment the deletion
    counter for the next aligned column and are excluded from the output.
    """
    aligned: list[str] = []
    dels: list[int] = []
    insertions = 0
    for ch in seq:
        if ch.islower() or ch == ".":
            insertions += 1
        else:
            aligned.append(ch)
            dels.append(insertions)
            insertions = 0
    return "".join(aligned), np.array(dels, dtype=np.float32)
