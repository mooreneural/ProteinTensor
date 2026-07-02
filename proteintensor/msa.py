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

# Byte-indexed lookup tables for vectorized A3M parsing. Any byte not mapped in
# CHAR_TO_TOKEN resolves to AA_UNK. Lowercase letters and '.' are insertions.
_BYTE_TO_TOKEN = np.full(256, AA_UNK, dtype=np.int32)
for _ch, _tok in CHAR_TO_TOKEN.items():
    _BYTE_TO_TOKEN[ord(_ch)] = _tok

_BYTE_IS_INSERTION = np.zeros(256, dtype=bool)
_BYTE_IS_INSERTION[ord("a"):ord("z") + 1] = True
_BYTE_IS_INSERTION[ord(".")] = True


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

    parsed = [_parse_a3m_row(s) for s in raw_seqs]
    query_len = int(parsed[0][1].shape[0])   # aligned length of the query row

    N_seq = len(parsed)
    tokens = np.full((N_seq, query_len), MSA_GAP, dtype=np.int32)
    deletion_matrix = np.zeros((N_seq, query_len), dtype=np.float32)

    for i, (_aligned_bytes, tok, dels) in enumerate(parsed):
        n = min(tok.shape[0], query_len)
        tokens[i, :n]          = tok[:n]
        deletion_matrix[i, :n] = dels[:n]

    profile, deletion_mean = compute_profile(tokens)

    # Hash the query sequence (row 0, gaps stripped) for provenance
    q_bytes = parsed[0][0]
    query_seq = q_bytes[q_bytes != ord("-")].tobytes().decode("ascii", "replace")
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


def _parse_a3m_row(seq: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one A3M row into aligned tokens and per-column deletion counts.

    Lowercase letters and '.' are insertions - they increment the deletion
    counter for the next aligned column and are excluded from the output.
    Fully vectorized: no per-character Python loop.

    Returns
    -------
    aligned_bytes : uint8   [n_aligned]  raw aligned characters (uppercase / '-')
    tokens        : int32   [n_aligned]  MSA vocab token per aligned column
    deletions     : float32 [n_aligned]  insertions immediately before each column
    """
    b = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    is_ins = _BYTE_IS_INSERTION[b]
    aligned_pos = np.flatnonzero(~is_ins)
    aligned_bytes = b[aligned_pos]
    tokens = _BYTE_TO_TOKEN[aligned_bytes]
    # Deletions before column k = insertions between aligned column k-1 and k.
    csum = np.cumsum(is_ins)
    deletions = np.diff(csum[aligned_pos], prepend=0).astype(np.float32)
    return aligned_bytes, tokens, deletions
