from __future__ import annotations
import numpy as np
from pathlib import Path

from ..schema import ProteinTensorData, sequence_to_tokens


def from_sequence(
    sequence: str,
    *,
    pdb_id: str = "",
    chain_id: str = "A",
    residue_start: int = 1,
) -> ProteinTensorData:
    """Build a sequence-only ProteinTensorData from a 1-letter amino-acid string.

    No 3D coordinates are produced - the result has ``has_structure == False`` and
    is the primary input form for sequence-driven predictors such as AlphaFold and
    Boltz. Unknown / ambiguity characters (B, Z, J, O, X, gaps) map to UNK.

    Parameters
    ----------
    sequence       1-letter amino-acid string. Whitespace is ignored.
    pdb_id         Identifier stored in metadata (e.g. a UniProt accession).
    chain_id       Single-character chain label applied to every residue.
    residue_start  PDB residue number assigned to the first residue (default 1).
    """
    tokens = sequence_to_tokens(sequence)
    if tokens.shape[0] == 0:
        raise ValueError("Empty sequence: no amino-acid residues to encode.")

    n = tokens.shape[0]
    chain_label = (chain_id[0] if chain_id else "A").encode()
    return ProteinTensorData(
        sequence_tokens=tokens,
        residue_index=np.arange(residue_start, residue_start + n, dtype=np.int32),
        chain_id=np.full(n, chain_label, dtype="S1"),
        pdb_id=pdb_id,
        method="sequence",
    )


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into a list of (header, sequence) tuples."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def from_fasta(path: str | Path, *, pdb_id: str = "") -> ProteinTensorData:
    """Build a ProteinTensorData from a FASTA file.

    A single record produces a single-chain sequence-only entry. Multiple records
    are treated as a multi-chain complex: chains are concatenated and labelled
    A, B, C, ... in file order, with residue numbering restarting per chain. This
    matches the multi-chain input that AlphaFold-Multimer and Boltz consume.
    """
    path = Path(path)
    records = parse_fasta(path.read_text())
    if not records:
        raise ValueError(f"No FASTA records found in '{path}'.")
    if not pdb_id:
        pdb_id = path.stem.upper().split("_")[0]

    if len(records) == 1:
        return from_sequence(records[0][1], pdb_id=pdb_id, chain_id="A")

    all_tokens: list[np.ndarray] = []
    all_res_idx: list[np.ndarray] = []
    all_chain: list[np.ndarray] = []
    for i, (_, seq) in enumerate(records):
        sub = from_sequence(seq, chain_id=_chain_label(i))
        all_tokens.append(sub.sequence_tokens)
        all_res_idx.append(sub.residue_index)
        all_chain.append(sub.chain_id)

    return ProteinTensorData(
        sequence_tokens=np.concatenate(all_tokens),
        residue_index=np.concatenate(all_res_idx),
        chain_id=np.concatenate(all_chain),
        pdb_id=pdb_id,
        method="sequence",
    )


def _chain_label(index: int) -> str:
    """Map a 0-based chain index to a label: 0->A .. 25->Z, then a, b, ..."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return alphabet[index] if index < len(alphabet) else "X"
