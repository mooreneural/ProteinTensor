"""Shared helpers for model input adapters.

Adapters convert a `.ptt` file into a model's native input format. These helpers
turn stored sequence tokens, chain labels, MSA tensors, and ligand SMILES into
the per-chain sequences and A3M text every adapter needs.
"""
from __future__ import annotations

from pathlib import Path

import zarr

# Token int -> 1-letter code (matches proteintensor.schema AA_VOCAB order).
INT_TO_1LETTER: dict[int, str] = {
    0: "A", 1: "R", 2: "N", 3: "D", 4: "C",
    5: "Q", 6: "E", 7: "G", 8: "H", 9: "I",
    10: "L", 11: "K", 12: "M", 13: "F", 14: "P",
    15: "S", 16: "T", 17: "W", 18: "Y", 19: "V",
    20: "X",   # UNK
    21: "-",   # MSA gap
}
MSA_GAP = 21


def open_ptt(path: str | Path) -> zarr.Group:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return zarr.open(str(path), mode="r")


def build_chain_seqs(tokens, chain_raw) -> dict[str, str]:
    """Return {chain_id: amino_acid_sequence} preserving chain order."""
    chains: dict[str, list[str]] = {}
    for tok, cid_b in zip(tokens, chain_raw):
        cid = cid_b.decode() if isinstance(cid_b, bytes) else str(cid_b)
        chains.setdefault(cid, []).append(INT_TO_1LETTER.get(int(tok), "X"))
    return {cid: "".join(seq) for cid, seq in chains.items()}


def build_chain_token_lists(tokens, chain_raw) -> dict[str, list[int]]:
    chains: dict[str, list[int]] = {}
    for tok, cid_b in zip(tokens, chain_raw):
        cid = cid_b.decode() if isinstance(cid_b, bytes) else str(cid_b)
        chains.setdefault(cid, []).append(int(tok))
    return chains


def _a3m_text(msa_tokens, query_seq: str) -> str:
    """Render an int32 [N_seq, N_res] MSA slice as A3M text (row 0 = query)."""
    lines = [">query", query_seq]
    for i in range(1, len(msa_tokens)):
        row = msa_tokens[i]
        seq = "".join(
            INT_TO_1LETTER.get(int(t), "X") if int(t) != MSA_GAP else "-"
            for t in row
        )
        if seq.replace("-", ""):   # skip all-gap rows
            lines.extend([f">seq{i}", seq])
    return "\n".join(lines) + "\n"


def single_seq_a3m(seq: str) -> str:
    """A single-sequence A3M (query only) - used where a tool needs an alignment
    file but no MSA is cached."""
    return f">query\n{seq}\n"


def chain_a3m_texts(
    store: zarr.Group,
    chain_seqs: dict[str, str],
    chain_tok_lists: dict[str, list[int]],
    *,
    msa_source: str = "default",
    max_seqs: int = 4096,
) -> dict[str, str]:
    """Return {chain_id: a3m_text} for chains that have a cached MSA.

    Returns an empty dict when no MSA source is cached, so each adapter can decide
    the no-MSA behavior (omit and let the model generate one, or fall back to a
    single-sequence A3M where the model requires precomputed alignments).
    """
    has_msa = "msa" in store and msa_source in list(store["msa"].keys())
    if not has_msa:
        return {}

    grp = store[f"msa/{msa_source}"]
    n_cap = min(max_seqs, int(grp["tokens"].shape[0]))
    msa_tokens = grp["tokens"][:n_cap]

    out: dict[str, str] = {}
    col = 0
    for cid, ctoks in chain_tok_lists.items():
        n = len(ctoks)
        out[cid] = _a3m_text(msa_tokens[:, col:col + n], chain_seqs[cid])
        col += n
    return out


def ligand_smiles(store: zarr.Group) -> list[tuple[str, str]]:
    """Return [(name, smiles)] for stored ligands that carry a SMILES string."""
    if "ligands" not in store:
        return []
    out: list[tuple[str, str]] = []
    for key in store["ligands"].keys():
        attrs = dict(store[f"ligands/{key}"].attrs)
        smi = attrs.get("smiles")
        if smi:
            out.append((attrs.get("name", key), smi))
    return out


def load_ptt(path: str | Path, msa_source: str, max_seqs: int):
    """Common load: return (pdb_id, chain_seqs, a3m_texts, ligands)."""
    store = open_ptt(path)
    tokens = store["sequence/tokens"][:]
    chain_raw = store["sequence/chain_id"][:]
    pdb_id = store.attrs.get("pdb_id", Path(path).stem) or Path(path).stem
    chain_seqs = build_chain_seqs(tokens, chain_raw)
    chain_toks = build_chain_token_lists(tokens, chain_raw)
    a3m = chain_a3m_texts(store, chain_seqs, chain_toks,
                          msa_source=msa_source, max_seqs=max_seqs)
    return pdb_id, chain_seqs, a3m, ligand_smiles(store)
