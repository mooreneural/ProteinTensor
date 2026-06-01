from __future__ import annotations
import numpy as np
from pathlib import Path

from ..schema import ProteinTensorData, AA_VOCAB, AA_UNK


def from_mmcif(path: str | Path, pdb_id: str = "") -> ProteinTensorData:
    """Parse an mmCIF (or PDB) file into a ProteinTensorData.

    Only polymer (amino acid) chains are included. Ligands, water, and
    alternative conformations are stripped. Only the first model is used.
    """
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError("gemmi is required: pip install gemmi") from exc

    path = Path(path)
    if not pdb_id:
        pdb_id = path.stem.upper().split("_")[0]  # e.g. "1abc" from "1abc_updated.cif"

    structure = gemmi.read_structure(str(path))
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()

    return _extract(structure, pdb_id)


def _info(info, *keys: str) -> str:
    for k in keys:
        try:
            return info[k]
        except (KeyError, Exception):
            pass
    return ""


def _extract(structure, pdb_id: str) -> ProteinTensorData:
    import gemmi

    seq_tokens: list[int]    = []
    res_indices: list[int]   = []
    chain_ids: list[bytes]   = []
    positions: list[list]    = []
    masks: list[bool]        = []
    bfactors: list[float]    = []
    atom_starts: list[int]   = []
    atom_counts: list[int]   = []
    cursor = 0

    resolution = float("nan")
    method = ""
    deposition_date = ""

    if structure.resolution:
        resolution = float(structure.resolution)

    info = structure.info
    method = _info(info, "_exptl.method", "_exptl_crystal.method")
    deposition_date = _info(info, "_pdbx_database_status.recvd_initial_deposition_date")

    model = structure[0]  # first model only
    for chain in model:
        polymer = chain.get_polymer()
        if polymer.check_polymer_type() not in (
            gemmi.PolymerType.PeptideL,
            gemmi.PolymerType.PeptideD,
        ):
            continue  # skip DNA, RNA, unknown

        chain_label = (chain.name[0] if chain.name else "A").encode()

        for residue in polymer:
            resname = residue.name.upper()
            token = AA_VOCAB.get(resname, AA_UNK)

            seq_tokens.append(token)
            res_indices.append(int(residue.seqid.num))
            chain_ids.append(chain_label)

            n = 0
            for atom in residue:
                pos = atom.pos
                positions.append([pos.x, pos.y, pos.z])
                masks.append(True)
                bfactors.append(float(atom.b_iso))
                n += 1

            atom_starts.append(cursor)
            atom_counts.append(n)
            cursor += n

    if not seq_tokens:
        raise ValueError(f"No polymer residues found in '{pdb_id}'")

    return ProteinTensorData(
        sequence_tokens=np.array(seq_tokens,  dtype=np.int32),
        residue_index=np.array(res_indices,   dtype=np.int32),
        chain_id=np.array(chain_ids,          dtype="S1"),
        atom_positions=np.array(positions,    dtype=np.float32).reshape(-1, 3),
        atom_mask=np.array(masks,             dtype=bool),
        b_factors=np.array(bfactors,          dtype=np.float32),
        residue_atom_start=np.array(atom_starts, dtype=np.int32),
        residue_atom_count=np.array(atom_counts, dtype=np.int32),
        pdb_id=pdb_id,
        resolution=resolution,
        method=method,
        deposition_date=deposition_date,
    )
