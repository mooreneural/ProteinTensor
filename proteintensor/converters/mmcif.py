from __future__ import annotations
import numpy as np
from pathlib import Path

from ..schema import (
    ProteinTensorData, AA_VOCAB, AA_UNK, BACKBONE_ATOMS, N_BACKBONE,
    NUC_VOCAB, NUC_UNK, MOL_PROTEIN, MOL_DNA, MOL_RNA, DNA_RESIDUES,
)
from ..bonds import build as build_bonds


def from_mmcif(
    path: str | Path,
    pdb_id: str = "",
    *,
    include_ligands: bool = False,
) -> ProteinTensorData:
    """Parse an mmCIF (or PDB) file into a ProteinTensorData.

    Only polymer (amino acid) chains are included in the structure tensors.
    Water and alternative conformations are stripped; only the first model is
    used. With ``include_ligands=True``, non-polymer non-water residues (drugs,
    cofactors, ions) are additionally captured on ``data.ligands``.
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

    data = _extract(structure, pdb_id)
    if include_ligands:
        from ..ligands import extract_from_gemmi
        data.ligands = extract_from_gemmi(structure)
    return data


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
    resnames: list[str]      = []
    mol_types: list[int]     = []
    positions: list[list]    = []
    masks: list[bool]        = []
    bfactors: list[float]    = []
    atom_starts: list[int]   = []
    atom_counts: list[int]   = []
    bb_pos_list: list        = []
    bb_mask_list: list       = []
    res_atom_maps: list[dict[str, int]] = []   # per-residue {atom_name: global_idx}
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
        ptype = polymer.check_polymer_type()
        if ptype in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            chain_kind = MOL_PROTEIN
        elif ptype in (gemmi.PolymerType.Dna, gemmi.PolymerType.Rna,
                       gemmi.PolymerType.DnaRnaHybrid):
            chain_kind = MOL_DNA   # refined per-residue below
        else:
            continue  # skip saccharides / unknown polymers

        chain_label = (chain.name[0] if chain.name else "A").encode()

        for residue in polymer:
            resname = residue.name.upper()
            if chain_kind == MOL_PROTEIN:
                token, mtype = AA_VOCAB.get(resname, AA_UNK), MOL_PROTEIN
            else:
                token = NUC_VOCAB.get(resname, NUC_UNK)
                mtype = MOL_DNA if resname in DNA_RESIDUES else MOL_RNA

            seq_tokens.append(token)
            res_indices.append(int(residue.seqid.num))
            chain_ids.append(chain_label)
            resnames.append(resname)
            mol_types.append(mtype)

            # All-atom ragged storage + atom-name -> global-index map
            atom_name_map: dict[str, int] = {}
            n = 0
            for atom in residue:
                pos = atom.pos
                atom_name_map[atom.name] = cursor + n
                positions.append([pos.x, pos.y, pos.z])
                masks.append(True)
                bfactors.append(float(atom.b_iso))
                n += 1
            res_atom_maps.append(atom_name_map)

            atom_starts.append(cursor)
            atom_counts.append(n)
            cursor += n

            # Backbone dense storage: N=0, CA=1, C=2, O=3
            atom_map = {a.name: a for a in residue}
            bb_pos  = np.zeros((N_BACKBONE, 3), dtype=np.float32)
            bb_mask = np.zeros(N_BACKBONE, dtype=bool)
            for bb_idx, bb_name in enumerate(BACKBONE_ATOMS):
                atom = atom_map.get(bb_name)
                if atom is not None:
                    p = atom.pos
                    bb_pos[bb_idx] = [p.x, p.y, p.z]
                    bb_mask[bb_idx] = True
            bb_pos_list.append(bb_pos)
            bb_mask_list.append(bb_mask)

    if not seq_tokens:
        raise ValueError(f"No polymer residues found in '{pdb_id}'")

    pos_arr = np.array(positions, dtype=np.float32).reshape(-1, 3)
    edge_index, edge_type = build_bonds(res_atom_maps, resnames, chain_ids, pos_arr)

    has_nucleic = any(m != MOL_PROTEIN for m in mol_types)
    mol_type_arr = np.array(mol_types, dtype=np.uint8) if has_nucleic else None

    return ProteinTensorData(
        sequence_tokens=np.array(seq_tokens,  dtype=np.int32),
        residue_index=np.array(res_indices,   dtype=np.int32),
        chain_id=np.array(chain_ids,          dtype="S1"),
        molecule_type=mol_type_arr,
        atom_positions=pos_arr,
        atom_mask=np.array(masks,             dtype=bool),
        b_factors=np.array(bfactors,          dtype=np.float32),
        residue_atom_start=np.array(atom_starts, dtype=np.int32),
        residue_atom_count=np.array(atom_counts, dtype=np.int32),
        backbone_positions=np.stack(bb_pos_list).astype(np.float32),
        backbone_mask=np.stack(bb_mask_list),
        bond_edge_index=edge_index,
        bond_edge_type=edge_type,
        pdb_id=pdb_id,
        resolution=resolution,
        method=method,
        deposition_date=deposition_date,
    )
