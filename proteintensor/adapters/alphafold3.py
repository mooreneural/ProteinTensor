"""AlphaFold 3 input adapter.

Converts a `.ptt` into an AlphaFold 3 input JSON (the `fold_input.json` format:
name / modelSeeds / sequences / dialect / version). Protein chains carry their
sequence and, when a cached MSA is present, an `unpairedMsa` A3M string so AF3 can
skip MSA generation. Ligands with a SMILES string are emitted as ligand entities.

This adapter generates input only - AlphaFold 3 itself is not bundled. Run the AF3
pipeline on the produced JSON separately.
"""
from __future__ import annotations

import json
from pathlib import Path

from ._common import load_ptt


def _ligand_ids(used: set[str], n: int) -> list[str]:
    pool = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            if c not in used]
    return pool[:n]


class AlphaFold3Adapter:
    """Convert a `.ptt` file into an AlphaFold 3 input JSON.

    Examples
    --------
    adapter = AlphaFold3Adapter("1abc.ptt")
    adapter.write_input("af3_input/1abc.json")     # -> AF3 fold_input JSON
    """

    def __init__(self, ptt_path: str | Path) -> None:
        self.path = Path(ptt_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def chains(self, msa_source: str = "default") -> dict[str, str]:
        return load_ptt(self.path, msa_source, 0)[1]

    def write_input(
        self,
        output_path: str | Path,
        *,
        msa_source: str = "default",
        model_seeds: tuple[int, ...] = (1,),
        version: int = 2,
        max_msa_seqs: int = 4096,
        include_msa: bool = True,
    ) -> Path:
        """Write an AlphaFold 3 input JSON and return its path."""
        pdb_id, chain_seqs, a3m, ligands = load_ptt(self.path, msa_source, max_msa_seqs)

        sequences: list[dict] = []
        for cid, seq in chain_seqs.items():
            entry: dict = {"id": cid, "sequence": seq}
            if include_msa and a3m.get(cid):
                entry["unpairedMsa"] = a3m[cid]
                entry["pairedMsa"] = ""
                entry["templates"] = []
            sequences.append({"protein": entry})

        for lid, (name, smi) in zip(
            _ligand_ids(set(chain_seqs), len(ligands)), ligands
        ):
            sequences.append({"ligand": {"id": lid, "smiles": smi}})

        data = {
            "name": pdb_id,
            "modelSeeds": list(model_seeds),
            "sequences": sequences,
            "dialect": "alphafold3",
            "version": version,
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "AlphaFold 3 is not bundled with ProteinTensor. Use write_input() to "
            "produce the input JSON, then run AlphaFold 3 on it "
            "(python run_alphafold.py --json_path=<file>)."
        )
