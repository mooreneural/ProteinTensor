"""Nesso-1 input adapter.

Nesso-1 (Recursion) is a coarse-grained cofolding model that predicts binding
affinity. Its input is a YAML file listing protein chains (amino-acid sequences)
and ligands (SMILES), plus `affinity` property targets:

    sequences:
      - protein:
          id: A
          sequence: MKTAYIAKQ...
      - ligand:
          id: B
          smiles: "Fc1ccc(cc1)..."
    properties:
      - affinity:
          binder: B

A `.ptt` already holds both halves - sequence tokens and pocket-centric ligand
SMILES - so this adapter maps directly onto Nesso's format.

This adapter generates input only; Nesso is not bundled. Run
`nesso predict <file>.yaml --out_dir <dir>` on the produced YAML separately.
"""
from __future__ import annotations

from pathlib import Path

from ._common import load_ptt


def _ligand_ids(used: set[str], n: int) -> list[str]:
    pool = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            if c not in used]
    return pool[:n]


class NessoAdapter:
    """Convert a `.ptt` file into a Nesso-1 prediction YAML.

    Examples
    --------
    adapter = NessoAdapter("6oim.ptt")
    adapter.write_input("nesso_input/6oim.yaml")   # protein chains + ligands + affinity
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
        max_msa_seqs: int = 4096,
        affinity: bool = True,
    ) -> Path:
        """Write a Nesso prediction YAML and return its path.

        Each protein chain becomes a `protein` entry; each ligand with a SMILES
        string becomes a `ligand` entry. When ``affinity`` is True, an
        `affinity` property target is added per ligand.
        """
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to write Nesso input: pip install pyyaml"
            ) from exc

        _pdb_id, chain_seqs, _a3m, ligands = load_ptt(self.path, msa_source, max_msa_seqs)

        sequences: list[dict] = [
            {"protein": {"id": cid, "sequence": seq}}
            for cid, seq in chain_seqs.items()
        ]

        properties: list[dict] = []
        for lid, (_name, smi) in zip(_ligand_ids(set(chain_seqs), len(ligands)), ligands):
            sequences.append({"ligand": {"id": lid, "smiles": smi}})
            if affinity:
                properties.append({"affinity": {"binder": lid}})

        data: dict = {"sequences": sequences}
        if properties:
            data["properties"] = properties

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
        return out

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "Nesso is not bundled with ProteinTensor. Use write_input() to produce "
            "the YAML, then run: nesso predict <file>.yaml --out_dir <dir>"
        )
