"""Chai-1 input adapter.

Converts a `.ptt` into Chai-1's input FASTA (`>protein|name=...` for chains,
`>ligand|name=...` with a SMILES string for ligands) plus optional per-chain A3M
MSA files. Chai consumes MSAs as `aligned.pqt`; the A3M files written here can be
converted with Chai's own a3m-to-pqt tooling.

This adapter generates input only - Chai-1 is not bundled. Run `chai-lab fold`
on the produced FASTA separately.
"""
from __future__ import annotations

from pathlib import Path

from ._common import load_ptt


class ChaiAdapter:
    """Convert a `.ptt` file into Chai-1 input files.

    Examples
    --------
    adapter = ChaiAdapter("1abc.ptt")
    fasta = adapter.write_input("chai_input/")   # -> chai_input/1abc.fasta (+ msa/)
    """

    def __init__(self, ptt_path: str | Path) -> None:
        self.path = Path(ptt_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def chains(self, msa_source: str = "default") -> dict[str, str]:
        return load_ptt(self.path, msa_source, 0)[1]

    def write_input(
        self,
        output_dir: str | Path,
        *,
        msa_source: str = "default",
        max_msa_seqs: int = 4096,
        write_msa: bool = True,
    ) -> Path:
        """Write a Chai-1 FASTA (+ optional A3M MSA files); return the FASTA path."""
        pdb_id, chain_seqs, a3m, ligands = load_ptt(self.path, msa_source, max_msa_seqs)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for cid, seq in chain_seqs.items():
            lines.append(f">protein|name={pdb_id}_{cid}")
            lines.append(seq)
        for name, smi in ligands:
            lines.append(f">ligand|name={name}")
            lines.append(smi)

        fasta = out / f"{pdb_id}.fasta"
        fasta.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if write_msa:
            msa_dir = out / "msa"
            msa_dir.mkdir(exist_ok=True)
            for cid, text in a3m.items():
                (msa_dir / f"{pdb_id}_{cid}.a3m").write_text(text, encoding="utf-8")

        return fasta

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "Chai-1 is not bundled with ProteinTensor. Use write_input() to produce "
            "the FASTA (and A3M files), then run `chai-lab fold <fasta> <out_dir>`."
        )
