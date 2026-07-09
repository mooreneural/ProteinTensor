"""OpenFold input adapter.

Converts a `.ptt` into OpenFold's inference inputs: a FASTA of the chain
sequences plus a per-chain alignment directory containing an A3M MSA
(`<out>/alignments/<tag>/uniref90_hits.a3m`), which is the precomputed-MSA layout
OpenFold's inference reads.

This adapter generates input only - OpenFold is not bundled. Point OpenFold's
inference at the FASTA and alignment directory separately.
"""
from __future__ import annotations

from pathlib import Path

from ._common import load_ptt, single_seq_a3m


class OpenFoldAdapter:
    """Convert a `.ptt` file into OpenFold FASTA + alignment inputs.

    Examples
    --------
    adapter = OpenFoldAdapter("1abc.ptt")
    fasta = adapter.write_input("openfold_input/")
    # -> openfold_input/1abc.fasta
    # -> openfold_input/alignments/1abc_A/uniref90_hits.a3m
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
        """Write an OpenFold FASTA + per-chain alignment dirs; return the FASTA path."""
        pdb_id, chain_seqs, a3m, _ligands = load_ptt(self.path, msa_source, max_msa_seqs)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        align_root = out / "alignments"

        lines: list[str] = []
        for cid, seq in chain_seqs.items():
            tag = f"{pdb_id}_{cid}"
            lines.append(f">{tag}")
            lines.append(seq)
            if write_msa:
                # OpenFold reads precomputed alignments; fall back to a
                # single-sequence A3M when no MSA is cached.
                adir = align_root / tag
                adir.mkdir(parents=True, exist_ok=True)
                text = a3m.get(cid) or single_seq_a3m(seq)
                (adir / "uniref90_hits.a3m").write_text(text, encoding="utf-8")

        fasta = out / f"{pdb_id}.fasta"
        fasta.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return fasta

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "OpenFold is not bundled with ProteinTensor. Use write_input() to produce "
            "the FASTA and alignment directory, then run OpenFold's inference on them."
        )
