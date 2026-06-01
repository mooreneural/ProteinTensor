"""
Boltz adapter for ProteinTensor.

Converts .ptt data into Boltz-compatible input files (YAML + A3M) and
optionally invokes boltz.main.predict() directly.

Boltz input format (v2)
-----------------------
sequences:
  - protein:
      id: A
      sequence: MADQLTEEQIAEFKEAFSLF
      msa: ./msa/A.a3m         # optional, omit to let Boltz run MMseqs2
  - protein:
      id: B
      sequence: AKLSILPWGHC
version: 1

Output layout written by write_input()
---------------------------------------
<output_dir>/
├── <pdb_id>.yaml
└── msa/
    ├── A.a3m    (chain A MSA, if MSA data present in .ptt)
    └── B.a3m    (chain B MSA, sliced from full-sequence MSA)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Token int -> single-letter code (matches proteintensor.schema.AA_VOCAB order)
_INT_TO_1LETTER: dict[int, str] = {
    0: "A", 1: "R", 2: "N", 3: "D", 4: "C",
    5: "Q", 6: "E", 7: "G", 8: "H", 9: "I",
    10: "L", 11: "K", 12: "M", 13: "F", 14: "P",
    15: "S", 16: "T", 17: "W", 18: "Y", 19: "V",
    20: "X",   # UNK
    21: "-",   # MSA_GAP
}


class BoltzAdapter:
    """Convert a .ptt file to Boltz input and optionally run prediction.

    Parameters
    ----------
    ptt_path : path to a ProteinTensor .ptt Zarr store

    Examples
    --------
    # Write input files only (no Boltz installed required)
    adapter = BoltzAdapter("1UBQ.ptt")
    yaml_path = adapter.write_input("boltz_run/input")

    # Full end-to-end prediction
    adapter.predict("boltz_run", model="boltz2", diffusion_samples=5)
    """

    def __init__(self, ptt_path: str | Path) -> None:
        self.path = Path(ptt_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chains(self) -> dict[str, str]:
        """Return {chain_id: amino_acid_sequence} for every chain in the .ptt."""
        import zarr
        store = zarr.open(str(self.path), mode="r")
        tokens   = store["sequence/tokens"][:]
        chain_raw = store["sequence/chain_id"][:]
        return _build_chain_seqs(tokens, chain_raw)

    def write_input(
        self,
        output_dir: str | Path,
        *,
        msa_source: str = "default",
        max_msa_seqs: int = 4096,
    ) -> Path:
        """Write a Boltz YAML + per-chain A3M files and return the YAML path.

        Parameters
        ----------
        output_dir    Directory to write into (created if absent).
        msa_source    Which MSA source from the .ptt to use. Ignored if none present.
        max_msa_seqs  Cap on MSA depth written to A3M (Boltz default cap is 8192).

        Returns
        -------
        Path to the written YAML file, ready to pass to boltz.main.predict().
        """
        import yaml, zarr
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        store      = zarr.open(str(self.path), mode="r")
        tokens     = store["sequence/tokens"][:]
        chain_raw  = store["sequence/chain_id"][:]
        pdb_id     = store.attrs.get("pdb_id", self.path.stem) or self.path.stem

        chain_seqs      = _build_chain_seqs(tokens, chain_raw)
        chain_tok_lists = _build_chain_token_lists(tokens, chain_raw)

        # Always write per-chain A3M files so Boltz never needs --use_msa_server.
        # If MSA data is cached in the .ptt, use it; otherwise write a
        # single-sequence A3M (query only), which is valid A3M and lets
        # Boltz run in single-sequence mode without fetching an MSA.
        msa_dir = output_dir / "msa"
        msa_dir.mkdir(exist_ok=True)

        has_msa = "msa" in store and msa_source in store.get("msa", zarr.open(store.store, mode="r"))
        a3m_paths: dict[str, str] = {}

        if has_msa:
            msa_grp    = store[f"msa/{msa_source}"]
            n_seq_cap  = min(max_msa_seqs, int(msa_grp["tokens"].shape[0]))
            msa_tokens = msa_grp["tokens"][:n_seq_cap]

            col = 0
            for cid, ctoks in chain_tok_lists.items():
                n = len(ctoks)
                chain_msa = msa_tokens[:, col : col + n]
                a3m_path  = msa_dir / f"{cid}.a3m"
                _write_a3m(chain_msa, chain_seqs[cid], a3m_path)
                a3m_paths[cid] = str(a3m_path.resolve())
                col += n
        else:
            # Single-sequence A3M — query only, no homologs
            for cid, seq in chain_seqs.items():
                a3m_path = msa_dir / f"{cid}.a3m"
                a3m_path.write_text(f">query\n{seq}\n", encoding="utf-8")
                a3m_paths[cid] = str(a3m_path.resolve())

        # Build YAML structure
        sequences = []
        for cid, seq in chain_seqs.items():
            entry: dict = {"id": cid, "sequence": seq}
            if cid in a3m_paths:
                entry["msa"] = a3m_paths[cid]
            sequences.append({"protein": entry})

        yaml_data = {"sequences": sequences, "version": 1}
        yaml_path = output_dir / f"{pdb_id}.yaml"
        with open(yaml_path, "w") as fh:
            yaml.dump(yaml_data, fh, default_flow_style=False, sort_keys=False)

        return yaml_path

    def predict(
        self,
        output_dir: str | Path,
        *,
        msa_source: str = "default",
        max_msa_seqs: int = 4096,
        model: str = "boltz2",
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        accelerator: str = "gpu",
        seed: Optional[int] = None,
        **boltz_kwargs,
    ) -> Path:
        """Write input files and run Boltz prediction end-to-end.

        Requires boltz to be installed (pip install boltz).

        Parameters
        ----------
        output_dir        Root directory; input files go under <output_dir>/input/,
                          Boltz writes predictions under <output_dir>/predictions/.
        msa_source        MSA source key in the .ptt file.
        max_msa_seqs      MSA depth cap passed to write_input().
        model             "boltz1" or "boltz2" (default).
        recycling_steps   Number of Evoformer recycling iterations.
        sampling_steps    Diffusion sampling steps.
        diffusion_samples Number of structure samples.
        accelerator       "gpu" or "cpu".
        seed              Random seed for reproducibility.
        **boltz_kwargs    Any extra kwargs forwarded to boltz.main.predict().

        Returns
        -------
        Path to the Boltz predictions directory.
        """
        try:
            from boltz.main import predict as _boltz_predict
        except ImportError as exc:
            raise ImportError(
                "boltz is required for prediction: pip install boltz"
            ) from exc

        output_dir   = Path(output_dir)
        input_dir    = output_dir / "input"
        predictions  = output_dir / "predictions"

        yaml_path = self.write_input(
            input_dir, msa_source=msa_source, max_msa_seqs=max_msa_seqs
        )

        # boltz.main.predict is a Click command; call .callback to bypass
        # Click's CLI argument parser and invoke the Python function directly.
        _boltz_predict.callback(
            data=str(yaml_path),
            out_dir=str(predictions),
            model=model,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            accelerator=accelerator,
            seed=seed,
            **boltz_kwargs,
        )

        return predictions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_chain_seqs(
    tokens: np.ndarray,
    chain_raw: np.ndarray,
) -> dict[str, str]:
    """Build {chain_id: sequence_string} preserving chain order."""
    chains: dict[str, list[str]] = {}
    for tok, cid_b in zip(tokens, chain_raw):
        cid = cid_b.decode() if isinstance(cid_b, bytes) else str(cid_b)
        chains.setdefault(cid, []).append(_INT_TO_1LETTER.get(int(tok), "X"))
    return {cid: "".join(seq) for cid, seq in chains.items()}


def _build_chain_token_lists(
    tokens: np.ndarray,
    chain_raw: np.ndarray,
) -> dict[str, list[int]]:
    chains: dict[str, list[int]] = {}
    for tok, cid_b in zip(tokens, chain_raw):
        cid = cid_b.decode() if isinstance(cid_b, bytes) else str(cid_b)
        chains.setdefault(cid, []).append(int(tok))
    return chains


def _write_a3m(msa_tokens: np.ndarray, query_seq: str, path: Path) -> None:
    """Write int32 [N_seq, N_res] MSA as A3M text.

    Row 0 is written as the query. Subsequent rows are aligned sequences
    with '-' for gaps; all-gap rows are omitted.
    """
    MSA_GAP = 21
    lines: list[str] = [">query", query_seq]
    for i in range(1, len(msa_tokens)):
        row = msa_tokens[i]
        seq = "".join(
            _INT_TO_1LETTER.get(int(t), "X") if int(t) != MSA_GAP else "-"
            for t in row
        )
        if seq.replace("-", ""):   # skip all-gap rows
            lines.extend([f">seq{i}", seq])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
