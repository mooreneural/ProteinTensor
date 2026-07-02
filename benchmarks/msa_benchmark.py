#!/usr/bin/env python
"""MSA caching benchmark for ProteinTensor.

Measures the *per-epoch* MSA load cost that a training run pays repeatedly:

    traditional : parse an A3M text file every epoch  (proteintensor.from_a3m)
    proteintensor: read the pre-tokenized MSA from a .ptt (proteintensor.read_msa)

Both produce identical token/profile arrays (verified per case), so this is a
fair like-for-like comparison of the recurring load, not the one-time cost.

IMPORTANT - what this does NOT measure
--------------------------------------
MSA *generation* (JackHMMER / MMseqs2 / ColabFold) is the large one-time cost the
README refers to. It requires those tools plus multi-GB sequence databases and is
not run here. No generation speedup is produced or claimed by this script - the
generation figure in the README is a literature-based projection, not a
measurement.

Usage
-----
    python benchmarks/msa_benchmark.py [--rounds N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# (n_res, n_seq) pairs chosen to match the depths in the README structure table.
CASES = [(76, 512), (300, 1024), (574, 2048), (1000, 4096)]
_AA = "ARNDCQEGHILKMFPSTWYV"


def _median_ms(fn, rounds: int) -> float:
    samples = []
    for _ in range(rounds):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(samples), 4)


def make_a3m(path: Path, n_seq: int, n_res: int, seed: int) -> None:
    """Write a realistic synthetic A3M: query + homologs with mutations, gaps,
    and occasional lowercase insertions. Content is synthetic; dimensions and
    file size are realistic, which is what the parse-vs-load timing depends on.
    """
    rng = np.random.default_rng(seed)
    aa = list(_AA)
    query = "".join(rng.choice(aa, n_res))
    lines = [">query", query]
    for i in range(n_seq - 1):
        row = list(query)
        r = rng.random(n_res)
        mut = r < 0.20
        gap = (r >= 0.20) & (r < 0.25)
        row_arr = np.array(row, dtype="<U1")
        row_arr[mut] = rng.choice(aa, mut.sum())
        row_arr[gap] = "-"
        seq = "".join(row_arr)
        if rng.random() < 0.1:  # ~10% of rows carry a short insertion
            pos = int(rng.integers(0, n_res))
            ins = "".join(rng.choice(aa, int(rng.integers(1, 4)))).lower()
            seq = seq[:pos] + ins + seq[pos:]
        lines.append(f">seq{i}")
        lines.append(seq)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_case(n_res: int, n_seq: int, rounds: int, pt) -> dict:
    from proteintensor.msa import from_a3m

    tmp = Path(tempfile.mkdtemp())
    a3m = tmp / "msa.a3m"
    make_a3m(a3m, n_seq, n_res, seed=n_res * n_seq)
    a3m_kb = round(a3m.stat().st_size / 1024, 1)

    # Traditional per-epoch path: parse A3M text -> tokens + profile.
    parsed = from_a3m(str(a3m), tool="synthetic", database="benchmark")

    # Build a sequence-only .ptt and cache the MSA into it.
    ptt = tmp / "case.ptt"
    pt.write(pt.from_sequence("A" * n_res), str(ptt))
    pt.add_msa(str(ptt), parsed, source="uniref90")
    ptt_kb = round(sum(f.stat().st_size for f in ptt.rglob("*") if f.is_file()) / 1024, 1)

    # Fidelity: the cached load must return identical tokens/profile.
    loaded = pt.read_msa(str(ptt), source="uniref90")
    fidelity = {
        "tokens": bool(np.array_equal(loaded.tokens, parsed.tokens)),
        "profile": bool(np.allclose(loaded.profile, parsed.profile)),
        "deletion_matrix": bool(np.array_equal(loaded.deletion_matrix, parsed.deletion_matrix)),
    }

    timings = {
        "a3m_parse": _median_ms(lambda: from_a3m(str(a3m)), rounds),
        "ptt_read_msa": _median_ms(lambda: pt.read_msa(str(ptt), source="uniref90"), rounds),
        "ptt_mmap_open": _median_ms(lambda: pt.mmap_msa_tokens(str(ptt), "uniref90"), rounds),
    }
    speedup = round(timings["a3m_parse"] / timings["ptt_read_msa"], 2)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    return {
        "n_res": n_res,
        "n_seq": n_seq,
        "a3m_kb": a3m_kb,
        "ptt_msa_kb": ptt_kb,
        "timings_ms": timings,
        "load_speedup_vs_a3m": speedup,
        "roundtrip_lossless": fidelity,
    }


def collect_env(pt) -> dict:
    import platform, sys, zarr
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "proteintensor": getattr(pt, "__version__", "unknown"),
        "measures": "per-epoch MSA load: A3M parse vs .ptt read. Generation NOT measured.",
    }


def write_markdown(record: dict) -> None:
    env = record["env"]
    lines = [
        "# MSA caching benchmark",
        "",
        "Measures the **per-epoch MSA load** cost: parsing an A3M text file every",
        "epoch vs reading the pre-tokenized MSA from a `.ptt`. Both yield identical",
        "arrays (verified). Generated by `python benchmarks/msa_benchmark.py`.",
        "",
        "> **Not measured here:** MSA *generation* (JackHMMER / MMseqs2 / ColabFold).",
        "> That is the large one-time cost cited in the README as a literature-based",
        "> projection (~2.4 min/protein); it needs those tools + multi-GB databases",
        "> and is **not** run or benchmarked by this script. No generation speedup is",
        "> claimed here.",
        "",
        f"- **Platform:** {env['platform']}",
        f"- **Python:** {env['python']} | numpy {env['numpy']} | zarr {env['zarr']} | "
        f"proteintensor {env['proteintensor']}",
        f"- **Run:** {env['timestamp_utc']}",
        "",
        "### Per-epoch MSA load (median ms, lower is better)",
        "",
        "| Residues | MSA depth | A3M KB | A3M parse | .ptt read_msa | mmap open | "
        "load speedup | lossless |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in record["cases"]:
        t = c["timings_ms"]
        lossless = "yes" if all(c["roundtrip_lossless"].values()) else "NO"
        lines.append(
            f"| {c['n_res']} | {c['n_seq']} | {c['a3m_kb']} | {t['a3m_parse']} | "
            f"{t['ptt_read_msa']} | {t['ptt_mmap_open']} | "
            f"**{c['load_speedup_vs_a3m']}x** | {lossless} |"
        )
    lines.append("")
    (Path(__file__).resolve().parent / "MSA_RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="ProteinTensor MSA caching benchmark")
    ap.add_argument("--rounds", type=int, default=7, help="timing rounds per measurement")
    args = ap.parse_args()

    import proteintensor as pt

    print(f"MSA caching benchmark, {args.rounds} rounds/measurement ...")
    cases = []
    for n_res, n_seq in CASES:
        print(f"  - {n_res} res x {n_seq} seqs")
        cases.append(benchmark_case(n_res, n_seq, args.rounds, pt))

    record = {"env": collect_env(pt), "cases": cases}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = record["env"]["timestamp_utc"].replace(":", "").replace("-", "").replace("+0000", "Z")
    out = RESULTS_DIR / f"msa_{stamp}.json"
    out.write_text(json.dumps(record, indent=2))
    write_markdown(record)
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    print(f"Wrote {(Path(__file__).resolve().parent / 'MSA_RESULTS.md').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
