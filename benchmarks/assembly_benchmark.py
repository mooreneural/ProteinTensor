#!/usr/bin/env python
"""Full feature-assembly benchmark - tests the README's headline speedup.

Reproduces the README methodology:

    Traditional  = mmCIF parse + read MSA from A3M file
    ProteinTensor= read the .ptt with all features pre-cached
                   (structure + MSA + distance matrix + ESM2 embedding)

For each structure it builds a fully-featured .ptt plus the matching A3M file,
then times both assembly paths and reports the per-structure and average
speedup so the "34x" claim can be checked on the current code / machine.

What is real vs synthetic
-------------------------
- Structure + coordinates: real, parsed from the local mmCIF.
- MSA depth and embedding shape: realistic (match the README table), but the
  numeric *content* is synthetic - timing depends on tensor dimensions and file
  size, not on the biological values. No MSA generation or ESM2 inference is run.

Usage:  python benchmarks/assembly_benchmark.py [--rounds N]
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
EMB_MODEL = "esm2_t33_650M_UR50D"
EMB_DIM = 1280

# (pdb id, MSA depth) - depths match the README feature-assembly table.
CASES = [
    ("1UBQ", 512), ("6LU7", 1024), ("4HHB", 2048),
    ("6M0J", 2048), ("6VXX", 8192), ("6OHW", 8192),
]
_AA = np.array(list("ARNDCQEGHILKMFPSTWYV"))


def _median_ms(fn, rounds: int) -> float:
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(samples), 3)


def write_a3m(path: Path, n_seq: int, n_res: int, query: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(">query\n" + query + "\n")
        base = np.array(list(query))
        for i in range(n_seq - 1):
            r = rng.random(n_res)
            row = base.copy()
            mut = r < 0.20
            row[mut] = rng.choice(_AA, int(mut.sum()))
            row[(r >= 0.20) & (r < 0.25)] = "-"
            fh.write(f">s{i}\n" + "".join(row) + "\n")


def benchmark_case(pid: str, depth: int, rounds: int, pt, workdir: Path) -> dict | None:
    from proteintensor.converters.mmcif import from_mmcif
    from proteintensor.msa import from_a3m
    from proteintensor.pairs import compute_distance_matrix
    from proteintensor.schema import tokens_to_sequence

    cif = REPO_ROOT / f"{pid}.cif"
    if not cif.exists():
        print(f"  ! {pid}.cif not found; skipping")
        return None

    data = from_mmcif(cif)
    n_res = int(data.sequence_tokens.shape[0])
    ptt = workdir / f"{pid}.ptt"
    pt.write(data, str(ptt))

    # Build the query 1-letter sequence and a matching synthetic A3M.
    query = tokens_to_sequence(data.sequence_tokens)
    a3m = workdir / f"{pid}.a3m"
    write_a3m(a3m, depth, n_res, query, seed=hash(pid) & 0xFFFF)

    # Cache MSA + distance matrix + ESM2-shaped embedding into the .ptt.
    msa = from_a3m(str(a3m), tool="synthetic", database="benchmark")
    pt.add_msa(str(ptt), msa, source="uniref90")
    pt.compute_and_store_distances(str(ptt))
    emb = np.random.default_rng(0).standard_normal((n_res, EMB_DIM)).astype(np.float16)
    pt.add_embedding(str(ptt), emb, model=EMB_MODEL, dtype="float16",
                     sequence_hash=pt.embedding_sequence_hash(data.sequence_tokens))

    def traditional():
        d = from_mmcif(cif)
        _ = from_a3m(str(a3m))
        _ = compute_distance_matrix(d.backbone_positions)  # traditional recomputes it

    def proteintensor():
        pt.read(str(ptt))
        pt.read_msa(str(ptt), source="uniref90")
        pt.read_pair_feature(str(ptt), "distance_matrix")
        pt.read_embedding(str(ptt), EMB_MODEL)

    trad_ms = _median_ms(traditional, rounds)
    ptt_ms = _median_ms(proteintensor, rounds)
    return {
        "id": pid, "n_res": n_res, "msa_depth": depth,
        "traditional_ms": trad_ms, "proteintensor_ms": ptt_ms,
        "speedup": round(trad_ms / ptt_ms, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    import proteintensor as pt
    print(f"Full feature-assembly benchmark, {args.rounds} rounds/measurement ...")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for pid, depth in CASES:
            print(f"  - {pid} (depth {depth})")
            r = benchmark_case(pid, depth, args.rounds, pt, Path(tmp))
            if r:
                rows.append(r)

    if not rows:
        print("No structures available to benchmark.")
        return

    avg = round(sum(r["speedup"] for r in rows) / len(rows), 1)
    import platform, sys, zarr
    record = {
        "env": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "zarr": zarr.__version__,
            "proteintensor": getattr(pt, "__version__", "unknown"),
            "note": "Structure real; MSA/embedding shapes realistic, content synthetic. "
                    "No MSA generation or ESM2 inference measured.",
        },
        "cases": rows, "average_speedup": avg,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = record["env"]["timestamp_utc"].replace(":", "").replace("-", "").replace("+0000", "Z")
    (RESULTS_DIR / f"assembly_{stamp}.json").write_text(json.dumps(record, indent=2))

    lines = [
        "# Full feature-assembly benchmark", "",
        "Traditional (mmCIF parse + A3M parse + distance-matrix compute) vs",
        "ProteinTensor (read structure + MSA + distance matrix + embedding from `.ptt`).",
        "Tests the README's headline feature-assembly speedup on the current code.", "",
        f"- **Platform:** {record['env']['platform']}",
        f"- **proteintensor:** {record['env']['proteintensor']} | "
        f"numpy {record['env']['numpy']} | zarr {record['env']['zarr']}",
        f"- **Run:** {record['env']['timestamp_utc']}", "",
        "| Structure | Res | MSA depth | Traditional | ProteinTensor | Speedup |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['id']} | {r['n_res']} | {r['msa_depth']} | "
                     f"{r['traditional_ms']} ms | {r['proteintensor_ms']} ms | "
                     f"**{r['speedup']}x** |")
    lines += ["", f"**Average speedup across {len(rows)} structures: {avg}x**", "",
              "> Structure is real; MSA depth and embedding shape are realistic but the",
              "> numeric content is synthetic (timing depends on dimensions, not values).",
              "> MSA generation and ESM2 inference are one-time costs and are not measured.", ""]
    (Path(__file__).resolve().parent / "ASSEMBLY_RESULTS.md").write_text("\n".join(lines) + "\n")

    print(f"\nAverage feature-assembly speedup: {avg}x across {len(rows)} structures")
    for r in rows:
        print(f"  {r['id']:6s} {r['n_res']:5d} res  trad {r['traditional_ms']:9.1f} ms  "
              f"ptt {r['proteintensor_ms']:7.1f} ms  ->  {r['speedup']}x")


if __name__ == "__main__":
    main()
