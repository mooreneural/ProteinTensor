from __future__ import annotations
import datetime
import sys
import tempfile
import time
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
SUPPORTED_INPUT = {".cif", ".mmcif", ".pdb", ".ent"}


@click.group()
@click.version_option(package_name="proteintensor")
def main():
    """ProteinTensor - AI-native biomolecular tensor format."""


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

@main.command()
@click.argument("input_path",  type=click.Path(exists=True,  path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--compression", default="blosc", show_default=True,
              type=click.Choice(["blosc", "none"]),
              help="Compression codec for the Zarr store.")
@click.option("--pdb-id", default="", help="Override the PDB ID stored in metadata.")
def convert(input_path: Path, output_path: Path, compression: str, pdb_id: str):
    """Convert an mmCIF or PDB file to ProteinTensor (.ptt) format."""
    from .converters.mmcif import from_mmcif
    from .writer import write

    if input_path.suffix.lower() not in SUPPORTED_INPUT:
        console.print(
            f"[yellow]Warning:[/yellow] unrecognized extension [bold]{input_path.suffix!r}[/bold], "
            "assuming mmCIF."
        )

    with console.status(f"Parsing [bold]{input_path.name}[/bold] ..."):
        t0 = time.perf_counter()
        data = from_mmcif(input_path, pdb_id=pdb_id)
        parse_ms = (time.perf_counter() - t0) * 1000

    with console.status(f"Writing [bold]{output_path}[/bold] ..."):
        t0 = time.perf_counter()
        write(data, output_path, compression=compression)
        write_ms = (time.perf_counter() - t0) * 1000

    src_bytes = input_path.stat().st_size
    dst_bytes = sum(f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file())

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_row("PDB ID",     data.pdb_id or "(unknown)")
    tbl.add_row("Chains",     _chain_summary(data.chain_id))
    tbl.add_row("Residues",   f"{data.sequence_tokens.shape[0]:,}")
    tbl.add_row("Atoms",      f"{data.atom_positions.shape[0]:,}")
    tbl.add_row("Resolution", f"{data.resolution:.2f} A" if data.resolution == data.resolution else "N/A")
    tbl.add_row("Method",     data.method or "N/A")
    tbl.add_row("")
    tbl.add_row("Parse time", f"{parse_ms:.1f} ms")
    tbl.add_row("Write time", f"{write_ms:.1f} ms")
    tbl.add_row("Source",     _fmt_bytes(src_bytes))
    tbl.add_row("Output",     _fmt_bytes(dst_bytes))
    tbl.add_row("Ratio",      f"{dst_bytes / src_bytes:.2f}x")

    console.print(Panel(tbl, title=f"[green]Converted -> {output_path}[/green]", expand=False))


# ---------------------------------------------------------------------------
# convert-seq
# ---------------------------------------------------------------------------

@main.command("convert-seq")
@click.argument("sequence_or_fasta", type=str)
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--compression", default="blosc", show_default=True,
              type=click.Choice(["blosc", "none"]),
              help="Compression codec for the Zarr store.")
@click.option("--pdb-id", default="", help="Identifier stored in metadata (e.g. UniProt accession).")
@click.option("--chain", default="A", show_default=True,
              help="Chain label applied to a raw sequence input.")
def convert_seq(sequence_or_fasta: str, output_path: Path, compression: str,
                pdb_id: str, chain: str):
    """Convert a protein sequence to ProteinTensor (.ptt) format.

    SEQUENCE_OR_FASTA may be a path to a FASTA file or a literal 1-letter
    amino-acid string. The result is a sequence-only .ptt (no coordinates) -
    the primary input form for AlphaFold- and Boltz-style predictors.
    """
    from .converters.sequence import from_sequence, from_fasta
    from .writer import write

    src = Path(sequence_or_fasta)
    is_file = src.exists() and src.is_file()

    t0 = time.perf_counter()
    if is_file:
        data = from_fasta(src, pdb_id=pdb_id)
        source_desc = src.name
    else:
        data = from_sequence(sequence_or_fasta, pdb_id=pdb_id, chain_id=chain)
        source_desc = f"<literal sequence: {data.num_residues} aa>"
    build_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    write(data, output_path, compression=compression)
    write_ms = (time.perf_counter() - t0) * 1000

    dst_bytes = sum(f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file())

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_row("PDB ID",    data.pdb_id or "(unknown)")
    tbl.add_row("Chains",    _chain_summary(data.chain_id))
    tbl.add_row("Residues",  f"{data.num_residues:,}")
    tbl.add_row("Structure", "no (sequence-only)")
    tbl.add_row("")
    tbl.add_row("Source",     source_desc)
    tbl.add_row("Build time", f"{build_ms:.1f} ms")
    tbl.add_row("Write time", f"{write_ms:.1f} ms")
    tbl.add_row("Output",     _fmt_bytes(dst_bytes))

    console.print(Panel(tbl, title=f"[green]Converted -> {output_path}[/green]", expand=False))


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def info(path: Path):
    """Print metadata stored in a .ptt file."""
    import zarr
    store = zarr.open(str(path), mode="r")
    attrs = dict(store.attrs)

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    for k, v in attrs.items():
        if k == "created_at" and v is not None:
            v = datetime.datetime.fromtimestamp(v).isoformat(timespec="seconds")
        if k == "resolution" and v is not None:
            v = f"{v:.2f} A"
        if k in ("num_residues", "num_atoms") and v is not None:
            v = f"{int(v):,}"
        tbl.add_row(k, str(v) if v is not None else "N/A")

    console.print(Panel(tbl, title=f"[bold]{path.name}[/bold]", expand=False))


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

@main.command("embeddings")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def embeddings_cmd(path: Path):
    """Show PLM embeddings stored in a .ptt file."""
    from .reader import list_embeddings
    import zarr, datetime

    features = list_embeddings(path)
    if not features:
        console.print(f"[yellow]No embeddings in {path.name}[/yellow]")
        return

    store = zarr.open(str(path), mode="r")
    tbl = Table(title=f"Embeddings - {path.name}")
    tbl.add_column("Model",    style="bold")
    tbl.add_column("Layer",    justify="right")
    tbl.add_column("Shape",    justify="right")
    tbl.add_column("Dtype",    justify="right")
    tbl.add_column("Size",     justify="right")
    tbl.add_column("Seq hash", style="dim")

    for model in features:
        grp   = store[f"embeddings/{model}"]
        attrs = dict(grp.attrs)
        arr   = grp["data"]
        sz    = sum(f.stat().st_size for f in (path/"embeddings"/model).rglob("*")
                   if f.is_file()) if (path/"embeddings"/model).exists() else 0
        h = attrs.get("sequence_hash", "")
        tbl.add_row(
            model,
            str(attrs.get("layer", "?")),
            str(arr.shape),
            attrs.get("dtype", "?"),
            _fmt_bytes(sz),
            (h[:12] + "...") if h else "N/A",
        )

    console.print(tbl)


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------

@main.command("pairs")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--compute", is_flag=True,
              help="Compute and store distance_matrix and contacts if not present.")
@click.option("--threshold", default=8.0, show_default=True,
              help="Contact distance cutoff in Angstroms (used with --compute).")
def pairs(path: Path, compute: bool, threshold: float):
    """Show pair features stored in a .ptt file, optionally computing standard ones."""
    from .reader import list_pair_features
    from .writer import compute_and_store_distances, compute_and_store_contacts
    import zarr, datetime

    if compute:
        store = zarr.open(str(path), mode="r")
        existing = list_pair_features(path)
        if "distance_matrix" not in existing:
            with console.status("Computing distance matrix ..."):
                compute_and_store_distances(path)
            console.print("[green]Stored:[/green] distance_matrix")
        if "contacts" not in existing:
            with console.status(f"Computing contacts (threshold={threshold} A) ..."):
                compute_and_store_contacts(path, threshold=threshold)
            console.print("[green]Stored:[/green] contacts")

    features = list_pair_features(path)
    if not features:
        console.print(f"[yellow]No pair features in {path.name}[/yellow]")
        console.print("Run with [bold]--compute[/bold] to generate distance_matrix and contacts.")
        return

    store = zarr.open(str(path), mode="r")
    tbl = Table(title=f"Pair features - {path.name}")
    tbl.add_column("Name",        style="bold")
    tbl.add_column("Shape",       justify="right")
    tbl.add_column("Dtype",       justify="right")
    tbl.add_column("Symmetric",   justify="center")
    tbl.add_column("Size",        justify="right")
    tbl.add_column("Description")

    for name in features:
        grp   = store[f"pairs/{name}"]
        attrs = dict(grp.attrs)
        arr   = grp["data"]
        sz    = sum(f.stat().st_size for f in (path / "pairs" / name).rglob("*")
                   if f.is_file()) if (path / "pairs" / name).exists() else 0
        tbl.add_row(
            name,
            str(arr.shape),
            attrs.get("dtype", "?"),
            "yes" if attrs.get("symmetric") else "no",
            _fmt_bytes(sz),
            attrs.get("description", ""),
        )

    console.print(tbl)


# ---------------------------------------------------------------------------
# msa-info
# ---------------------------------------------------------------------------

@main.command("msa-info")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def msa_info(path: Path):
    """Show MSA sources and provenance stored in a .ptt file."""
    from .reader import list_msas
    import zarr, datetime

    store = zarr.open(str(path), mode="r")
    sources = list_msas(path)

    if not sources:
        console.print(f"[yellow]No MSA data found in {path.name}[/yellow]")
        return

    for source in sources:
        attrs = dict(store[f"msa/{source}"].attrs)
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_row("Source",      source)
        tbl.add_row("Sequences",   f"{attrs.get('num_sequences', '?'):,}")
        tbl.add_row("Residues",    f"{attrs.get('num_residues', '?'):,}")
        tbl.add_row("Tool",        attrs.get("tool", "?"))
        tbl.add_row("Version",     attrs.get("tool_version", "?"))
        tbl.add_row("Database",    attrs.get("database", "?"))
        tbl.add_row("DB date",     attrs.get("database_date", "?") or "unknown")
        tbl.add_row("Seq hash",    (attrs.get("sequence_hash") or "")[:16] + "...")
        if attrs.get("created_at"):
            ts = datetime.datetime.fromtimestamp(attrs["created_at"]).isoformat(timespec="seconds")
            tbl.add_row("Cached at", ts)
        console.print(Panel(tbl, title=f"[bold]{path.name} / msa / {source}[/bold]", expand=False))


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--ptt-path", type=click.Path(path_type=Path), default=None,
              help="Path to a pre-converted .ptt. Created from INPUT_PATH if omitted.")
@click.option("--rounds", default=10, show_default=True,
              help="Number of timed load iterations per format.")
def benchmark(input_path: Path, ptt_path: Path | None, rounds: int):
    """Compare mmCIF parsing vs ProteinTensor loading speed."""
    from .converters.mmcif import from_mmcif
    from .reader import read
    from .writer import write

    _tmpdir = None

    if ptt_path is None:
        _tmpdir = tempfile.mkdtemp(suffix=".ptt")
        ptt_path = Path(_tmpdir)
        console.print(f"Converting [bold]{input_path.name}[/bold] to a temporary .ptt ...")
        write(from_mmcif(input_path), ptt_path)
        console.print("Conversion complete.\n")

    import zarr
    store = zarr.open(str(ptt_path), mode="r")
    n_res   = store.attrs["num_residues"]
    n_atoms = store.attrs["num_atoms"]

    console.print(
        f"Benchmarking [bold]{rounds}[/bold] rounds - "
        f"[cyan]{n_res:,}[/cyan] residues, [cyan]{n_atoms:,}[/cyan] atoms\n"
    )

    mmcif_ms = _time_rounds(rounds, lambda: from_mmcif(input_path))
    ptt_ms   = _time_rounds(rounds, lambda: read(ptt_path))

    speedup = float(np.median(mmcif_ms)) / float(np.median(ptt_ms))

    tbl = Table(title="Benchmark Results")
    tbl.add_column("Metric",        style="bold")
    tbl.add_column("mmCIF",         justify="right")
    tbl.add_column("ProteinTensor", justify="right")
    tbl.add_column("Speedup",       justify="right", style="green")

    def _row(label: str, a: float, b: float) -> None:
        tbl.add_row(label, f"{a:.1f} ms", f"{b:.1f} ms", f"{a/b:.1f}x" if b > 0 else "-")

    _row("Median", float(np.median(mmcif_ms)), float(np.median(ptt_ms)))
    _row("Mean",   float(np.mean(mmcif_ms)),   float(np.mean(ptt_ms)))
    _row("Min",    float(np.min(mmcif_ms)),     float(np.min(ptt_ms)))
    _row("P95",    float(np.percentile(mmcif_ms, 95)), float(np.percentile(ptt_ms, 95)))

    console.print(tbl)
    console.print(f"\n[bold green]Overall speedup: {speedup:.1f}x faster[/bold green]")

    if _tmpdir:
        import shutil
        shutil.rmtree(_tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# convert-dir
# ---------------------------------------------------------------------------

@main.command("convert-dir")
@click.argument("input_dir",  type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--compression", default="blosc", show_default=True,
              type=click.Choice(["blosc", "none"]),
              help="Compression codec for the Zarr stores.")
@click.option("--workers", default=0, show_default=True,
              help="Parallel worker processes (0 = auto: min(8, CPU count)).")
@click.option("--recursive", is_flag=True, help="Search INPUT_DIR recursively.")
@click.option("--skip-existing/--overwrite", default=True, show_default=True,
              help="Skip inputs whose .ptt already exists, or rebuild them.")
def convert_dir(input_dir: Path, output_dir: Path, compression: str,
                workers: int, recursive: bool, skip_existing: bool):
    """Batch-convert a directory of mmCIF/PDB files to .ptt format.

    Discovers .cif/.mmcif/.pdb/.ent files in INPUT_DIR and writes one .ptt per
    structure into OUTPUT_DIR, in parallel with progress reporting. Files that
    fail to parse are skipped and listed in the summary.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from rich.progress import (
        Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    files = _discover_structures(input_dir, recursive)
    if not files:
        console.print(
            f"[yellow]No structure files found in {input_dir}[/yellow] "
            f"(looked for {', '.join(sorted(SUPPORTED_INPUT))})"
        )
        return

    tasks: list[tuple[str, str, str]] = []
    skipped_existing = 0
    for f in files:
        out = output_dir / f"{f.stem}.ptt"
        if skip_existing and out.exists():
            skipped_existing += 1
            continue
        tasks.append((str(f), str(out), compression))

    if not tasks:
        console.print(
            f"[green]All {len(files)} structures already converted[/green] "
            f"({skipped_existing} skipped). Use --overwrite to rebuild."
        )
        return

    if workers <= 0:
        workers = min(8, os.cpu_count() or 1)
    workers = min(workers, len(tasks))

    results: list[dict] = []
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("Converting", total=len(tasks))
        if workers == 1:
            for t in tasks:
                results.append(_convert_one_file(t))
                progress.advance(bar)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_convert_one_file, t) for t in tasks]
                for fut in as_completed(futures):
                    results.append(fut.result())
                    progress.advance(bar)

    ok     = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_row("Input dir",          str(input_dir))
    tbl.add_row("Output dir",         str(output_dir))
    tbl.add_row("Converted",          f"{len(ok)}")
    tbl.add_row("Failed",             f"{len(failed)}")
    tbl.add_row("Skipped (existing)", f"{skipped_existing}")
    tbl.add_row("Workers",            f"{workers}")
    if ok:
        tbl.add_row("Total residues", f"{sum(r['n_res'] for r in ok):,}")
    console.print(Panel(tbl, title="[green]Batch conversion complete[/green]", expand=False))

    if failed:
        ftbl = Table(title="[red]Failed conversions[/red]", box=None, padding=(0, 2))
        ftbl.add_column("File", style="bold")
        ftbl.add_column("Error")
        for r in failed[:20]:
            ftbl.add_row(r["file"], r["error"])
        console.print(ftbl)
        if len(failed) > 20:
            console.print(f"... and {len(failed) - 20} more failures")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _discover_structures(input_dir: Path, recursive: bool) -> list[Path]:
    """Return sorted structure files (by supported extension) under input_dir."""
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        p for p in it if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT
    )


def _convert_one_file(task: tuple[str, str, str]) -> dict:
    """Convert one structure file to .ptt. Module-level so it is picklable for
    ProcessPoolExecutor. Returns a result dict; never raises."""
    inp, outp, compression = task
    from .converters.mmcif import from_mmcif
    from .writer import write
    try:
        t0 = time.perf_counter()
        data = from_mmcif(Path(inp))
        write(data, Path(outp), compression=compression)
        return {
            "file": Path(inp).name,
            "ok": True,
            "n_res": int(data.sequence_tokens.shape[0]),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as exc:  # report and continue - one bad file must not stop the batch
        return {"file": Path(inp).name, "ok": False, "error": str(exc)}


def _time_rounds(n: int, fn) -> np.ndarray:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"


def _chain_summary(chain_id: np.ndarray) -> str:
    unique = sorted({c.decode() if isinstance(c, bytes) else c for c in chain_id})
    if len(unique) <= 6:
        return ", ".join(unique)
    return f"{', '.join(unique[:6])}, … ({len(unique)} chains)"
