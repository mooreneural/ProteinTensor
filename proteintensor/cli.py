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
    """ProteinTensor — AI-native biomolecular tensor format."""


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
        f"Benchmarking [bold]{rounds}[/bold] rounds — "
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
        tbl.add_row(label, f"{a:.1f} ms", f"{b:.1f} ms", f"{a/b:.1f}x" if b > 0 else "—")

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
# helpers
# ---------------------------------------------------------------------------

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
