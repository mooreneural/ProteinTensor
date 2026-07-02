"""Tests for the batch convert-dir CLI command."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from proteintensor.cli import main, _discover_structures, _convert_one_file


# ---------------------------------------------------------------------------
# file discovery
# ---------------------------------------------------------------------------

def test_discover_structures_filters_by_extension(tmp_path):
    for name in ("a.cif", "b.pdb", "c.mmcif", "e.ent", "d.txt", "notes.md"):
        (tmp_path / name).write_text("")
    found = _discover_structures(tmp_path, recursive=False)
    names = sorted(p.name for p in found)
    assert names == ["a.cif", "b.pdb", "c.mmcif", "e.ent"]


def test_discover_structures_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.cif").write_text("")
    (sub / "b.cif").write_text("")
    assert len(_discover_structures(tmp_path, recursive=False)) == 1
    assert len(_discover_structures(tmp_path, recursive=True)) == 2


def test_convert_dir_no_files(tmp_path):
    indir = tmp_path / "in"
    indir.mkdir()
    result = CliRunner().invoke(main, ["convert-dir", str(indir), str(tmp_path / "out")])
    assert result.exit_code == 0
    assert "No structure files found" in result.output


def test_convert_one_file_reports_error_for_bad_input(tmp_path):
    bad = tmp_path / "bad.cif"
    bad.write_text("this is not a structure file")
    out = tmp_path / "bad.ptt"
    result = _convert_one_file((str(bad), str(out), "blosc", False))
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# integration (needs a real structure file present in the repo root)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not Path("1UBQ.cif").exists(), reason="1UBQ.cif not present")
def test_convert_dir_integration(tmp_path):
    import proteintensor as pt

    indir = tmp_path / "in"
    indir.mkdir()
    shutil.copy("1UBQ.cif", indir / "1UBQ.cif")
    outdir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(main, ["convert-dir", str(indir), str(outdir), "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert (outdir / "1UBQ.ptt").exists()
    assert pt.read(str(outdir / "1UBQ.ptt")).num_residues == 76

    # second run skips the already-converted file
    again = runner.invoke(main, ["convert-dir", str(indir), str(outdir), "--workers", "1"])
    assert again.exit_code == 0
    assert "already converted" in again.output


@pytest.mark.skipif(not Path("1UBQ.cif").exists(), reason="1UBQ.cif not present")
def test_convert_dir_overwrite_rebuilds(tmp_path):
    indir = tmp_path / "in"
    indir.mkdir()
    shutil.copy("1UBQ.cif", indir / "1UBQ.cif")
    outdir = tmp_path / "out"

    runner = CliRunner()
    runner.invoke(main, ["convert-dir", str(indir), str(outdir), "--workers", "1"])
    result = runner.invoke(
        main, ["convert-dir", str(indir), str(outdir), "--workers", "1", "--overwrite"]
    )
    assert result.exit_code == 0
    assert "Converted" in result.output
