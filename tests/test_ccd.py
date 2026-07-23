"""Tests for CCD-code -> SMILES resolution and ligand SMILES backfill."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

import proteintensor as pt
from proteintensor.ccd import parse_smiles, ccd_to_smiles


def _online() -> bool:
    try:
        urllib.request.urlopen("https://data.rcsb.org/rest/v1/core/chemcomp/HOH", timeout=5)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# parse_smiles - pure, offline
# --------------------------------------------------------------------------

def test_parse_prefers_stereo():
    j = {"rcsb_chem_comp_descriptor": {"SMILES_stereo": "C[C@H](O)N", "SMILES": "CC(O)N"}}
    assert parse_smiles(j) == "C[C@H](O)N"


def test_parse_falls_back_to_plain():
    assert parse_smiles({"rcsb_chem_comp_descriptor": {"SMILES": "CCO"}}) == "CCO"


def test_parse_pdbx_cactvs_fallback():
    j = {"rcsb_chem_comp_descriptor": {},
         "pdbx_chem_comp_descriptor": [
             {"type": "SMILES", "program": "ACDLabs", "descriptor": "ignored"},
             {"type": "SMILES_CANONICAL", "program": "CACTVS", "descriptor": "[Mg+2]"},
         ]}
    assert parse_smiles(j) == "[Mg+2]"


def test_parse_none_when_absent():
    assert parse_smiles({}) is None


# --------------------------------------------------------------------------
# ccd_to_smiles - offline cache behavior
# --------------------------------------------------------------------------

def test_ccd_offline_hits_cache(tmp_path):
    (tmp_path / "ccd_smiles.json").write_text(json.dumps({"GDP": "NC1=Nc2..."}))
    assert ccd_to_smiles("gdp", allow_network=False, cache_dir=str(tmp_path)) == "NC1=Nc2..."


def test_ccd_offline_uncached_is_none(tmp_path):
    assert ccd_to_smiles("ZZZ", allow_network=False, cache_dir=str(tmp_path)) is None


# --------------------------------------------------------------------------
# resolve_ligand_smiles - offline via seeded cache, on a real structure
# --------------------------------------------------------------------------

@pytest.mark.skipif(not Path("6OIM.cif").exists(), reason="6OIM.cif not present")
def test_resolve_backfills_from_cache(tmp_path):
    ptt = tmp_path / "6OIM.ptt"
    pt.write(pt.from_mmcif("6OIM.cif", include_ligands=True), str(ptt))
    codes = pt.list_ligands(str(ptt))               # ['MG', 'GDP', 'MOV']
    # seed the cache so no network is needed
    cdir = tmp_path / "cache"
    cdir.mkdir()
    (cdir / "ccd_smiles.json").write_text(json.dumps({c: f"SMILES_{c}" for c in codes}))

    n = pt.resolve_ligand_smiles(str(ptt), allow_network=False, cache_dir=str(cdir))
    assert n == len(codes)
    smis = {l.name: l.smiles for l in pt.read_ligands(str(ptt))}
    assert smis["GDP"] == "SMILES_GDP"


# --------------------------------------------------------------------------
# live network check (skipped offline)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not _online(), reason="no network")
def test_ccd_live_resolves_water(tmp_path):
    smi = ccd_to_smiles("HOH", allow_network=True, cache_dir=str(tmp_path))
    assert smi and "O" in smi   # water resolves to an oxygen-containing SMILES
