"""Resolve PDB Chemical Component Dictionary (CCD) codes to SMILES.

Structure-extracted ligands carry their 3-letter CCD code (e.g. GDP, HEM, and
drug codes) but not a SMILES string. This module fetches the canonical SMILES
from the authoritative RCSB Chemical Component Dictionary - never hand-written -
and caches each code locally so it is fetched at most once.

Network access is opt-in: the resolver only runs when you call it.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

RCSB_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
_DEFAULT_CACHE = Path.home() / ".cache" / "proteintensor" / "ccd_smiles.json"


def parse_smiles(chemcomp_json: dict) -> str | None:
    """Extract a canonical SMILES from an RCSB chemcomp JSON response.

    Prefers the RCSB-curated stereo SMILES, then the plain SMILES, then a
    CACTVS/OpenEye canonical SMILES from the pdbx descriptor list.
    """
    desc = chemcomp_json.get("rcsb_chem_comp_descriptor") or {}
    for key in ("SMILES_stereo", "SMILES"):
        if desc.get(key):
            return desc[key]

    fallback = None
    for e in chemcomp_json.get("pdbx_chem_comp_descriptor") or []:
        if e.get("type") == "SMILES_CANONICAL" and e.get("descriptor"):
            if e.get("program") == "CACTVS":
                return e["descriptor"]
            fallback = fallback or e["descriptor"]
    return fallback


def _cache_path(cache_dir: str | Path | None) -> Path:
    base = Path(cache_dir) if cache_dir else _DEFAULT_CACHE.parent
    return base / "ccd_smiles.json"


def _load_cache(path: Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _save_cache(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=0))


def ccd_to_smiles(
    code: str,
    *,
    allow_network: bool = True,
    cache_dir: str | Path | None = None,
    timeout: float = 10.0,
) -> str | None:
    """Resolve a CCD 3-letter code to canonical SMILES (cached).

    Returns None if the code is unknown, the lookup fails, or ``allow_network``
    is False and the code is not already cached. Never raises on network error.
    """
    code = code.strip().upper()
    if not code:
        return None
    path = _cache_path(cache_dir)
    cache = _load_cache(path)
    if code in cache:
        return cache[code] or None
    if not allow_network:
        return None
    try:
        with urllib.request.urlopen(RCSB_URL.format(code=code), timeout=timeout) as r:
            data = json.load(r)
        smi = parse_smiles(data)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    cache[code] = smi or ""       # cache negatives too (empty string)
    _save_cache(path, cache)
    return smi
