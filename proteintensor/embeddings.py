"""
Protein language model embedding storage for ProteinTensor.

Embeddings are stored under embeddings/<model_name>/ inside an existing .ptt
file. Multiple models coexist per file; each is independently versioned.

Supported models (not exhaustive)
----------------------------------
ESM-2 family  : esm2_t6_8M_UR50D      D=320
                esm2_t12_35M_UR50D     D=480
                esm2_t30_150M_UR50D    D=640
                esm2_t33_650M_UR50D    D=1280
                esm2_t36_3B_UR50D      D=2560
                esm2_t48_15B_UR50D     D=5120
ESM-3 family  : esm3_sm_open_v1        D=1152
                esm3_open_v1           D=1536

Storage tip
-----------
Use dtype="float16" to halve memory.  For 650M-parameter ESM-2, a 574-residue
protein is 1.5 MB at float16 vs 3 MB at float32.  Most downstream tasks tolerate
float16 without measurable quality loss.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

# Known embedding dimensions for common models
KNOWN_DIMS: dict[str, int] = {
    "esm2_t6_8M_UR50D":      320,
    "esm2_t12_35M_UR50D":    480,
    "esm2_t30_150M_UR50D":   640,
    "esm2_t33_650M_UR50D":  1280,
    "esm2_t36_3B_UR50D":    2560,
    "esm2_t48_15B_UR50D":   5120,
    "esm3_sm_open_v1":      1152,
    "esm3_open_v1":         1536,
    "esm3_large_multimer_v1": 2560,
}


@dataclass
class EmbeddingData:
    """Per-residue embedding from a protein language model."""

    data:          np.ndarray  # [N_res, D]  float32 or float16
    model:         str         # e.g. "esm2_t33_650M_UR50D"
    layer:         int         # source layer index (-1 = final layer)
    dim:           int         # D
    dtype:         str         # "float32" | "float16"
    sequence_hash: str         # SHA-256 of the input sequence for cache validation
    created_at:    float

    @property
    def num_residues(self) -> int:
        return int(self.data.shape[0])


def sequence_hash(tokens: np.ndarray) -> str:
    """Compute SHA-256 of an int32 sequence token array."""
    return hashlib.sha256(tokens.astype(np.int32).tobytes()).hexdigest()
