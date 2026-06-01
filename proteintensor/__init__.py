from .reader import read, mmap_positions, mmap_tokens
from .writer import write
from .schema import ProteinTensorData, AA_VOCAB, AA_VOCAB_SIZE, FORMAT_VERSION

__version__ = "0.1.0"

__all__ = [
    "read",
    "write",
    "mmap_positions",
    "mmap_tokens",
    "ProteinTensorData",
    "AA_VOCAB",
    "AA_VOCAB_SIZE",
    "FORMAT_VERSION",
]
