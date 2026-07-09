from .reader import (
    read, read_backbone, read_bonds, read_msa, read_pair_feature, read_embedding,
    list_msas, list_pair_features, list_embeddings,
    read_pair_feature_sparse, list_pair_features_sparse,
    mmap_positions, mmap_tokens, mmap_backbone, mmap_msa_tokens,
    mmap_pair_feature, mmap_embedding,
)
from .writer import (
    write, add_msa, add_pair_feature, add_embedding,
    compute_and_store_distances, compute_and_store_contacts,
    add_pair_feature_sparse,
    compute_and_store_distances_sparse, compute_and_store_contacts_sparse,
)
from .msa import MsaData, from_a3m, compute_profile, MSA_GAP, MSA_MASK, MSA_VOCAB_SIZE
from .pairs import (
    PairFeature, SparsePairFeature, compute_distance_matrix, compute_contact_map,
)
from .embeddings import EmbeddingData, KNOWN_DIMS, sequence_hash as embedding_sequence_hash
from .adapters.boltz import BoltzAdapter
from .adapters.alphafold3 import AlphaFold3Adapter
from .adapters.chai import ChaiAdapter
from .adapters.openfold import OpenFoldAdapter
from .schema import (
    ProteinTensorData,
    BackboneData,
    BondData,
    AA_VOCAB,
    AA_VOCAB_SIZE,
    NUC_VOCAB,
    MOL_PROTEIN,
    MOL_DNA,
    MOL_RNA,
    BACKBONE_ATOMS,
    N_BACKBONE,
    FORMAT_VERSION,
)
from .bonds import (
    BOND_SINGLE,
    BOND_DOUBLE,
    BOND_TRIPLE,
    BOND_AROMATIC,
    BOND_PEPTIDE,
    BOND_DISULFIDE,
    BOND_TYPE_NAMES,
)
from .dataset import ProteinDataset, create_dataset, add_to_dataset
from .remote import consolidate
from .converters import from_mmcif, from_sequence, from_fasta, parse_fasta
from .ligands import (
    read_ligands, list_ligands, add_ligand, from_smiles,
    compute_and_store_pocket, read_binding_site, read_interactions,
)
from .schema import LigandData

__version__ = "0.4.0"

__all__ = [
    # Converters - input
    "from_mmcif", "from_sequence", "from_fasta", "parse_fasta",
    # I/O - structure
    "read", "write",
    "read_backbone", "read_bonds",
    "mmap_positions", "mmap_tokens", "mmap_backbone",
    # I/O - MSA
    "read_msa", "add_msa", "list_msas", "mmap_msa_tokens",
    # I/O - pair features
    "read_pair_feature", "add_pair_feature", "list_pair_features", "mmap_pair_feature",
    "compute_and_store_distances", "compute_and_store_contacts",
    # I/O - sparse pair features
    "read_pair_feature_sparse", "list_pair_features_sparse", "add_pair_feature_sparse",
    "compute_and_store_distances_sparse", "compute_and_store_contacts_sparse",
    "SparsePairFeature",
    # I/O - embeddings
    "read_embedding", "add_embedding", "list_embeddings", "mmap_embedding",
    # Ligands / small molecules
    "read_ligands", "list_ligands", "add_ligand", "from_smiles", "LigandData",
    "compute_and_store_pocket", "read_binding_site", "read_interactions",
    # Data containers
    "ProteinTensorData", "BackboneData", "BondData", "MsaData", "PairFeature", "EmbeddingData",
    # MSA utilities
    "from_a3m", "compute_profile",
    "MSA_GAP", "MSA_MASK", "MSA_VOCAB_SIZE",
    # Pair utilities
    "compute_distance_matrix", "compute_contact_map",
    # Embedding utilities
    "KNOWN_DIMS", "embedding_sequence_hash",
    # Adapters
    "BoltzAdapter", "AlphaFold3Adapter", "ChaiAdapter", "OpenFoldAdapter",
    # Schema constants
    "AA_VOCAB", "AA_VOCAB_SIZE", "NUC_VOCAB", "MOL_PROTEIN", "MOL_DNA", "MOL_RNA",
    "BACKBONE_ATOMS", "N_BACKBONE", "FORMAT_VERSION",
    # Bond constants
    "BOND_SINGLE", "BOND_DOUBLE", "BOND_TRIPLE",
    "BOND_AROMATIC", "BOND_PEPTIDE", "BOND_DISULFIDE", "BOND_TYPE_NAMES",
    # Dataset
    "ProteinDataset", "create_dataset", "add_to_dataset",
    # Cloud / remote
    "consolidate",
]
