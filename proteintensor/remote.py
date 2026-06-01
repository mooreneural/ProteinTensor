"""
Cloud storage support for ProteinTensor via fsspec.

Enables reading .ptt files from S3, GCS, Azure Blob, or any fsspec-compatible
storage without downloading them first. Zarr handles the actual byte I/O; this
module provides the URL-aware store opener and consolidated-metadata helpers.

Supported URL schemes (requires the matching backend package):
  s3://bucket/path.ptt          pip install s3fs
  gs://bucket/path.ptt          pip install gcsfs
  az://container/path.ptt       pip install adlfs
  memory://path.ptt             built into fsspec (useful for testing)

Consolidated metadata
---------------------
A .ptt file has many Zarr sub-groups. Without consolidated metadata, opening a
remote store issues one HTTP request per group/array just to read shapes,
dtypes, and chunk layouts. Calling consolidate() once before uploading writes a
single .zmetadata file so every subsequent open reads all metadata in one request.

Typical workflow
----------------
  # Local build
  pt.write(data, "1abc.ptt")
  pt.add_msa("1abc.ptt", msa)
  pt.consolidate("1abc.ptt")          # write .zmetadata

  # Upload (any S3-compatible tool)
  aws s3 cp -r 1abc.ptt s3://my-bucket/proteins/1abc.ptt

  # Train directly from cloud
  data = pt.read("s3://my-bucket/proteins/1abc.ptt")
  ds   = pt.ProteinDataset("s3://my-bucket/training.ptt")
"""
from __future__ import annotations

from pathlib import Path
import zarr


def open_store(
    path: str | Path,
    mode: str = "r",
    storage_options: dict | None = None,
) -> zarr.Group:
    """Open a Zarr group from a local path or remote fsspec URL.

    For remote paths, tries consolidated metadata first (.zmetadata) and
    falls back to un-consolidated if the file has not been consolidated yet.

    Parameters
    ----------
    path             Local path (str or Path) or fsspec URL
                     (s3://, gs://, az://, memory://, ...)
    mode             Zarr open mode: "r" read-only, "r+" read-write, "a" append
    storage_options  Extra kwargs forwarded to fsspec.get_mapper() -
                     credentials, endpoint_url, etc.

    Returns
    -------
    zarr.Group rooted at path.
    """
    path_str = str(path)
    if _is_url(path_str):
        _require_fsspec()
        import fsspec
        mapper = fsspec.get_mapper(path_str, **(storage_options or {}))
        try:
            return zarr.open_consolidated(mapper, mode=mode)
        except (KeyError, ValueError):
            return zarr.open(mapper, mode=mode)
    return zarr.open(path_str, mode=mode)


def consolidate(
    path: str | Path,
    storage_options: dict | None = None,
) -> None:
    """Write consolidated Zarr metadata for fast remote reads.

    Writes a .zmetadata file at the root of the store. Subsequent remote opens
    via open_store() will use it to fetch all metadata in a single HTTP request.

    Call this once after finishing a .ptt file (after all add_msa, add_embedding,
    add_pair_feature calls) before uploading to cloud storage. It is safe to call
    on an already-consolidated store.

    Parameters
    ----------
    path             Local or remote path to a .ptt or dataset .ptt file.
    storage_options  fsspec options for remote paths.
    """
    path_str = str(path)
    if _is_url(path_str):
        _require_fsspec()
        import fsspec
        mapper = fsspec.get_mapper(path_str, **(storage_options or {}))
        zarr.consolidate_metadata(mapper)
    else:
        zarr.consolidate_metadata(path_str)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _is_url(path: str) -> bool:
    return "://" in path


def _require_fsspec() -> None:
    try:
        import fsspec  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "fsspec is required for remote storage access. "
            "Install it with:  pip install 'proteintensor[cloud]'"
        ) from exc
