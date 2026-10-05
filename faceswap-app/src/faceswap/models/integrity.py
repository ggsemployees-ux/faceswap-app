"""SHA-256 model file verification per FR-016, spec §14.3, §18, and §33.

A model file is hashed in chunks and compared with the manifest digest before it is used.
License and approval gating, and mapping failures to AppError, belong to Step 12 (D-11.6).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from faceswap.models.manifest import ModelManifest

DEFAULT_CHUNK_SIZE = 1024 * 1024
"""Binary read size for streaming SHA-256. A file is never loaded all at once."""


class IntegrityError(ValueError):
    """Checksum mismatch or a model file that escapes its model directory.

    path is the file that was checked. expected and actual are lowercase hex digests,
    and actual is empty when the path is rejected before it is read. The message names
    the file and both digests. It does not include file contents.
    """

    path: Path
    expected: str
    actual: str

    def __init__(
        self,
        path: Path,
        expected: str,
        actual: str,
        *,
        outside_directory: bool = False,
    ) -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        problem = "is outside the model directory" if outside_directory else "checksum mismatch"
        super().__init__(f"{path.name} {problem}: expected {expected}, actual {actual}")


def compute_sha256(path: Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the lowercase SHA-256 hex digest of path, reading it in chunks."""
    size = _require_chunk_size(chunk_size)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: Path,
    expected_sha256: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Raise IntegrityError when path's SHA-256 differs from expected_sha256.

    expected_sha256 must be 64 hexadecimal characters. A missing file raises
    FileNotFoundError.
    """
    size = _require_chunk_size(chunk_size)
    expected = _require_digest(expected_sha256)
    actual = compute_sha256(path, chunk_size=size)
    if actual != expected:
        raise IntegrityError(path, expected, actual)
    return None


def verify_model_file(
    manifest: ModelManifest,
    model_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    """Verify manifest.file_name inside model_dir and return its resolved path.

    A name that resolves outside model_dir raises IntegrityError before the file is read.
    """
    size = _require_chunk_size(chunk_size)
    base = model_dir.resolve()
    candidate = (model_dir / manifest.file_name).resolve()
    if not candidate.is_relative_to(base):
        raise IntegrityError(
            candidate,
            manifest.sha256,
            "",
            outside_directory=True,
        )
    verify_file(candidate, manifest.sha256, chunk_size=size)
    return candidate


def _require_chunk_size(chunk_size: int) -> int:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an int")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return chunk_size


def _require_digest(expected_sha256: object) -> str:
    if not isinstance(expected_sha256, str):
        raise TypeError("expected_sha256 must be a str")
    if re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None:
        raise ValueError("expected_sha256 must be 64 hexadecimal characters")
    return expected_sha256.lower()
