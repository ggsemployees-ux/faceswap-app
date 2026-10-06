"""Model package verification gate per spec §14.3, §18, FR-016, and ADR-010.

A package is loaded, checked against the build-mode license policy, then checksum-verified.
Failures become FaceSwapError values. No inference session is created.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from faceswap.models.integrity import DEFAULT_CHUNK_SIZE, IntegrityError, verify_model_file
from faceswap.models.manifest import ApprovalStatus, ManifestError, ModelManifest, load_manifest
from faceswap.runtime.errors import ErrorCode, FaceSwapError, make_error


class BuildMode(enum.StrEnum):
    """Release policy applied before a model package may be used."""

    DEVELOPMENT = "development"
    COMMERCIAL = "commercial"


@dataclass(frozen=True, slots=True)
class VerifiedModel:
    """A manifest whose license policy and model-file checksum have been accepted."""

    manifest: ModelManifest
    path: Path


def check_license_policy(manifest: ModelManifest, mode: BuildMode) -> None:
    """Raise FaceSwapError when manifest violates the license rules for mode.

    Every violation is collected into one error. Rejected models fail in every mode.
    Commercial builds also require approval, commercial use, redistribution, and a
    license id other than UNKNOWN.
    """
    _require_mode(mode)
    violations: list[str] = []
    status = manifest.approval_status
    if status is ApprovalStatus.REJECTED:
        violations.append(f"approval_status {status.value} is rejected")
    if mode is BuildMode.COMMERCIAL:
        if status is not ApprovalStatus.APPROVED:
            violations.append(f"approval_status {status.value} is not approved")
        if not manifest.commercial_use_allowed:
            violations.append("commercial_use_allowed is false")
        if not manifest.redistribution_allowed:
            violations.append("redistribution_allowed is false")
        if manifest.license_id.strip().casefold() == "unknown":
            violations.append(f"license_id {manifest.license_id.strip()} is UNKNOWN")
    if violations:
        detail = f"{manifest.model_id}: {'; '.join(violations)}"
        raise FaceSwapError(make_error(ErrorCode.MODEL_LICENSE_INVALID, detail))


def verify_model_package(
    manifest_path: Path,
    mode: BuildMode,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> VerifiedModel:
    """Load, license-check, and checksum-verify the package at manifest_path.

    The model file is the manifest's file_name in the same directory. License policy
    runs before the file is hashed.
    """
    _require_mode(mode)
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        field = exc.field if exc.field is not None else "none"
        detail = f"{manifest_path.name} field {field}"
        raise FaceSwapError(make_error(ErrorCode.MODEL_CHECKSUM_INVALID, detail)) from exc
    except OSError as exc:
        detail = f"manifest could not be read: {manifest_path.name}"
        raise FaceSwapError(make_error(ErrorCode.MODEL_CHECKSUM_INVALID, detail)) from exc
    check_license_policy(manifest, mode)
    try:
        resolved = verify_model_file(manifest, manifest_path.parent, chunk_size=chunk_size)
    except IntegrityError as exc:
        detail = f"{exc.path.name} expected {exc.expected} actual {exc.actual}"
        raise FaceSwapError(make_error(ErrorCode.MODEL_CHECKSUM_INVALID, detail)) from exc
    except OSError as exc:
        detail = f"model file missing or unreadable: {manifest.file_name}"
        raise FaceSwapError(make_error(ErrorCode.MODEL_CHECKSUM_INVALID, detail)) from exc
    return VerifiedModel(manifest=manifest, path=resolved)


class ModelManager:
    """Verifies model packages for one build mode. Holds no other state."""

    def __init__(self, mode: BuildMode) -> None:
        """Store the build mode used for every later verification."""
        self._mode = _require_mode(mode)

    @property
    def mode(self) -> BuildMode:
        """Build mode supplied when this manager was created."""
        return self._mode

    def verify_package(self, path: Path) -> VerifiedModel:
        """Verify one manifest path."""
        return verify_model_package(path, self._mode)

    def verify_packages(self, paths: Sequence[Path]) -> tuple[VerifiedModel, ...]:
        """Verify each path in order and stop at the first failure.

        Two packages with the same model_id are a checksum failure.
        """
        verified: list[VerifiedModel] = []
        seen: set[str] = set()
        for path in paths:
            model = self.verify_package(path)
            model_id = model.manifest.model_id
            if model_id in seen:
                detail = f"duplicate model_id {model_id}"
                raise FaceSwapError(make_error(ErrorCode.MODEL_CHECKSUM_INVALID, detail))
            seen.add(model_id)
            verified.append(model)
        return tuple(verified)


def _require_mode(mode: object) -> BuildMode:
    if not isinstance(mode, BuildMode):
        raise TypeError("mode must be a BuildMode")
    return mode
