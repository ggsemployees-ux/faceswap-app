"""Model manifest parsing per spec §14.1, §14.3, and FR-016.

A manifest is one UTF-8 JSON object. This module validates that document and returns an
immutable ModelManifest. Checksum verification against a file, and license, approval, and
commercial gating, are later steps (D-10.9).
"""

from __future__ import annotations

import enum
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

MANIFEST_SCHEMA_VERSION = 1
"""Supported model-manifest schema version."""

REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "model_id",
        "model_version",
        "model_role",
        "file_name",
        "sha256",
        "input_names",
        "input_shapes",
        "normalization",
        "output_contract",
        "supported_execution_providers",
        "minimum_vram_mb",
        "license_id",
        "redistribution_allowed",
        "commercial_use_allowed",
        "attribution_text",
        "source_url",
        "approval_status",
    }
)
"""JSON keys required by spec §14.1."""


class ModelRole(enum.StrEnum):
    """Pipeline stage a model package implements."""

    DETECTOR = "detector"
    LANDMARKS = "landmarks"
    EMBEDDER = "embedder"
    SWAPPER = "swapper"
    ENHANCER = "enhancer"


class ApprovalStatus(enum.StrEnum):
    """Review state recorded in the manifest."""

    APPROVED = "approved"
    PENDING = "pending"
    REJECTED = "rejected"


class ManifestError(ValueError):
    """Invalid model manifest.

    field is the offending key, or None when the whole document is unusable.
    The message names that field and the rule that failed, and does not include file content.
    """

    field: str | None

    def __init__(self, message: str, *, field: str | None) -> None:
        super().__init__(message)
        self.field = field


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Immutable validated manifest. Lists are tuples and JSON objects are read-only."""

    model_id: str
    model_version: str
    model_role: ModelRole
    file_name: str
    sha256: str
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int | str, ...], ...]
    normalization: Mapping[str, object]
    output_contract: Mapping[str, object]
    supported_execution_providers: tuple[str, ...]
    minimum_vram_mb: int
    license_id: str
    redistribution_allowed: bool
    commercial_use_allowed: bool
    attribution_text: str
    source_url: str
    approval_status: ApprovalStatus


def parse_manifest(data: object) -> ModelManifest:
    """Validate an already-decoded JSON value and return an immutable manifest."""
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object", field=None)
    _reject_missing_and_unknown(data)
    _require_schema_version(data["schema_version"])
    input_names = _require_name_list(data["input_names"], "input_names")
    return ModelManifest(
        model_id=_require_non_blank_str(data["model_id"], "model_id"),
        model_version=_require_non_blank_str(data["model_version"], "model_version"),
        model_role=_require_role(data["model_role"]),
        file_name=_require_file_name(data["file_name"]),
        sha256=_require_sha256(data["sha256"]),
        input_names=input_names,
        input_shapes=_require_shapes(data["input_shapes"], len(input_names)),
        normalization=_require_object(data["normalization"], "normalization"),
        output_contract=_require_object(data["output_contract"], "output_contract"),
        supported_execution_providers=_require_name_list(
            data["supported_execution_providers"],
            "supported_execution_providers",
        ),
        minimum_vram_mb=_require_non_negative_int(data["minimum_vram_mb"], "minimum_vram_mb"),
        license_id=_require_non_blank_str(data["license_id"], "license_id"),
        redistribution_allowed=_require_bool(
            data["redistribution_allowed"],
            "redistribution_allowed",
        ),
        commercial_use_allowed=_require_bool(
            data["commercial_use_allowed"],
            "commercial_use_allowed",
        ),
        attribution_text=_require_str(data["attribution_text"], "attribution_text"),
        source_url=_require_str(data["source_url"], "source_url"),
        approval_status=_require_approval(data["approval_status"]),
    )


def load_manifest(path: Path) -> ModelManifest:
    """Read one UTF-8 manifest file and validate it.

    Invalid text or JSON becomes ManifestError. FileNotFoundError and other OSError
    propagate unchanged.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError("manifest file is not valid UTF-8", field=None) from exc
    try:
        data: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError("manifest file is not valid JSON", field=None) from exc
    return parse_manifest(data)


def _reject_missing_and_unknown(data: dict[object, object]) -> None:
    missing = sorted(key for key in REQUIRED_KEYS if key not in data)
    if missing:
        field = missing[0]
        raise ManifestError(f"{field} is required", field=field)
    for key in data:
        if key not in REQUIRED_KEYS:
            if not isinstance(key, str):
                raise ManifestError("manifest keys must be strings", field=None)
            raise ManifestError(f"{key} is not a manifest field", field=key)


def _require_schema_version(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != MANIFEST_SCHEMA_VERSION:
        raise ManifestError("schema_version must be the integer 1", field="schema_version")


def _require_non_blank_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-blank string", field=field)
    return value


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{field} must be a string", field=field)
    return value


def _require_file_name(value: object) -> str:
    name = _require_non_blank_str(value, "file_name")
    if "/" in name or "\\" in name or ":" in name or name in {".", ".."} or Path(name).name != name:
        raise ManifestError(
            "file_name must be a plain file name without a directory or drive",
            field="file_name",
        )
    return name


def _require_sha256(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ManifestError("sha256 must be 64 hexadecimal characters", field="sha256")
    return value.lower()


def _require_role(value: object) -> ModelRole:
    if not isinstance(value, str):
        raise ManifestError(
            "model_role must be detector, landmarks, embedder, swapper, or enhancer",
            field="model_role",
        )
    try:
        return ModelRole(value)
    except ValueError:
        raise ManifestError(
            "model_role must be detector, landmarks, embedder, swapper, or enhancer",
            field="model_role",
        ) from None


def _require_approval(value: object) -> ApprovalStatus:
    if not isinstance(value, str):
        raise ManifestError(
            "approval_status must be approved, pending, or rejected",
            field="approval_status",
        )
    try:
        return ApprovalStatus(value)
    except ValueError:
        raise ManifestError(
            "approval_status must be approved, pending, or rejected",
            field="approval_status",
        ) from None


def _require_name_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) == 0:
        raise ManifestError(
            f"{field} must be a non-empty list of non-blank strings",
            field=field,
        )
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ManifestError(
                f"{field} must be a non-empty list of non-blank strings",
                field=field,
            )
        names.append(item)
    return tuple(names)


def _require_shapes(value: object, count: int) -> tuple[tuple[int | str, ...], ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ManifestError(
            "input_shapes must be a list with one shape per input name",
            field="input_shapes",
        )
    shapes: list[tuple[int | str, ...]] = []
    for shape in value:
        if not isinstance(shape, list) or len(shape) == 0:
            raise ManifestError(
                "input_shapes entries must be non-empty lists of dimensions",
                field="input_shapes",
            )
        shapes.append(tuple(_require_dimension(dimension) for dimension in shape))
    return tuple(shapes)


def _require_dimension(value: object) -> int | str:
    if isinstance(value, str):
        if value.strip():
            return value
    elif not isinstance(value, bool) and isinstance(value, int) and value > 0:
        return value
    raise ManifestError(
        "input_shapes dimensions must be positive integers or non-blank strings",
        field="input_shapes",
    )


def _require_non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(
            f"{field} must be an integer greater than or equal to 0",
            field=field,
        )
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{field} must be a boolean", field=field)
    return value


def _require_object(value: object, field: str) -> Mapping[str, object]:
    frozen = _freeze_value(value, field)
    if isinstance(value, dict) and isinstance(frozen, Mapping):
        return frozen
    raise ManifestError(f"{field} must be a JSON object", field=field)


def _freeze_value(value: object, field: str) -> object:
    if isinstance(value, dict):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ManifestError(f"{field} keys must be strings", field=field)
            frozen[key] = _freeze_value(item, field)
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze_value(item, field) for item in value)
    return value
