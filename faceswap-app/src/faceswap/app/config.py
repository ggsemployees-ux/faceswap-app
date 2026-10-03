"""Immutable session configuration snapshot per spec §22.1.

Persisted user settings (FR-018) are a separate versioned JSON document.
"""

from __future__ import annotations

import enum
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


class QualityProfile(enum.StrEnum):
    """Selectable quality profile for a running session."""

    AUTO = "auto"
    PERFORMANCE = "performance"
    QUALITY = "quality"


@dataclass(frozen=True, slots=True)
class DetectionPolicy:
    """How often a running session re-verifies face detection."""

    verification_interval_frames: int = 5

    def __post_init__(self) -> None:
        if self.verification_interval_frames < 1:
            raise ValueError("verification_interval_frames must be >= 1")


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Immutable configuration snapshot for one running session."""

    camera_id: str
    width: int = 1920
    height: int = 1080
    fps: int = 30
    provider: str = "CUDAExecutionProvider"
    model_ids: tuple[str, ...] = ()
    detection_policy: DetectionPolicy = field(default_factory=DetectionPolicy)
    enhancer_enabled: bool = False
    quality_profile: QualityProfile = QualityProfile.AUTO

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty or whitespace")
        if self.width <= 0:
            raise ValueError("width must be > 0")
        if self.height <= 0:
            raise ValueError("height must be > 0")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        if not self.provider.strip():
            raise ValueError("provider must not be empty or whitespace")
        if not isinstance(self.model_ids, tuple):
            raise TypeError("model_ids must be a tuple")


APP_DIR_NAME = "faceswap"
"""Directory name under LOCALAPPDATA for persisted settings."""

SETTINGS_FILE_NAME = "settings.json"
"""File name of the versioned user-settings document."""

SETTINGS_SCHEMA_VERSION = 1
"""Schema version written by save_settings and accepted by load_settings."""


class SettingsLoadStatus(enum.StrEnum):
    """Outcome of reading a settings file without changing it."""

    LOADED = "loaded"
    MISSING = "missing"
    INVALID = "invalid"
    UNSUPPORTED_VERSION = "unsupported_version"


@dataclass(frozen=True, slots=True)
class UserSettings:
    """Persisted non-sensitive preferences. None camera_id means none chosen yet."""

    camera_id: str | None = None
    quality_profile: QualityProfile = QualityProfile.AUTO

    def __post_init__(self) -> None:
        if self.camera_id is not None and not isinstance(self.camera_id, str):
            raise TypeError("camera_id must be a str or None")
        if isinstance(self.camera_id, str) and not self.camera_id.strip():
            raise ValueError("camera_id must not be empty or whitespace")
        if not isinstance(self.quality_profile, QualityProfile):
            raise TypeError("quality_profile must be a QualityProfile")


@dataclass(frozen=True, slots=True)
class SettingsLoadResult:
    """Settings values together with why those values were chosen."""

    settings: UserSettings
    status: SettingsLoadStatus


def default_settings_path(env: Mapping[str, str] | None = None) -> Path:
    """Return LOCALAPPDATA/APP_DIR_NAME/SETTINGS_FILE_NAME without creating it."""
    source = os.environ if env is None else env
    local_app_data = source.get("LOCALAPPDATA")
    if local_app_data is None or not local_app_data.strip():
        raise RuntimeError("LOCALAPPDATA is missing or blank")
    return Path(local_app_data) / APP_DIR_NAME / SETTINGS_FILE_NAME


def load_settings(path: Path) -> SettingsLoadResult:
    """Load settings from path. A bad file yields defaults and is left unchanged."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return SettingsLoadResult(UserSettings(), SettingsLoadStatus.MISSING)
    except UnicodeDecodeError:
        return SettingsLoadResult(UserSettings(), SettingsLoadStatus.INVALID)
    try:
        payload: object = json.loads(text)
    except json.JSONDecodeError:
        return SettingsLoadResult(UserSettings(), SettingsLoadStatus.INVALID)
    return _settings_from_payload(payload)


def save_settings(settings: UserSettings, path: Path) -> None:
    """Atomically replace path with the version-1 JSON for settings."""
    encoded = (
        json.dumps(
            {
                "camera_id": settings.camera_id,
                "quality_profile": settings.quality_profile.value,
                "schema_version": SETTINGS_SCHEMA_VERSION,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, raw_name = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _settings_from_payload(payload: object) -> SettingsLoadResult:
    invalid = SettingsLoadResult(UserSettings(), SettingsLoadStatus.INVALID)
    if not isinstance(payload, dict):
        return invalid
    if "schema_version" not in payload:
        return invalid
    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        return invalid
    if version != SETTINGS_SCHEMA_VERSION:
        return SettingsLoadResult(UserSettings(), SettingsLoadStatus.UNSUPPORTED_VERSION)
    if "camera_id" not in payload or "quality_profile" not in payload:
        return invalid
    camera_id = payload["camera_id"]
    quality_raw = payload["quality_profile"]
    if camera_id is not None and not isinstance(camera_id, str):
        return invalid
    if isinstance(camera_id, str) and not camera_id.strip():
        return invalid
    if not isinstance(quality_raw, str):
        return invalid
    try:
        quality_profile = QualityProfile(quality_raw)
    except ValueError:
        return invalid
    try:
        settings = UserSettings(camera_id=camera_id, quality_profile=quality_profile)
    except (TypeError, ValueError):
        return invalid
    return SettingsLoadResult(settings, SettingsLoadStatus.LOADED)
