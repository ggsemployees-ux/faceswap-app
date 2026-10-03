"""Immutable session configuration snapshot per spec §22.1."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


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
