"""Structured application errors (machine code + user-safe message) per spec §23, §18 and FR-017.

The detail field is for internal diagnostics only. It is never shown to users and must
not contain image data, face crops, or embeddings (spec §19).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


class ErrorCode(enum.StrEnum):
    """Machine-readable failure codes from spec §18."""

    CAMERA_UNAVAILABLE = "camera_unavailable"
    CAPTURE_FORMAT_UNSUPPORTED = "capture_format_unsupported"
    CUDA_PROVIDER_UNAVAILABLE = "cuda_provider_unavailable"
    MODEL_CHECKSUM_INVALID = "model_checksum_invalid"
    MODEL_LICENSE_INVALID = "model_license_invalid"
    GPU_OUT_OF_MEMORY = "gpu_out_of_memory"
    VIRTUAL_CAMERA_PERMISSION_DENIED = "virtual_camera_permission_denied"
    VIRTUAL_CAMERA_CONSUMER_DISCONNECTED = "virtual_camera_consumer_disconnected"
    NATIVE_COMPONENT_FAILURE = "native_component_failure"
    ENVIRONMENT_UNSUPPORTED = "environment_unsupported"
    INTERNAL_ERROR = "internal_error"


DEFAULT_USER_MESSAGES: Mapping[ErrorCode, str] = MappingProxyType(
    {
        ErrorCode.CAMERA_UNAVAILABLE: (
            "The selected camera is unavailable or in use by another application. "
            "Close other apps using the camera, then rescan or retry."
        ),
        ErrorCode.CAPTURE_FORMAT_UNSUPPORTED: (
            "The selected camera does not support 1920x1080 at 30 FPS. "
            "Choose a different camera or format."
        ),
        ErrorCode.CUDA_PROVIDER_UNAVAILABLE: (
            "NVIDIA CUDA acceleration is unavailable. "
            "Check that a supported NVIDIA GPU and driver are installed."
        ),
        ErrorCode.MODEL_CHECKSUM_INVALID: (
            "A model package failed its integrity check and was not loaded. "
            "Reinstall or replace the model package."
        ),
        ErrorCode.MODEL_LICENSE_INVALID: (
            "A model package is missing approved license information and was not loaded."
        ),
        ErrorCode.GPU_OUT_OF_MEMORY: (
            "The GPU ran out of memory. "
            "Switch to the Performance profile or a smaller model, then start again."
        ),
        ErrorCode.VIRTUAL_CAMERA_PERMISSION_DENIED: (
            "Windows blocked access to the virtual camera. "
            "Check camera privacy settings in Windows Settings."
        ),
        ErrorCode.VIRTUAL_CAMERA_CONSUMER_DISCONNECTED: (
            "The application using the virtual camera disconnected. "
            "Reconnect from that application; no restart is needed."
        ),
        ErrorCode.NATIVE_COMPONENT_FAILURE: (
            "An internal virtual camera component failed. "
            "Stop and start the virtual camera; if it persists, export diagnostics."
        ),
        ErrorCode.ENVIRONMENT_UNSUPPORTED: (
            "This PC does not meet the minimum requirements (Windows 11 build 22000 or later, "
            "64-bit, Python 3.12). Check the system requirements, then retry."
        ),
        ErrorCode.INTERNAL_ERROR: (
            "An unexpected internal error occurred. "
            "Retry, and if the problem continues, export diagnostics."
        ),
    }
)


@dataclass(frozen=True, slots=True)
class AppError:
    """Immutable error value with a machine code, user-safe message, and internal detail."""

    code: ErrorCode
    user_message: str
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.code, ErrorCode):
            raise TypeError("code must be an ErrorCode")
        if not self.user_message.strip():
            raise ValueError("user_message must not be empty or whitespace")
        if not isinstance(self.detail, str):
            raise TypeError("detail must be a str")


def make_error(code: ErrorCode, detail: str = "") -> AppError:
    """Build an AppError using the default user message for code."""
    return AppError(code=code, user_message=DEFAULT_USER_MESSAGES[code], detail=detail)


class FaceSwapError(Exception):
    """Boundary exception that carries an AppError and shows only its user message."""

    error: AppError

    def __init__(self, error: AppError) -> None:
        if not isinstance(error, AppError):
            raise TypeError("error must be an AppError")
        super().__init__(error.user_message)
        self.error = error

    def __str__(self) -> str:
        return self.error.user_message
