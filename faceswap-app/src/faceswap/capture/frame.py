"""Live frame contract per spec §10.3 and §10.2 item 14.

A capture thread attaches a monotonic timestamp and a sequence number to each frame.
gpu_handle from §10.3 is deferred until the GPU stack exists (inference steps).
FramePacket does not copy cpu_buffer. After validation it marks that buffer read-only,
transferring ownership so later consumers cannot mutate the pixels.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass

import numpy
from numpy.typing import NDArray


class PixelFormat(enum.StrEnum):
    """Pixel layout of a captured CPU buffer."""

    BGR24 = "bgr24"


def monotonic_timestamp_ns() -> int:
    """Return a monotonic capture timestamp in nanoseconds."""
    return time.monotonic_ns()


@dataclass(frozen=True, slots=True, eq=False)
class FramePacket:
    """Immutable live frame. Equality and hashing use object identity.

    Construction takes ownership of cpu_buffer without copying it: the buffer is
    made read-only so preview and the virtual camera can share it safely.
    """

    sequence_id: int
    capture_timestamp_ns: int
    width: int
    height: int
    pixel_format: PixelFormat
    cpu_buffer: NDArray[numpy.uint8]
    camera_id: str

    def __post_init__(self) -> None:
        _require_non_negative_int(self.sequence_id, "sequence_id")
        _require_non_negative_int(self.capture_timestamp_ns, "capture_timestamp_ns")
        _require_positive_int(self.width, "width")
        _require_positive_int(self.height, "height")
        if not isinstance(self.pixel_format, PixelFormat):
            raise TypeError("pixel_format must be a PixelFormat")
        if not isinstance(self.camera_id, str):
            raise TypeError("camera_id must be a str")
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty or whitespace")
        _validate_bgr24_buffer(self.cpu_buffer, self.height, self.width)
        self.cpu_buffer.flags.writeable = False

    def age_ms(self, now_ns: int) -> float:
        """Return milliseconds from capture_timestamp_ns to now_ns."""
        if isinstance(now_ns, bool) or not isinstance(now_ns, int):
            raise TypeError("now_ns must be an int")
        if now_ns < self.capture_timestamp_ns:
            raise ValueError("now_ns must be >= capture_timestamp_ns")
        return (now_ns - self.capture_timestamp_ns) / 1_000_000


def _require_non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_bgr24_buffer(buffer: object, height: int, width: int) -> None:
    if not isinstance(buffer, numpy.ndarray):
        raise TypeError("cpu_buffer must be a numpy.ndarray")
    if buffer.dtype != numpy.uint8:
        raise TypeError("cpu_buffer dtype must be uint8")
    if buffer.ndim != 3 or buffer.shape != (height, width, 3):
        raise ValueError("cpu_buffer shape must be (height, width, 3)")
    if not buffer.flags.c_contiguous:
        raise ValueError("cpu_buffer must be C-contiguous")
