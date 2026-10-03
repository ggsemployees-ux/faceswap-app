"""Unit tests for the structured application error model."""

import ast
import dataclasses
import pathlib
import re

import pytest

from faceswap.runtime.errors import (
    DEFAULT_USER_MESSAGES,
    AppError,
    ErrorCode,
    FaceSwapError,
    make_error,
)

_ERRORS_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "faceswap" / "runtime" / "errors.py"
)
_ALLOWED_IMPORTS = frozenset({"__future__", "dataclasses", "enum", "types", "collections.abc"})
_CODE_VALUE = re.compile(r"^[a-z]+(_[a-z]+)*$")
_SECRET = "SECRET-DETAIL-123"
_EXPECTED_MESSAGES = {
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
}
_APP_ERROR_FIELDS = ("code", "user_message", "detail")


def test_error_code_members_are_exact() -> None:
    assert tuple(ErrorCode.__members__) == (
        "CAMERA_UNAVAILABLE",
        "CAPTURE_FORMAT_UNSUPPORTED",
        "CUDA_PROVIDER_UNAVAILABLE",
        "MODEL_CHECKSUM_INVALID",
        "MODEL_LICENSE_INVALID",
        "GPU_OUT_OF_MEMORY",
        "VIRTUAL_CAMERA_PERMISSION_DENIED",
        "VIRTUAL_CAMERA_CONSUMER_DISCONNECTED",
        "NATIVE_COMPONENT_FAILURE",
    )
    values = [member.value for member in ErrorCode]
    assert len(values) == len(set(values)) == 9
    assert all(_CODE_VALUE.fullmatch(value) for value in values)
    assert {member.name: member.value for member in ErrorCode} == {
        code.name: code.value for code in _EXPECTED_MESSAGES
    }


def test_default_user_messages_match_the_approved_text() -> None:
    assert set(DEFAULT_USER_MESSAGES) == set(ErrorCode)
    assert dict(DEFAULT_USER_MESSAGES) == _EXPECTED_MESSAGES
    assert all(message.strip() for message in DEFAULT_USER_MESSAGES.values())


def test_default_user_messages_are_read_only() -> None:
    with pytest.raises(TypeError):
        DEFAULT_USER_MESSAGES[ErrorCode.CAMERA_UNAVAILABLE] = "changed"  # type: ignore[index]


@pytest.mark.parametrize("code", list(ErrorCode))
def test_make_error_uses_the_default_message(code: ErrorCode) -> None:
    error = make_error(code)
    detailed = make_error(code, detail="x")

    assert error == AppError(code=code, user_message=DEFAULT_USER_MESSAGES[code], detail="")
    assert detailed.code is code
    assert detailed.user_message == DEFAULT_USER_MESSAGES[code]
    assert detailed.detail == "x"


@pytest.mark.parametrize("field_name", _APP_ERROR_FIELDS)
def test_app_error_is_frozen(field_name: str) -> None:
    error = make_error(ErrorCode.CAMERA_UNAVAILABLE)

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(error, field_name, getattr(error, field_name))


@pytest.mark.parametrize("user_message", ["", "   "])
def test_blank_user_message_is_rejected(user_message: str) -> None:
    with pytest.raises(ValueError, match="user_message"):
        AppError(code=ErrorCode.CAMERA_UNAVAILABLE, user_message=user_message)


def test_code_must_be_an_error_code() -> None:
    with pytest.raises(TypeError, match="code"):
        AppError(code="camera_unavailable", user_message="Shown to the user.")  # type: ignore[arg-type]


def test_detail_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="detail"):
        AppError(
            code=ErrorCode.CAMERA_UNAVAILABLE,
            user_message="Shown to the user.",
            detail=123,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("code", list(ErrorCode))
def test_detail_is_absent_from_user_visible_text(code: ErrorCode) -> None:
    error = make_error(code, detail=_SECRET)
    raised = FaceSwapError(error)

    assert _SECRET not in error.user_message
    assert _SECRET not in str(raised)
    assert all(_SECRET not in str(arg) for arg in raised.args)


def test_face_swap_error_carries_the_app_error() -> None:
    error = make_error(ErrorCode.GPU_OUT_OF_MEMORY)

    assert issubclass(FaceSwapError, Exception)
    with pytest.raises(FaceSwapError) as caught:
        raise FaceSwapError(error)

    assert caught.value.error is error
    assert str(caught.value) == error.user_message


def test_face_swap_error_rejects_a_non_app_error() -> None:
    with pytest.raises(TypeError, match="error"):
        FaceSwapError("not an app error")  # type: ignore[arg-type]


def test_equal_app_errors_compare_equal_and_share_a_hash() -> None:
    first = AppError(
        code=ErrorCode.MODEL_CHECKSUM_INVALID,
        user_message="A model package failed its integrity check and was not loaded.",
        detail="checksum",
    )
    second = AppError(
        code=ErrorCode.MODEL_CHECKSUM_INVALID,
        user_message="A model package failed its integrity check and was not loaded.",
        detail="checksum",
    )

    assert first == second
    assert hash(first) == hash(second)


def test_errors_module_import_safety() -> None:
    tree = ast.parse(_ERRORS_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in _ALLOWED_IMPORTS
                assert not alias.name.startswith("faceswap")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module in _ALLOWED_IMPORTS
            assert not module.startswith("faceswap")
