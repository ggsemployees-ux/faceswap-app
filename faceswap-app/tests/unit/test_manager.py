"""Unit tests for the model package verification gate."""

import ast
import dataclasses
import hashlib
import json
import pathlib

import pytest

from faceswap.models.manager import (
    BuildMode,
    ModelManager,
    VerifiedModel,
    check_license_policy,
    verify_model_package,
)
from faceswap.models.manifest import ManifestError, load_manifest
from faceswap.runtime.errors import DEFAULT_USER_MESSAGES, ErrorCode, FaceSwapError

_MANAGER_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "faceswap" / "models" / "manager.py"
)
_ALLOWED_MODULES = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "enum",
        "pathlib",
        "faceswap.models.manifest",
        "faceswap.models.integrity",
        "faceswap.runtime.errors",
    }
)
_COMMERCIAL_FAILURES: tuple[tuple[dict[str, object], str], ...] = (
    ({"approval_status": "pending"}, "pending"),
    ({"approval_status": "rejected"}, "rejected"),
    ({"commercial_use_allowed": False}, "commercial_use_allowed"),
    ({"redistribution_allowed": False}, "redistribution_allowed"),
    ({"license_id": "UNKNOWN"}, "UNKNOWN"),
    ({"license_id": "  unknown  "}, "unknown"),
)


def _write_package(
    directory: pathlib.Path,
    model_id: str,
    **overrides: object,
) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = b"model-bytes-" + model_id.encode("utf-8")
    file_name = f"{model_id}.bin"
    model_path = directory / file_name
    model_path.write_bytes(payload)
    data: dict[str, object] = {
        "schema_version": 1,
        "model_id": model_id,
        "model_version": "1.0.0",
        "model_role": "swapper",
        "file_name": file_name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "input_names": ["source"],
        "input_shapes": [[1, 3, 4, 4]],
        "normalization": {"mean": [0.5]},
        "output_contract": {"output": [1]},
        "supported_execution_providers": ["CUDAExecutionProvider"],
        "minimum_vram_mb": 0,
        "license_id": "MIT",
        "redistribution_allowed": True,
        "commercial_use_allowed": True,
        "attribution_text": "",
        "source_url": "",
        "approval_status": "approved",
    }
    data.update(overrides)
    manifest_path = directory / f"{model_id}.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    return manifest_path


def _error(path: pathlib.Path, mode: BuildMode) -> FaceSwapError:
    with pytest.raises(FaceSwapError) as exc_info:
        verify_model_package(path, mode)
    return exc_info.value


def test_g1_build_mode_members() -> None:
    assert [(member.name, member.value) for member in BuildMode] == [
        ("DEVELOPMENT", "development"),
        ("COMMERCIAL", "commercial"),
    ]


def test_g2_development_allows_pending_non_commercial(tmp_path: pathlib.Path) -> None:
    path = _write_package(
        tmp_path,
        "dev-swapper",
        approval_status="pending",
        commercial_use_allowed=False,
        redistribution_allowed=False,
    )

    verified = verify_model_package(path, BuildMode.DEVELOPMENT)

    assert isinstance(verified, VerifiedModel)
    assert verified.manifest.model_id == "dev-swapper"
    assert verified.manifest.approval_status.value == "pending"
    assert verified.path == (tmp_path / "dev-swapper.bin").resolve()


def test_g3_commercial_success(tmp_path: pathlib.Path) -> None:
    path = _write_package(tmp_path, "commercial-swapper")

    verified = verify_model_package(path, BuildMode.COMMERCIAL)

    assert verified.manifest.license_id == "MIT"
    assert verified.manifest.approval_status.value == "approved"
    assert verified.path.name == "commercial-swapper.bin"


@pytest.mark.parametrize(("overrides", "token"), _COMMERCIAL_FAILURES)
def test_g4_commercial_fails_closed(
    overrides: dict[str, object],
    token: str,
    tmp_path: pathlib.Path,
) -> None:
    path = _write_package(tmp_path, "gated", **overrides)
    error = _error(path, BuildMode.COMMERCIAL)

    assert error.error.code is ErrorCode.MODEL_LICENSE_INVALID
    assert "gated" in error.error.detail
    assert token in error.error.detail


def test_g5_multiple_license_violations(tmp_path: pathlib.Path) -> None:
    path = _write_package(
        tmp_path,
        "many",
        approval_status="rejected",
        commercial_use_allowed=False,
        redistribution_allowed=False,
        license_id="UNKNOWN",
    )
    error = _error(path, BuildMode.COMMERCIAL)

    assert error.error.code is ErrorCode.MODEL_LICENSE_INVALID
    detail = error.error.detail
    assert "many" in detail
    assert "rejected" in detail
    assert "commercial_use_allowed" in detail
    assert "redistribution_allowed" in detail
    assert "UNKNOWN" in detail


def test_g6_rejected_fails_in_development(tmp_path: pathlib.Path) -> None:
    path = _write_package(tmp_path, "rejected-dev", approval_status="rejected")
    error = _error(path, BuildMode.DEVELOPMENT)

    assert error.error.code is ErrorCode.MODEL_LICENSE_INVALID
    assert "rejected" in error.error.detail


def test_g7_license_is_checked_before_hashing(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_package(tmp_path, "pending-first", approval_status="pending")

    def fail_if_called(*_args: object, **_kwargs: object) -> pathlib.Path:
        raise AssertionError("verify_model_file was called")

    monkeypatch.setattr("faceswap.models.manager.verify_model_file", fail_if_called)
    error = _error(path, BuildMode.COMMERCIAL)

    assert error.error.code is ErrorCode.MODEL_LICENSE_INVALID


def test_g8_manifest_failures_chain_the_cause(tmp_path: pathlib.Path) -> None:
    missing = _error(tmp_path / "missing.json", BuildMode.DEVELOPMENT)
    assert missing.error.code is ErrorCode.MODEL_CHECKSUM_INVALID
    assert missing.__cause__ is not None
    assert "could not be read" in missing.error.detail

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    invalid_error = _error(invalid, BuildMode.DEVELOPMENT)
    assert invalid_error.error.code is ErrorCode.MODEL_CHECKSUM_INVALID
    assert isinstance(invalid_error.__cause__, ManifestError)
    assert "invalid.json" in invalid_error.error.detail

    incomplete = _write_package(tmp_path, "no-sha")
    payload = json.loads(incomplete.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    del payload["sha256"]
    incomplete.write_text(json.dumps(payload), encoding="utf-8")
    key_error = _error(incomplete, BuildMode.DEVELOPMENT)
    assert key_error.error.code is ErrorCode.MODEL_CHECKSUM_INVALID
    assert isinstance(key_error.__cause__, ManifestError)
    assert "sha256" in key_error.error.detail


def test_g9_model_file_failures(tmp_path: pathlib.Path) -> None:
    missing_manifest = _write_package(tmp_path, "gone")
    (tmp_path / "gone.bin").unlink()
    missing = _error(missing_manifest, BuildMode.DEVELOPMENT)
    assert missing.error.code is ErrorCode.MODEL_CHECKSUM_INVALID
    assert "missing or unreadable" in missing.error.detail
    assert isinstance(missing.__cause__, OSError)

    changed_manifest = _write_package(tmp_path, "changed")
    original = hashlib.sha256(b"model-bytes-changed").hexdigest()
    tampered = b"tampered-bytes"
    (tmp_path / "changed.bin").write_bytes(tampered)
    actual = hashlib.sha256(tampered).hexdigest()
    changed = _error(changed_manifest, BuildMode.DEVELOPMENT)
    assert changed.error.code is ErrorCode.MODEL_CHECKSUM_INVALID
    assert original in changed.error.detail
    assert actual in changed.error.detail
    assert isinstance(changed.__cause__, Exception)


def test_g10_user_message_hides_detail(tmp_path: pathlib.Path) -> None:
    path = _write_package(tmp_path, "hidden", approval_status="pending")
    error = _error(path, BuildMode.COMMERCIAL)
    code = error.error.code

    assert error.error.user_message == DEFAULT_USER_MESSAGES[code]
    assert str(error) == error.error.user_message
    assert error.error.detail not in str(error)


def test_g11_model_manager(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypeError):
        ModelManager("development")  # type: ignore[arg-type]

    manager = ModelManager(BuildMode.DEVELOPMENT)
    assert manager.mode is BuildMode.DEVELOPMENT

    first = _write_package(tmp_path, "first", approval_status="pending")
    second = _write_package(tmp_path, "second", approval_status="pending")
    verified = manager.verify_packages((first, second))
    assert manager.verify_package(first).manifest.model_id == "first"
    assert tuple(model.manifest.model_id for model in verified) == ("first", "second")

    rejected = _write_package(tmp_path, "third", approval_status="rejected")
    with pytest.raises(FaceSwapError) as exc_info:
        manager.verify_packages((first, rejected))
    assert exc_info.value.error.code is ErrorCode.MODEL_LICENSE_INVALID

    duplicate = _write_package(tmp_path / "other", "first", approval_status="pending")
    with pytest.raises(FaceSwapError) as duplicate_error:
        manager.verify_packages((first, duplicate))
    assert duplicate_error.value.error.code is ErrorCode.MODEL_CHECKSUM_INVALID
    assert "duplicate" in duplicate_error.value.error.detail
    assert "first" in duplicate_error.value.error.detail


def test_g12_non_build_mode_is_type_error(tmp_path: pathlib.Path) -> None:
    path = _write_package(tmp_path, "typed")
    manifest = load_manifest(path)

    with pytest.raises(TypeError):
        check_license_policy(manifest, "commercial")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        verify_model_package(path, None)  # type: ignore[arg-type]


def test_g13_verified_model_is_frozen(tmp_path: pathlib.Path) -> None:
    verified = verify_model_package(_write_package(tmp_path, "frozen"), BuildMode.DEVELOPMENT)
    field_name = "path"

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(verified, field_name, verified.path)


def test_g14_import_safety() -> None:
    source = _MANAGER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "InferenceSession" not in source

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in _ALLOWED_MODULES
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "") in _ALLOWED_MODULES

    for statement in tree.body:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and _is_manager_call(node):
                raise AssertionError("module-level ModelManager call")


def _is_manager_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "ModelManager"
    if isinstance(func, ast.Attribute):
        return func.attr == "ModelManager"
    return False
