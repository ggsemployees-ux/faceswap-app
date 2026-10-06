"""Unit tests for the Phase 1 application controller."""

import ast
import dataclasses
import hashlib
import json
import pathlib

import pytest

from faceswap.app.config import QualityProfile, SettingsLoadStatus, UserSettings, save_settings
from faceswap.app.controller import AppController
from faceswap.app.state_machine import AppState, InvalidTransitionError, StateChange, StateMachine
from faceswap.models.manager import BuildMode, ModelManager, VerifiedModel
from faceswap.runtime.device_profile import CheckStatus, EnvironmentReport, SystemInfo
from faceswap.runtime.errors import DEFAULT_USER_MESSAGES, ErrorCode, FaceSwapError, make_error

_CONTROLLER_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "faceswap" / "app" / "controller.py"
)
_ALLOWED_MODULES = frozenset(
    {
        "__future__",
        "collections.abc",
        "pathlib",
        "faceswap.app.config",
        "faceswap.app.state_machine",
        "faceswap.runtime.errors",
        "faceswap.runtime.device_profile",
        "faceswap.models.manager",
    }
)
_FORBIDDEN_PARTS = ("PySide6", "logging", "threading", "metrics", "log_setup", "onnx", "numpy")
_ENVIRONMENT_MESSAGE = (
    "This PC does not meet the minimum requirements (Windows 11 build 22000 or later, "
    "64-bit, Python 3.12). Check the system requirements, then retry."
)
_INTERNAL_MESSAGE = (
    "An unexpected internal error occurred. "
    "Retry, and if the problem continues, export diagnostics."
)


def good_info() -> SystemInfo:
    return SystemInfo("win32", 22631, "AMD64", (3, 12, 10), 32 * 1024**3, 16)


def win10_info() -> SystemInfo:
    return SystemInfo("win32", 19045, "AMD64", (3, 12, 10), 32 * 1024**3, 16)


def write_package(
    directory: pathlib.Path,
    model_id: str,
    **overrides: object,
) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = b"model-bytes-" + model_id.encode("utf-8")
    file_name = f"{model_id}.bin"
    (directory / file_name).write_bytes(payload)
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


def _controller(
    manager: ModelManager,
    *,
    manifest_paths: tuple[pathlib.Path, ...] = (),
    settings_path: pathlib.Path | None = None,
    provider: object | None = None,
    evaluator: object | None = None,
    machine: StateMachine | None = None,
) -> tuple[AppController, list[StateChange]]:
    changes: list[StateChange] = []
    state_machine = StateMachine() if machine is None else machine
    kwargs: dict[str, object] = {
        "model_manager": manager,
        "manifest_paths": manifest_paths,
        "settings_path": settings_path,
        "state_machine": state_machine,
        "system_info_provider": good_info if provider is None else provider,
    }
    if evaluator is not None:
        kwargs["environment_evaluator"] = evaluator
    controller = AppController(**kwargs)  # type: ignore[arg-type]
    controller.state_machine.add_listener(changes.append)
    return controller, changes


def _pairs(changes: list[StateChange]) -> list[tuple[AppState, AppState]]:
    return [(change.previous, change.current) for change in changes]


def test_c1_happy_path(tmp_path: pathlib.Path) -> None:
    first = write_package(tmp_path, "alpha")
    second = write_package(tmp_path, "beta")
    controller, changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        manifest_paths=(first, second),
    )

    assert controller.initialize() is AppState.READY
    assert controller.state is AppState.READY
    assert controller.last_error is None
    assert tuple(model.manifest.model_id for model in controller.models) == ("alpha", "beta")
    assert len(controller.models) == 2
    report = controller.environment_report
    assert report is not None
    assert report.passed is True
    assert controller.can_return_to_ready is True
    assert _pairs(changes) == [
        (AppState.INIT, AppState.ENV_CHECK),
        (AppState.ENV_CHECK, AppState.MODEL_LOADING),
        (AppState.MODEL_LOADING, AppState.READY),
    ]


def test_c2_empty_manifest_list_reaches_ready() -> None:
    controller, _changes = _controller(ModelManager(BuildMode.DEVELOPMENT))

    assert controller.initialize() is AppState.READY
    assert controller.models == ()


def test_c3_windows_10_fails_before_model_verification(tmp_path: pathlib.Path) -> None:
    calls: list[object] = []

    def spy(_paths: object) -> tuple[VerifiedModel, ...]:
        calls.append(_paths)
        raise AssertionError("verify_packages was called")

    manager = ModelManager(BuildMode.DEVELOPMENT)
    manager.verify_packages = spy  # type: ignore[assignment]
    controller, changes = _controller(
        manager,
        manifest_paths=(tmp_path / "missing.json",),
        provider=win10_info,
    )

    assert controller.initialize() is AppState.ERROR
    assert controller.last_error is not None
    assert controller.last_error.code is ErrorCode.ENVIRONMENT_UNSUPPORTED
    assert "windows_build" in controller.last_error.detail
    assert controller.can_return_to_ready is False
    assert calls == []
    assert _pairs(changes) == [
        (AppState.INIT, AppState.ENV_CHECK),
        (AppState.ENV_CHECK, AppState.ERROR),
    ]


def test_c4_cpu_warning_does_not_block() -> None:
    warned = dataclasses.replace(good_info(), logical_cpus=4)
    controller, _changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        provider=lambda: warned,
    )

    assert controller.initialize() is AppState.READY
    report = controller.environment_report
    assert report is not None
    assert report.get("cpu").status is CheckStatus.WARN


def test_c5_commercial_pending_license_fails(tmp_path: pathlib.Path) -> None:
    manifest = write_package(tmp_path, "pending-model", approval_status="pending")
    controller, changes = _controller(
        ModelManager(BuildMode.COMMERCIAL),
        manifest_paths=(manifest,),
    )

    assert controller.initialize() is AppState.ERROR
    assert controller.last_error is not None
    assert controller.last_error.code is ErrorCode.MODEL_LICENSE_INVALID
    assert _pairs(changes)[-1] == (AppState.MODEL_LOADING, AppState.ERROR)


def test_c6_checksum_mismatch_fails(tmp_path: pathlib.Path) -> None:
    manifest = write_package(tmp_path, "swapper")
    (tmp_path / "swapper.bin").write_bytes(b"tampered")
    controller, _changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        manifest_paths=(manifest,),
    )

    assert controller.initialize() is AppState.ERROR
    assert controller.last_error is not None
    assert controller.last_error.code is ErrorCode.MODEL_CHECKSUM_INVALID


def test_c7_unexpected_exceptions_become_internal_error() -> None:
    def raise_runtime() -> SystemInfo:
        raise RuntimeError("boom")

    runtime_controller, _runtime_changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        provider=raise_runtime,
    )
    assert runtime_controller.initialize() is AppState.ERROR
    runtime_error = runtime_controller.last_error
    assert runtime_error is not None
    assert runtime_error.code is ErrorCode.INTERNAL_ERROR
    assert "RuntimeError" in runtime_error.detail
    assert runtime_controller.state is AppState.ERROR

    def raise_value(_info: SystemInfo) -> EnvironmentReport:
        raise ValueError("bad report")

    value_controller, _value_changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        evaluator=raise_value,
    )
    assert value_controller.initialize() is AppState.ERROR
    value_error = value_controller.last_error
    assert value_error is not None
    assert value_error.code is ErrorCode.INTERNAL_ERROR
    assert "ValueError" in value_error.detail
    assert value_controller.state is AppState.ERROR

    manager = ModelManager(BuildMode.DEVELOPMENT)

    def raise_key(_paths: object) -> tuple[VerifiedModel, ...]:
        raise KeyError("missing")

    manager.verify_packages = raise_key  # type: ignore[assignment]
    key_controller, _key_changes = _controller(manager)
    assert key_controller.initialize() is AppState.ERROR
    key_error = key_controller.last_error
    assert key_error is not None
    assert key_error.code is ErrorCode.INTERNAL_ERROR
    assert "KeyError" in key_error.detail
    assert key_controller.state is AppState.ERROR


def test_c8_second_initialize_is_rejected() -> None:
    controller, _changes = _controller(ModelManager(BuildMode.DEVELOPMENT))
    assert controller.initialize() is AppState.READY
    state = controller.state
    models = controller.models

    with pytest.raises(InvalidTransitionError):
        controller.initialize()

    assert controller.state is state
    assert controller.models == models


def test_c9_retry_after_environment_failure(tmp_path: pathlib.Path) -> None:
    manifest = write_package(tmp_path, "retry-model")
    profiles = [win10_info(), good_info()]

    def provider() -> SystemInfo:
        return profiles.pop(0)

    controller, _changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        manifest_paths=(manifest,),
        provider=provider,
    )

    assert controller.initialize() is AppState.ERROR
    assert controller.retry() is AppState.READY
    assert controller.last_error is None
    assert tuple(model.manifest.model_id for model in controller.models) == ("retry-model",)
    with pytest.raises(InvalidTransitionError):
        controller.retry()
    assert controller.state is AppState.READY


def test_c10_return_to_ready_only_after_models_were_verified(tmp_path: pathlib.Path) -> None:
    blocked, _blocked_changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        provider=win10_info,
    )
    assert blocked.initialize() is AppState.ERROR
    with pytest.raises(InvalidTransitionError):
        blocked.return_to_ready()
    assert blocked.state is AppState.ERROR

    manifest = write_package(tmp_path, "ready-model")
    controller, _changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        manifest_paths=(manifest,),
    )
    assert controller.initialize() is AppState.READY
    models = controller.models
    controller.report_failure(make_error(ErrorCode.CAMERA_UNAVAILABLE))
    assert controller.state is AppState.ERROR
    assert controller.can_return_to_ready is True
    assert controller.return_to_ready() is AppState.READY
    assert controller.models == models
    assert controller.last_error is None


def test_c11_second_failure_report_keeps_the_first_error() -> None:
    controller, _changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        provider=win10_info,
    )
    assert controller.initialize() is AppState.ERROR
    first = controller.last_error
    assert first is not None
    controller.report_failure(make_error(ErrorCode.CAMERA_UNAVAILABLE, detail="later"))
    assert controller.last_error == first
    with pytest.raises(TypeError):
        controller.report_failure("not an app error")  # type: ignore[arg-type]
    assert controller.last_error == first


def test_c12_settings_load_and_retry_does_not_reload(tmp_path: pathlib.Path) -> None:
    saved = UserSettings(camera_id="cam1", quality_profile=QualityProfile.QUALITY)
    settings_path = tmp_path / "settings.json"
    save_settings(saved, settings_path)
    loaded, _loaded_changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        settings_path=settings_path,
    )
    assert loaded.initialize() is AppState.READY
    assert loaded.user_settings == saved
    assert loaded.settings_status is SettingsLoadStatus.LOADED

    missing, _missing_changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        settings_path=tmp_path / "absent.json",
    )
    assert missing.initialize() is AppState.READY
    assert missing.user_settings == UserSettings()
    assert missing.settings_status is SettingsLoadStatus.MISSING

    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_bytes(b"\xff not-json")
    corrupt, _corrupt_changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        settings_path=corrupt_path,
    )
    assert corrupt.initialize() is AppState.READY
    assert corrupt.user_settings == UserSettings()
    assert corrupt.settings_status is SettingsLoadStatus.INVALID

    omitted, _omitted_changes = _controller(ModelManager(BuildMode.DEVELOPMENT))
    assert omitted.settings_status is None
    assert omitted.initialize() is AppState.READY
    assert omitted.user_settings == UserSettings()
    assert omitted.settings_status is None

    profiles = [win10_info(), good_info()]

    def provider() -> SystemInfo:
        return profiles.pop(0)

    retrying, _retry_changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        settings_path=settings_path,
        provider=provider,
    )
    assert retrying.initialize() is AppState.ERROR
    assert retrying.user_settings == saved
    save_settings(
        UserSettings(camera_id="cam2", quality_profile=QualityProfile.QUALITY),
        settings_path,
    )
    assert retrying.retry() is AppState.READY
    assert retrying.user_settings == saved
    assert retrying.settings_status is SettingsLoadStatus.LOADED


def test_c13_identical_inputs_produce_identical_transitions(tmp_path: pathlib.Path) -> None:
    manifest = write_package(tmp_path, "same")
    first, first_changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        manifest_paths=(manifest,),
    )
    second, second_changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        manifest_paths=(manifest,),
    )

    assert first.initialize() is second.initialize()
    assert first_changes == second_changes
    assert first.state is second.state


def test_c14_new_error_codes_hide_detail_from_the_user_message() -> None:
    assert len(tuple(ErrorCode)) == 11
    assert ErrorCode.ENVIRONMENT_UNSUPPORTED in ErrorCode
    assert ErrorCode.INTERNAL_ERROR in ErrorCode
    assert DEFAULT_USER_MESSAGES[ErrorCode.ENVIRONMENT_UNSUPPORTED] == _ENVIRONMENT_MESSAGE
    assert DEFAULT_USER_MESSAGES[ErrorCode.INTERNAL_ERROR] == _INTERNAL_MESSAGE

    controller, _changes = _controller(
        ModelManager(BuildMode.DEVELOPMENT),
        provider=win10_info,
    )
    assert controller.initialize() is AppState.ERROR
    error = controller.last_error
    assert error is not None
    visible = FaceSwapError(error)
    assert str(visible) == DEFAULT_USER_MESSAGES[error.code]
    assert error.detail not in str(visible)


def test_c15_import_safety() -> None:
    source = _CONTROLLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _assert_allowed(alias.name)
        elif isinstance(node, ast.ImportFrom):
            _assert_allowed(node.module or "")

    for statement in tree.body:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and _is_controller_call(node):
                raise AssertionError("module-level AppController call")


def _assert_allowed(module: str) -> None:
    assert module in _ALLOWED_MODULES
    assert not any(part in module for part in _FORBIDDEN_PARTS)


def _is_controller_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "AppController"
    if isinstance(func, ast.Attribute):
        return func.attr == "AppController"
    return False
