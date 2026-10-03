"""Unit tests for the immutable session configuration snapshot."""

import ast
import dataclasses
import pathlib

import pytest

from faceswap.app.config import DetectionPolicy, QualityProfile, SessionConfig

_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "faceswap" / "app" / "config.py"
)
_ALLOWED_IMPORT_ROOTS = {"__future__", "dataclasses", "enum"}
_SESSION_FIELDS = (
    "camera_id",
    "width",
    "height",
    "fps",
    "provider",
    "model_ids",
    "detection_policy",
    "enhancer_enabled",
    "quality_profile",
)


def test_session_config_defaults() -> None:
    config = SessionConfig(camera_id="cam0")

    assert config.camera_id == "cam0"
    assert config.width == 1920
    assert config.height == 1080
    assert config.fps == 30
    assert config.provider == "CUDAExecutionProvider"
    assert config.model_ids == ()
    assert config.detection_policy == DetectionPolicy(verification_interval_frames=5)
    assert config.enhancer_enabled is False
    assert config.quality_profile is QualityProfile.AUTO


def test_camera_id_is_required() -> None:
    with pytest.raises(TypeError):
        SessionConfig()  # type: ignore[call-arg]


@pytest.mark.parametrize("field_name", _SESSION_FIELDS)
def test_session_config_fields_are_frozen(field_name: str) -> None:
    config = SessionConfig(camera_id="cam0")

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, field_name, getattr(config, field_name))


def test_detection_policy_is_frozen() -> None:
    policy = DetectionPolicy()
    field_name = "verification_interval_frames"

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(policy, field_name, 1)


@pytest.mark.parametrize("field_name", ["width", "height", "fps"])
@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_dimensions_are_rejected(field_name: str, value: int) -> None:
    with pytest.raises(ValueError, match=field_name):
        if field_name == "width":
            SessionConfig(camera_id="cam0", width=value)
        elif field_name == "height":
            SessionConfig(camera_id="cam0", height=value)
        else:
            SessionConfig(camera_id="cam0", fps=value)


@pytest.mark.parametrize("camera_id", ["", "   "])
def test_blank_camera_id_is_rejected(camera_id: str) -> None:
    with pytest.raises(ValueError, match="camera_id"):
        SessionConfig(camera_id=camera_id)


@pytest.mark.parametrize("provider", ["", "   "])
def test_blank_provider_is_rejected(provider: str) -> None:
    with pytest.raises(ValueError, match="provider"):
        SessionConfig(camera_id="cam0", provider=provider)


def test_model_ids_must_be_a_tuple() -> None:
    model_ids = ["detector", "swapper"]

    with pytest.raises(TypeError, match="model_ids"):
        SessionConfig(camera_id="cam0", model_ids=model_ids)  # type: ignore[arg-type]

    config = SessionConfig(camera_id="cam0", model_ids=("detector", "swapper"))
    assert config.model_ids == ("detector", "swapper")


@pytest.mark.parametrize("interval", [0, -1])
def test_detection_policy_rejects_non_positive_interval(interval: int) -> None:
    with pytest.raises(ValueError, match="verification_interval_frames"):
        DetectionPolicy(verification_interval_frames=interval)


def test_detection_policy_accepts_interval_of_one() -> None:
    policy = DetectionPolicy(verification_interval_frames=1)

    assert policy.verification_interval_frames == 1


def test_replace_updates_copy_and_revalidates() -> None:
    config = SessionConfig(camera_id="cam0")

    updated = dataclasses.replace(config, fps=60)

    assert updated is not config
    assert updated.fps == 60
    assert config.fps == 30
    with pytest.raises(ValueError, match="fps"):
        dataclasses.replace(config, fps=0)


def test_equal_configs_compare_equal_and_share_a_hash() -> None:
    first = SessionConfig(camera_id="cam0", model_ids=("detector",))
    second = SessionConfig(camera_id="cam0", model_ids=("detector",))

    assert first == second
    assert hash(first) == hash(second)
    assert {first: "session"}[second] == "session"


def test_quality_profile_members() -> None:
    assert tuple(QualityProfile.__members__) == ("AUTO", "PERFORMANCE", "QUALITY")
    assert QualityProfile.AUTO.value == "auto"
    assert QualityProfile.PERFORMANCE.value == "performance"
    assert QualityProfile.QUALITY.value == "quality"


def test_config_module_import_safety() -> None:
    tree = ast.parse(_CONFIG_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                assert root in _ALLOWED_IMPORT_ROOTS
                assert not alias.name.startswith("faceswap")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", maxsplit=1)[0]
            assert root in _ALLOWED_IMPORT_ROOTS
            assert not module.startswith("faceswap")

    for statement in tree.body:
        value = _assignment_value(statement)
        if value is not None and _calls_session_config(value):
            raise AssertionError("module-level assignment calls SessionConfig")


def _assignment_value(statement: ast.stmt) -> ast.expr | None:
    if isinstance(statement, ast.Assign):
        return statement.value
    if isinstance(statement, ast.AnnAssign):
        return statement.value
    if isinstance(statement, ast.AugAssign):
        return statement.value
    return None


def _calls_session_config(expression: ast.expr) -> bool:
    for node in ast.walk(expression):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "SessionConfig":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "SessionConfig":
            return True
    return False
