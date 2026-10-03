"""Unit tests for persisted user settings."""

import ast
import dataclasses
import json
import pathlib

import pytest

from faceswap.app.config import (
    APP_DIR_NAME,
    SETTINGS_FILE_NAME,
    QualityProfile,
    SettingsLoadStatus,
    UserSettings,
    default_settings_path,
    load_settings,
    save_settings,
)

_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "faceswap" / "app" / "config.py"
)
_ALLOWED_MODULES = frozenset(
    {
        "__future__",
        "dataclasses",
        "enum",
        "json",
        "os",
        "pathlib",
        "tempfile",
        "collections.abc",
    }
)
_FORBIDDEN_CALLS = frozenset({"load_settings", "save_settings", "default_settings_path", "open"})
_LOCAL_APP_DATA = pathlib.Path(r"C:\Users\test\AppData\Local")


def test_user_settings_defaults() -> None:
    settings = UserSettings()

    assert settings.camera_id is None
    assert settings.quality_profile is QualityProfile.AUTO


@pytest.mark.parametrize("field_name", ["camera_id", "quality_profile"])
def test_user_settings_is_frozen(field_name: str) -> None:
    settings = UserSettings()

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(settings, field_name, getattr(settings, field_name))


@pytest.mark.parametrize("camera_id", ["", "   "])
def test_blank_camera_id_is_rejected(camera_id: str) -> None:
    with pytest.raises(ValueError, match="camera_id"):
        UserSettings(camera_id=camera_id)


def test_non_string_camera_id_is_rejected() -> None:
    with pytest.raises(TypeError, match="camera_id"):
        UserSettings(camera_id=123)  # type: ignore[arg-type]


def test_plain_string_quality_profile_is_rejected() -> None:
    with pytest.raises(TypeError, match="quality_profile"):
        UserSettings(quality_profile="auto")  # type: ignore[arg-type]


def test_default_settings_path_uses_the_given_env() -> None:
    path = default_settings_path({"LOCALAPPDATA": str(_LOCAL_APP_DATA)})

    assert path == _LOCAL_APP_DATA / "faceswap" / "settings.json"
    assert path == _LOCAL_APP_DATA / APP_DIR_NAME / SETTINGS_FILE_NAME


@pytest.mark.parametrize("env", [{}, {"LOCALAPPDATA": "   "}])
def test_default_settings_path_rejects_a_blank_location(env: dict[str, str]) -> None:
    with pytest.raises(RuntimeError, match="LOCALAPPDATA"):
        default_settings_path(env)


def test_default_settings_path_creates_nothing(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "not-created"

    result = default_settings_path({"LOCALAPPDATA": str(root)})

    assert result == root / "faceswap" / "settings.json"
    assert not root.exists()


@pytest.mark.parametrize(
    "settings",
    [
        UserSettings(camera_id="cam0", quality_profile=QualityProfile.QUALITY),
        UserSettings(camera_id=None, quality_profile=QualityProfile.AUTO),
    ],
)
def test_save_and_load_round_trip(tmp_path: pathlib.Path, settings: UserSettings) -> None:
    path = tmp_path / "settings.json"

    save_settings(settings, path)
    loaded = load_settings(path)

    assert loaded.status is SettingsLoadStatus.LOADED
    assert loaded.settings == settings


def test_save_creates_parents_and_exact_json(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "missing" / "nested" / "settings.json"
    settings = UserSettings(camera_id="cam0", quality_profile=QualityProfile.PERFORMANCE)

    save_settings(settings, path)

    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    payload = json.loads(text)
    assert set(payload) == {"schema_version", "camera_id", "quality_profile"}
    assert payload["schema_version"] == 1
    assert payload["camera_id"] == "cam0"
    assert payload["quality_profile"] == "performance"
    assert isinstance(payload["quality_profile"], str)


def test_missing_file_returns_defaults(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "settings.json"

    loaded = load_settings(path)

    assert loaded.status is SettingsLoadStatus.MISSING
    assert loaded.settings == UserSettings()
    assert not path.exists()


def test_corrupt_json_is_invalid_and_unchanged(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "settings.json"
    original = b"{not json"
    path.write_bytes(original)

    loaded = load_settings(path)

    assert loaded.status is SettingsLoadStatus.INVALID
    assert loaded.settings == UserSettings()
    assert path.read_bytes() == original


def test_invalid_utf8_is_invalid_and_unchanged(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "settings.json"
    original = b"\xff\xfe{"
    path.write_bytes(original)

    loaded = load_settings(path)

    assert loaded.status is SettingsLoadStatus.INVALID
    assert loaded.settings == UserSettings()
    assert path.read_bytes() == original


@pytest.mark.parametrize("payload", [[], "x", 1])
def test_non_object_json_is_invalid(tmp_path: pathlib.Path, payload: object) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_settings(path)

    assert loaded.status is SettingsLoadStatus.INVALID
    assert loaded.settings == UserSettings()


@pytest.mark.parametrize(
    "payload",
    [
        {"camera_id": None, "quality_profile": "auto"},
        {"schema_version": "1", "camera_id": None, "quality_profile": "auto"},
        {"schema_version": 1.0, "camera_id": None, "quality_profile": "auto"},
        {"schema_version": True, "camera_id": None, "quality_profile": "auto"},
    ],
)
def test_bad_schema_version_is_invalid(tmp_path: pathlib.Path, payload: dict[str, object]) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_settings(path)

    assert loaded.status is SettingsLoadStatus.INVALID
    assert loaded.settings == UserSettings()


@pytest.mark.parametrize("version", [0, 2])
def test_unsupported_schema_version_is_unchanged(tmp_path: pathlib.Path, version: int) -> None:
    path = tmp_path / "settings.json"
    original = json.dumps(
        {"schema_version": version, "camera_id": "cam0", "quality_profile": "auto"}
    )
    path.write_text(original, encoding="utf-8")

    loaded = load_settings(path)

    assert loaded.status is SettingsLoadStatus.UNSUPPORTED_VERSION
    assert loaded.settings == UserSettings()
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 1, "camera_id": "cam0", "quality_profile": "turbo"},
        {"schema_version": 1, "camera_id": "cam0"},
        {"schema_version": 1, "camera_id": 123, "quality_profile": "auto"},
        {"schema_version": 1, "camera_id": "   ", "quality_profile": "auto"},
        {"schema_version": 1, "quality_profile": "auto"},
    ],
)
def test_bad_version_one_fields_are_invalid(
    tmp_path: pathlib.Path, payload: dict[str, object]
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_settings(path)

    assert loaded.status is SettingsLoadStatus.INVALID
    assert loaded.settings == UserSettings()


def test_unknown_keys_are_ignored(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_id": "cam0",
                "quality_profile": "quality",
                "theme": "dark",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_settings(path)

    assert loaded.status is SettingsLoadStatus.LOADED
    assert loaded.settings == UserSettings(camera_id="cam0", quality_profile=QualityProfile.QUALITY)


def test_save_replaces_existing_file_without_leftovers(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "settings.json"
    save_settings(UserSettings(camera_id="old", quality_profile=QualityProfile.AUTO), path)

    save_settings(UserSettings(camera_id="new", quality_profile=QualityProfile.QUALITY), path)

    loaded = load_settings(path)
    assert loaded.status is SettingsLoadStatus.LOADED
    assert loaded.settings.camera_id == "new"
    assert loaded.settings.quality_profile is QualityProfile.QUALITY
    assert [item.name for item in tmp_path.iterdir()] == ["settings.json"]


def test_failed_replace_keeps_the_existing_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.json"
    save_settings(UserSettings(camera_id="cam0", quality_profile=QualityProfile.AUTO), path)
    original = path.read_bytes()

    def fail_replace(source: object, target: object) -> None:
        raise OSError(f"replace failed for {source} -> {target}")

    monkeypatch.setattr("faceswap.app.config.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        save_settings(UserSettings(camera_id="other", quality_profile=QualityProfile.QUALITY), path)

    assert path.read_bytes() == original
    assert [item.name for item in tmp_path.iterdir()] == ["settings.json"]


def test_settings_module_import_safety() -> None:
    tree = ast.parse(_CONFIG_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in _ALLOWED_MODULES
                assert not alias.name.startswith("faceswap")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module in _ALLOWED_MODULES
            assert not module.startswith("faceswap")

    for statement in tree.body:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and _is_forbidden_call(node):
                raise AssertionError("module-level call is not allowed")
            if isinstance(node, ast.Attribute) and node.attr == "environ":
                raise AssertionError("module-level os.environ access")


def _is_forbidden_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _FORBIDDEN_CALLS
    return isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_CALLS
