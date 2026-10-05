"""Unit tests for model manifest parsing."""

import ast
import dataclasses
import json
import pathlib

import pytest

from faceswap.models.manifest import (
    MANIFEST_SCHEMA_VERSION,
    REQUIRED_KEYS,
    ApprovalStatus,
    ManifestError,
    ModelManifest,
    ModelRole,
    load_manifest,
    parse_manifest,
)

_MANIFEST_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "faceswap" / "models" / "manifest.py"
)
_ALLOWED_MODULES = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "enum",
        "json",
        "pathlib",
        "re",
        "types",
    }
)
_FORBIDDEN_MODULES = frozenset({"faceswap", "onnx", "onnxruntime", "hashlib", "logging"})
_KEYS = (
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
)
_REJECTED_FILE_NAMES = (
    "../model.onnx",
    "dir/model.onnx",
    "dir\\model.onnx",
    "C:model.onnx",
    ".",
    "..",
    "",
    "   ",
)


def _valid() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_id": "test-swapper",
        "model_version": "1.0.0",
        "model_role": "swapper",
        "file_name": "model.onnx",
        "sha256": "a" * 64,
        "input_names": ["source", "target"],
        "input_shapes": [["batch", 512], [1, 3, 128, 128]],
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "output_contract": {"output": [1, 3, 128, 128]},
        "supported_execution_providers": ["CUDAExecutionProvider"],
        "minimum_vram_mb": 6144,
        "license_id": "test-license",
        "redistribution_allowed": True,
        "commercial_use_allowed": False,
        "attribution_text": "Example attribution",
        "source_url": "https://example.invalid/model",
        "approval_status": "pending",
    }


def _error(data: dict[str, object]) -> ManifestError:
    with pytest.raises(ManifestError) as exc_info:
        parse_manifest(data)
    return exc_info.value


def test_mf1_constants_and_enums() -> None:
    assert MANIFEST_SCHEMA_VERSION == 1
    assert REQUIRED_KEYS == set(_KEYS)
    assert len(_KEYS) == 18
    assert [(member.name, member.value) for member in ModelRole] == [
        ("DETECTOR", "detector"),
        ("LANDMARKS", "landmarks"),
        ("EMBEDDER", "embedder"),
        ("SWAPPER", "swapper"),
        ("ENHANCER", "enhancer"),
    ]
    assert [(member.name, member.value) for member in ApprovalStatus] == [
        ("APPROVED", "approved"),
        ("PENDING", "pending"),
        ("REJECTED", "rejected"),
    ]


def test_mf2_valid_manifest_parses() -> None:
    manifest = parse_manifest(_valid())

    assert manifest.model_id == "test-swapper"
    assert manifest.model_role is ModelRole.SWAPPER
    assert manifest.approval_status is ApprovalStatus.PENDING
    assert manifest.input_names == ("source", "target")
    assert manifest.input_shapes == (("batch", 512), (1, 3, 128, 128))
    assert isinstance(manifest.input_shapes[0][0], str)
    assert manifest.supported_execution_providers == ("CUDAExecutionProvider",)
    assert manifest.sha256 == "a" * 64
    assert manifest.minimum_vram_mb == 6144


def test_mf3_objects_are_deeply_read_only() -> None:
    data = _valid()
    data["normalization"] = {"color": {"space": "rgb"}, "mean": [0.5, 0.5, 0.5]}
    manifest = parse_manifest(data)
    normalization = manifest.normalization
    output = manifest.output_contract["output"]

    assert isinstance(output, tuple)
    assert output == (1, 3, 128, 128)
    mean = normalization["mean"]
    assert isinstance(mean, tuple)
    with pytest.raises(TypeError):
        normalization["mean"] = (0.0, 0.0, 0.0)  # type: ignore[index]
    with pytest.raises(TypeError):
        mean[0] = 0.0  # type: ignore[index]
    nested = normalization["color"]
    with pytest.raises(TypeError):
        nested["space"] = "bgr"  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.output_contract["output"] = (1,)  # type: ignore[index]


@pytest.mark.parametrize("key", _KEYS)
def test_mf4_missing_key(key: str) -> None:
    data = _valid()
    del data[key]
    error = _error(data)

    assert error.field == key
    assert key in str(error)


def test_mf5_extra_key() -> None:
    data = _valid()
    data["sha265"] = "typo"
    error = _error(data)

    assert error.field == "sha265"
    assert "sha265" in str(error)


@pytest.mark.parametrize("value", [2, "1", True])
def test_mf6_schema_version(value: object) -> None:
    data = _valid()
    data["schema_version"] = value
    error = _error(data)

    assert error.field == "schema_version"
    assert "schema_version" in str(error)


@pytest.mark.parametrize("field", ["model_id", "model_version", "license_id"])
@pytest.mark.parametrize("value", ["", "   ", 5])
def test_mf7_blank_or_non_string_identifiers(field: str, value: object) -> None:
    data = _valid()
    data[field] = value
    error = _error(data)

    assert error.field == field


def test_mf8_file_name_accepts_plain_name() -> None:
    data = _valid()
    data["file_name"] = "model.onnx"

    assert parse_manifest(data).file_name == "model.onnx"


@pytest.mark.parametrize("name", _REJECTED_FILE_NAMES)
def test_mf8_file_name_rejects_paths(name: str) -> None:
    data = _valid()
    data["file_name"] = name

    assert _error(data).field == "file_name"


def test_mf9_sha256_lowercase_and_uppercase() -> None:
    lower = _valid()
    upper = _valid()
    upper["sha256"] = "A" * 64

    assert parse_manifest(lower).sha256 == "a" * 64
    assert parse_manifest(upper).sha256 == "a" * 64


@pytest.mark.parametrize("value", ["a" * 63, "a" * 65, "g" + "a" * 63, None])
def test_mf9_sha256_rejected(value: object) -> None:
    data = _valid()
    data["sha256"] = value

    assert _error(data).field == "sha256"


def test_mf10_empty_input_names() -> None:
    data = _valid()
    data["input_names"] = []

    assert _error(data).field == "input_names"


def test_mf10_blank_input_name() -> None:
    data = _valid()
    data["input_names"] = ["source", "   "]

    assert _error(data).field == "input_names"


def test_mf10_shape_length_mismatch() -> None:
    data = _valid()
    data["input_shapes"] = [["batch", 512]]

    assert _error(data).field == "input_shapes"


def test_mf10_empty_shape() -> None:
    data = _valid()
    data["input_shapes"] = [[], [1, 3, 128, 128]]

    assert _error(data).field == "input_shapes"


@pytest.mark.parametrize("dimension", [0, -1, 1.5, True, ""])
def test_mf10_bad_dimension(dimension: object) -> None:
    data = _valid()
    data["input_shapes"] = [[dimension], [1, 3, 128, 128]]

    assert _error(data).field == "input_shapes"


def test_mf10_symbolic_dimension_accepted() -> None:
    manifest = parse_manifest(_valid())

    assert manifest.input_shapes[0] == ("batch", 512)


@pytest.mark.parametrize(
    "providers",
    [[], ["CUDAExecutionProvider", ""], ["CUDAExecutionProvider", 1]],
)
def test_mf11_providers(providers: list[object]) -> None:
    data = _valid()
    data["supported_execution_providers"] = providers

    assert _error(data).field == "supported_execution_providers"


@pytest.mark.parametrize("value", [0, 6144])
def test_mf12_vram_accepted(value: int) -> None:
    data = _valid()
    data["minimum_vram_mb"] = value

    assert parse_manifest(data).minimum_vram_mb == value


@pytest.mark.parametrize("value", [-1, 1.5, "6144", True])
def test_mf12_vram_rejected(value: object) -> None:
    data = _valid()
    data["minimum_vram_mb"] = value

    assert _error(data).field == "minimum_vram_mb"


@pytest.mark.parametrize("field", ["redistribution_allowed", "commercial_use_allowed"])
@pytest.mark.parametrize("value", [True, False])
def test_mf13_booleans_accepted(field: str, value: bool) -> None:
    data = _valid()
    data[field] = value
    manifest = parse_manifest(data)

    if field == "redistribution_allowed":
        assert manifest.redistribution_allowed is value
    else:
        assert manifest.commercial_use_allowed is value


@pytest.mark.parametrize("field", ["redistribution_allowed", "commercial_use_allowed"])
@pytest.mark.parametrize("value", ["true", 1, None])
def test_mf13_booleans_rejected(field: str, value: object) -> None:
    data = _valid()
    data[field] = value

    assert _error(data).field == field


def test_mf14_unknown_role_and_approval() -> None:
    role = _valid()
    role["model_role"] = "foo"
    approval = _valid()
    approval["approval_status"] = "maybe"

    assert _error(role).field == "model_role"
    assert _error(approval).field == "approval_status"


@pytest.mark.parametrize("field", ["normalization", "output_contract"])
@pytest.mark.parametrize("value", [["mean"], "rgb"])
def test_mf15_object_fields_reject_non_objects(field: str, value: object) -> None:
    data = _valid()
    data[field] = value

    assert _error(data).field == field


@pytest.mark.parametrize("field", ["attribution_text", "source_url"])
def test_mf16_empty_strings_accepted(field: str) -> None:
    data = _valid()
    data[field] = ""
    manifest = parse_manifest(data)

    if field == "attribution_text":
        assert manifest.attribution_text == ""
    else:
        assert manifest.source_url == ""


@pytest.mark.parametrize("field", ["attribution_text", "source_url"])
@pytest.mark.parametrize("value", [1, None])
def test_mf16_non_strings_rejected(field: str, value: object) -> None:
    data = _valid()
    data[field] = value

    assert _error(data).field == field


@pytest.mark.parametrize("value", [[], "x"])
def test_mf17_non_dict_document(value: object) -> None:
    with pytest.raises(ManifestError) as exc_info:
        parse_manifest(value)

    assert exc_info.value.field is None


def test_mf18_load_manifest(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_valid()), encoding="utf-8")
    assert load_manifest(path) == parse_manifest(_valid())

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ManifestError) as invalid_json:
        load_manifest(invalid)
    assert invalid_json.value.field is None

    raw = tmp_path / "raw.json"
    raw.write_bytes(b"\xff")
    with pytest.raises(ManifestError) as invalid_text:
        load_manifest(raw)
    assert invalid_text.value.field is None

    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.json")

    marker = "MARKER-UNIQUE-9f3a"
    payload = _valid()
    payload["model_role"] = marker
    marked = tmp_path / "marked.json"
    text = json.dumps(payload)
    marked.write_text(text, encoding="utf-8")
    with pytest.raises(ManifestError) as marked_error:
        load_manifest(marked)
    message = str(marked_error.value)
    assert marker not in message
    assert text not in message


def test_mf19_manifest_is_frozen() -> None:
    manifest = parse_manifest(_valid())
    field_name = "model_id"

    assert isinstance(manifest, ModelManifest)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(manifest, field_name, "other")


def test_mf20_import_safety() -> None:
    tree = ast.parse(_MANIFEST_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _assert_allowed_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            _assert_allowed_module(node.module or "")

    for statement in tree.body:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and _is_forbidden_call(node):
                raise AssertionError("module-level load_manifest or open call")


def _assert_allowed_module(module: str) -> None:
    assert module in _ALLOWED_MODULES
    assert not module.startswith("faceswap")
    assert module not in _FORBIDDEN_MODULES


def _is_forbidden_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in {"load_manifest", "open"}
    if isinstance(func, ast.Attribute):
        return func.attr in {"load_manifest", "open"}
    return False
