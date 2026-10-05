"""Unit tests for SHA-256 model file verification."""

import ast
import dataclasses
import hashlib
import pathlib

import pytest

from faceswap.models.integrity import (
    DEFAULT_CHUNK_SIZE,
    IntegrityError,
    compute_sha256,
    verify_file,
    verify_model_file,
)
from faceswap.models.manifest import ModelManifest, parse_manifest

_INTEGRITY_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "faceswap" / "models" / "integrity.py"
)
_ALLOWED_MODULES = frozenset(
    {
        "__future__",
        "hashlib",
        "pathlib",
        "re",
        "faceswap.models.manifest",
    }
)
_FORBIDDEN_MODULES = frozenset({"onnx", "onnxruntime", "numpy", "logging"})
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
_FORBIDDEN_CALLS = frozenset({"compute_sha256", "verify_file", "verify_model_file", "open"})


def _manifest(file_name: str, sha256: str) -> ModelManifest:
    return parse_manifest(
        {
            "schema_version": 1,
            "model_id": "test-swapper",
            "model_version": "1.0.0",
            "model_role": "swapper",
            "file_name": file_name,
            "sha256": sha256,
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
    )


def _write(directory: pathlib.Path, name: str, payload: bytes) -> pathlib.Path:
    path = directory / name
    path.write_bytes(payload)
    return path


def test_i1_default_chunk_size() -> None:
    assert DEFAULT_CHUNK_SIZE == 1024 * 1024


def test_i2_known_sha256_vectors(tmp_path: pathlib.Path) -> None:
    empty = _write(tmp_path, "empty.bin", b"")
    abc = _write(tmp_path, "abc.bin", b"abc")

    empty_digest = compute_sha256(empty)
    abc_digest = compute_sha256(abc)

    assert empty_digest == _EMPTY_SHA256
    assert abc_digest == _ABC_SHA256
    assert empty_digest == empty_digest.lower()
    assert abc_digest == abc_digest.lower()


def test_i3_chunk_sizes_match_hashlib(tmp_path: pathlib.Path) -> None:
    payload = bytes(index % 251 for index in range(100_003))
    path = _write(tmp_path, "chunks.bin", payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert compute_sha256(path, chunk_size=1) == expected
    assert compute_sha256(path, chunk_size=7) == expected
    assert compute_sha256(path, chunk_size=4096) == expected
    assert compute_sha256(path, chunk_size=DEFAULT_CHUNK_SIZE) == expected


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_i4_chunk_size_rejects_non_positive(chunk_size: int, tmp_path: pathlib.Path) -> None:
    path = _write(tmp_path, "model.bin", b"abc")

    with pytest.raises(ValueError):
        compute_sha256(path, chunk_size=chunk_size)
    with pytest.raises(ValueError):
        verify_file(path, _ABC_SHA256, chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", [1.5, True])
def test_i4_chunk_size_rejects_non_int(chunk_size: object, tmp_path: pathlib.Path) -> None:
    path = _write(tmp_path, "model.bin", b"abc")

    with pytest.raises(TypeError):
        compute_sha256(path, chunk_size=chunk_size)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        verify_file(path, _ABC_SHA256, chunk_size=chunk_size)  # type: ignore[arg-type]


def test_i5_compute_sha256_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_sha256(tmp_path / "missing.bin")


def test_i6_verify_file_accepts_either_hex_case(tmp_path: pathlib.Path) -> None:
    path = _write(tmp_path, "abc.bin", b"abc")

    assert verify_file(path, _ABC_SHA256) is None  # type: ignore[func-returns-value]
    assert verify_file(path, _ABC_SHA256.upper()) is None  # type: ignore[func-returns-value]


def test_i7_verify_file_mismatch_reports_digests(tmp_path: pathlib.Path) -> None:
    path = _write(tmp_path, "abc.bin", b"abc")
    expected = "b" * 64

    with pytest.raises(IntegrityError) as exc_info:
        verify_file(path, expected.upper())
    error = exc_info.value

    assert error.path == path
    assert error.expected == expected
    assert error.actual == _ABC_SHA256
    message = str(error)
    assert path.name in message
    assert error.expected in message
    assert error.actual in message


@pytest.mark.parametrize("digest", ["a" * 63, "g" + ("a" * 63), ""])
def test_i8_malformed_digest_is_value_error(digest: str, tmp_path: pathlib.Path) -> None:
    path = _write(tmp_path, "abc.bin", b"abc")

    with pytest.raises(ValueError) as exc_info:
        verify_file(path, digest)

    assert exc_info.type is ValueError


@pytest.mark.parametrize("digest", [123, None])
def test_i8_non_str_digest_is_type_error(digest: object, tmp_path: pathlib.Path) -> None:
    path = _write(tmp_path, "abc.bin", b"abc")

    with pytest.raises(TypeError):
        verify_file(path, digest)  # type: ignore[arg-type]


def test_i9_verify_file_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_file(tmp_path / "missing.bin", _ABC_SHA256)


def test_i10_one_byte_change_fails_verification(tmp_path: pathlib.Path) -> None:
    path = _write(tmp_path, "abc.bin", b"abc")
    digest = compute_sha256(path)
    path.write_bytes(b"abd")

    with pytest.raises(IntegrityError) as exc_info:
        verify_file(path, digest)

    assert exc_info.value.actual != digest


def test_i11_verify_model_file_returns_resolved_path(tmp_path: pathlib.Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _write(model_dir, "swapper.bin", b"abc")
    manifest = _manifest("swapper.bin", _ABC_SHA256)

    resolved = verify_model_file(manifest, model_dir)

    assert resolved == (model_dir / "swapper.bin").resolve()


def test_i12_verify_model_file_mismatch_and_missing(tmp_path: pathlib.Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _write(model_dir, "swapper.bin", b"abc")

    with pytest.raises(IntegrityError):
        verify_model_file(_manifest("swapper.bin", "a" * 64), model_dir)
    with pytest.raises(FileNotFoundError):
        verify_model_file(_manifest("missing.bin", _ABC_SHA256), model_dir)


def test_i13_path_escape_is_rejected_without_reading(tmp_path: pathlib.Path) -> None:
    payload = b"outside-file-contents"
    outside = _write(tmp_path, "outside.bin", payload)
    digest = hashlib.sha256(payload).hexdigest()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    manifest = dataclasses.replace(_manifest("swapper.bin", digest), file_name="../outside.bin")

    with pytest.raises(IntegrityError) as exc_info:
        verify_model_file(manifest, model_dir)
    error = exc_info.value

    assert "outside" in str(error)
    assert error.actual == ""
    assert outside.read_bytes() == payload


def test_i14_import_safety() -> None:
    tree = ast.parse(_INTEGRITY_PATH.read_text(encoding="utf-8"))

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
                raise AssertionError("module-level hash or open call")


def _assert_allowed_module(module: str) -> None:
    assert module in _ALLOWED_MODULES
    assert module not in _FORBIDDEN_MODULES
    if module.startswith("faceswap"):
        assert module == "faceswap.models.manifest"


def _is_forbidden_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _FORBIDDEN_CALLS
    if isinstance(func, ast.Attribute):
        return func.attr in _FORBIDDEN_CALLS
    return False
