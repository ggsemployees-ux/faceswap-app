"""Unit tests for the live frame contract."""

import ast
import dataclasses
import pathlib
import tomllib

import numpy
import pytest

from faceswap.capture.frame import FramePacket, PixelFormat, monotonic_timestamp_ns

_FRAME_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "faceswap" / "capture" / "frame.py"
)
_PYPROJECT_PATH = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
_ALLOWED_MODULES = frozenset({"__future__", "dataclasses", "enum", "time", "numpy", "numpy.typing"})
_ARRAY_CONSTRUCTORS = frozenset(
    {"array", "asarray", "empty", "zeros", "ones", "full", "ndarray", "frombuffer"}
)
_DEV_PINS = ["pytest==9.1.1", "mypy==2.4.0", "ruff==0.16.10"]


def _buffer(height: int = 4, width: int = 6) -> numpy.ndarray:
    return numpy.zeros((height, width, 3), dtype=numpy.uint8)


def _packet(
    *,
    sequence_id: int = 7,
    capture_timestamp_ns: int = 1_000_000_000,
    width: int = 6,
    height: int = 4,
    pixel_format: PixelFormat = PixelFormat.BGR24,
    cpu_buffer: numpy.ndarray | None = None,
    camera_id: str = "cam0",
) -> FramePacket:
    return FramePacket(
        sequence_id=sequence_id,
        capture_timestamp_ns=capture_timestamp_ns,
        width=width,
        height=height,
        pixel_format=pixel_format,
        cpu_buffer=cpu_buffer if cpu_buffer is not None else _buffer(height, width),
        camera_id=camera_id,
    )


def test_pixel_format_has_only_bgr24() -> None:
    assert [(member.name, member.value) for member in PixelFormat] == [("BGR24", "bgr24")]


def test_valid_packet_reads_back_its_fields() -> None:
    buffer = _buffer()
    packet = FramePacket(
        sequence_id=7,
        capture_timestamp_ns=1_000_000_000,
        width=6,
        height=4,
        pixel_format=PixelFormat.BGR24,
        cpu_buffer=buffer,
        camera_id="cam0",
    )

    assert packet.sequence_id == 7
    assert packet.capture_timestamp_ns == 1_000_000_000
    assert packet.width == 6
    assert packet.height == 4
    assert packet.pixel_format is PixelFormat.BGR24
    assert packet.cpu_buffer is buffer
    assert packet.camera_id == "cam0"

    full_hd = numpy.zeros((1080, 1920, 3), dtype=numpy.uint8)
    hd_packet = FramePacket(
        sequence_id=0,
        capture_timestamp_ns=0,
        width=1920,
        height=1080,
        pixel_format=PixelFormat.BGR24,
        cpu_buffer=full_hd,
        camera_id="cam0",
    )
    assert hd_packet.cpu_buffer.shape == (1080, 1920, 3)


@pytest.mark.parametrize(
    "field_name",
    [
        "sequence_id",
        "capture_timestamp_ns",
        "width",
        "height",
        "pixel_format",
        "cpu_buffer",
        "camera_id",
    ],
)
def test_frame_packet_fields_are_frozen(field_name: str) -> None:
    packet = _packet()

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(packet, field_name, getattr(packet, field_name))


def test_buffer_is_not_copied_and_becomes_read_only() -> None:
    buffer = _buffer()
    packet = _packet(cpu_buffer=buffer)

    assert packet.cpu_buffer is buffer
    with pytest.raises(ValueError):
        buffer[0, 0, 0] = 1


@pytest.mark.parametrize(
    ("field_name", "value", "expected"),
    [
        ("sequence_id", -1, ValueError),
        ("sequence_id", True, TypeError),
        ("sequence_id", 1.5, TypeError),
        ("capture_timestamp_ns", -1, ValueError),
        ("capture_timestamp_ns", True, TypeError),
        ("width", 0, ValueError),
        ("height", 0, ValueError),
        ("width", True, TypeError),
        ("camera_id", "", ValueError),
        ("camera_id", "   ", ValueError),
        ("camera_id", 5, TypeError),
        ("pixel_format", "bgr24", TypeError),
    ],
)
def test_field_validation(field_name: str, value: object, expected: type[Exception]) -> None:
    kwargs: dict[str, object] = {
        "sequence_id": 0,
        "capture_timestamp_ns": 0,
        "width": 6,
        "height": 4,
        "pixel_format": PixelFormat.BGR24,
        "cpu_buffer": _buffer(),
        "camera_id": "cam0",
    }
    kwargs[field_name] = value

    with pytest.raises(expected):
        FramePacket(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("buffer", "expected"),
    [
        ([[0, 0, 0]], TypeError),
        (numpy.zeros((4, 6, 3), dtype=numpy.float32), TypeError),
        (numpy.zeros((6, 4, 3), dtype=numpy.uint8), ValueError),
        (numpy.zeros((4, 6), dtype=numpy.uint8), ValueError),
        (numpy.zeros((4, 6, 4), dtype=numpy.uint8), ValueError),
        (numpy.zeros((4, 12, 3), dtype=numpy.uint8)[:, ::2], ValueError),
    ],
)
def test_buffer_validation_leaves_arrays_writable(
    buffer: object, expected: type[Exception]
) -> None:
    writeable_before = buffer.flags.writeable if isinstance(buffer, numpy.ndarray) else None

    with pytest.raises(expected):
        FramePacket(
            sequence_id=0,
            capture_timestamp_ns=0,
            width=6,
            height=4,
            pixel_format=PixelFormat.BGR24,
            cpu_buffer=buffer,  # type: ignore[arg-type]
            camera_id="cam0",
        )

    if isinstance(buffer, numpy.ndarray):
        assert buffer.flags.writeable is writeable_before


def test_monotonic_timestamp_never_decreases() -> None:
    stamps = [monotonic_timestamp_ns() for _ in range(1_000)]

    assert all(isinstance(stamp, int) for stamp in stamps)
    assert all(stamps[index] <= stamps[index + 1] for index in range(len(stamps) - 1))


def test_age_ms() -> None:
    packet = _packet(capture_timestamp_ns=1_000_000_000)

    assert packet.age_ms(1_033_000_000) == 33.0
    assert packet.age_ms(1_000_000_000) == 0.0
    with pytest.raises(ValueError, match="now_ns"):
        packet.age_ms(999_999_999)
    with pytest.raises(TypeError, match="now_ns"):
        packet.age_ms(True)


def test_identity_equality_and_hashing() -> None:
    first = _packet()
    second = _packet()

    assert first != second
    assert first == first
    assert len({first, second}) == 2
    assert first in {first}


def test_pyproject_pins_only_numpy() -> None:
    parsed = tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))
    project = parsed["project"]
    assert isinstance(project, dict)
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    assert len(dependencies) == 1
    entry = dependencies[0]
    assert isinstance(entry, str)
    assert entry.startswith("numpy==")
    version = entry.removeprefix("numpy==")
    parts = version.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() and not part.startswith("+") for part in parts)
    assert not any(token in entry for token in (">", "<", "~", "*"))
    optional = project["optional-dependencies"]
    assert isinstance(optional, dict)
    assert optional["dev"] == _DEV_PINS


def test_frame_module_import_safety() -> None:
    tree = ast.parse(_FRAME_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in _ALLOWED_MODULES
                assert not alias.name.startswith("faceswap")
                assert alias.name != "cv2"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module in _ALLOWED_MODULES
            assert not module.startswith("faceswap")
            assert module != "cv2"

    for statement in tree.body:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and _is_forbidden_module_call(node):
                raise AssertionError("module-level constructor call")


def _is_forbidden_module_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "FramePacket" or func.id in _ARRAY_CONSTRUCTORS
    if isinstance(func, ast.Attribute):
        return func.attr == "FramePacket" or func.attr in _ARRAY_CONSTRUCTORS
    return False
