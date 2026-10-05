"""Unit tests for the bounded latest-frame queue."""

import ast
import pathlib
import threading
import time

import numpy
import pytest

from faceswap.capture.frame import FramePacket, PixelFormat
from faceswap.runtime.queues import DEFAULT_CAPACITY, MAX_CAPACITY, LatestFrameQueue

_QUEUES_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "faceswap" / "runtime" / "queues.py"
)
_ALLOWED_MODULES = frozenset(
    {
        "__future__",
        "collections",
        "collections.abc",
        "math",
        "threading",
        "time",
        "typing",
    }
)
_JOIN_TIMEOUT_S = 10.0


def _packet(sequence_id: int) -> FramePacket:
    return FramePacket(
        sequence_id=sequence_id,
        capture_timestamp_ns=sequence_id,
        width=6,
        height=4,
        pixel_format=PixelFormat.BGR24,
        cpu_buffer=numpy.zeros((4, 6, 3), dtype=numpy.uint8),
        camera_id="cam0",
    )


def _join(worker: threading.Thread) -> None:
    worker.join(_JOIN_TIMEOUT_S)
    assert not worker.is_alive()


def test_q1_defaults_and_initial_state() -> None:
    assert MAX_CAPACITY == 3
    assert DEFAULT_CAPACITY == 3
    queue = LatestFrameQueue[int]()

    assert queue.capacity == 3
    assert queue.depth == 0
    assert queue.dropped_count == 0
    assert queue.closed is False
    assert len(queue) == 0


@pytest.mark.parametrize(("capacity", "expected"), [(0, ValueError), (4, ValueError)])
def test_q2_capacity_out_of_range(capacity: int, expected: type[Exception]) -> None:
    with pytest.raises(expected):
        LatestFrameQueue[int](capacity=capacity)


@pytest.mark.parametrize("capacity", [1.5, True])
def test_q2_capacity_rejects_non_int(capacity: object) -> None:
    with pytest.raises(TypeError):
        LatestFrameQueue[int](capacity=capacity)  # type: ignore[arg-type]


@pytest.mark.parametrize("capacity", [1, 2, 3])
def test_q2_capacity_accepted(capacity: int) -> None:
    queue = LatestFrameQueue[int](capacity=capacity)

    assert queue.capacity == capacity


def test_q3_put_then_get_latest() -> None:
    queue = LatestFrameQueue[str]()
    queue.put("frame")

    assert queue.get_latest() == "frame"
    assert queue.depth == 0
    assert queue.dropped_count == 0


def test_q4_overflow_drops_oldest() -> None:
    drops: list[int] = []
    queue = LatestFrameQueue[int](capacity=3, on_drop=drops.append)
    for value in (1, 2, 3, 4, 5):
        queue.put(value)

    assert queue.depth == 3
    assert queue.dropped_count == 2
    assert drops == [1, 1]
    assert queue.get_latest() == 5


def test_q5_get_latest_returns_newest_and_drops_older() -> None:
    drops: list[int] = []
    queue = LatestFrameQueue[int](on_drop=drops.append)
    for value in (1, 2, 3):
        queue.put(value)

    assert queue.get_latest() == 3
    assert queue.depth == 0
    assert queue.dropped_count == 2
    assert drops == [2]


def test_q6_timeouts_on_empty_queue() -> None:
    queue = LatestFrameQueue[int]()
    immediate = time.monotonic()

    assert queue.get_latest(timeout=0) is None
    assert time.monotonic() - immediate < 0.04

    started = time.monotonic()
    assert queue.get_latest(timeout=0.05) is None
    assert time.monotonic() - started >= 0.04


def test_q7_waiting_consumer_wakes_on_put() -> None:
    queue = LatestFrameQueue[str]()
    received: list[str | None] = []

    def consume() -> None:
        received.append(queue.get_latest(timeout=5))

    worker = threading.Thread(target=consume)
    worker.start()
    time.sleep(0.05)
    started = time.monotonic()
    queue.put("frame")
    _join(worker)

    assert time.monotonic() - started < 1.0
    assert received == ["frame"]


def test_q8_close_wakes_waiter_and_discards_without_drops() -> None:
    queue = LatestFrameQueue[int]()
    queue.put(1)
    queue.put(2)
    queue.close()

    assert queue.depth == 0
    assert queue.dropped_count == 0
    assert queue.closed is True
    assert queue.get_latest() is None
    queue.put(3)
    assert queue.depth == 0
    assert queue.dropped_count == 0
    queue.close()

    waiting: LatestFrameQueue[int] = LatestFrameQueue()
    received: list[int | None] = []

    def consume() -> None:
        received.append(waiting.get_latest(timeout=None))

    worker = threading.Thread(target=consume)
    worker.start()
    time.sleep(0.05)
    waiting.close()
    _join(worker)

    assert received == [None]
    assert waiting.closed is True


def test_q9_timeout_validation() -> None:
    queue = LatestFrameQueue[int]()

    with pytest.raises(ValueError):
        queue.get_latest(-1)
    with pytest.raises(ValueError):
        queue.get_latest(float("nan"))
    with pytest.raises(ValueError):
        queue.get_latest(float("inf"))
    with pytest.raises(TypeError):
        queue.get_latest("1")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        queue.get_latest(True)


def test_q10_on_drop_is_optional_and_exception_commits_state() -> None:
    queue = LatestFrameQueue[int](capacity=1, on_drop=None)
    queue.put(1)
    queue.put(2)

    assert queue.depth == 1
    assert queue.dropped_count == 1
    assert queue.get_latest() == 2

    def fail(_count: int) -> None:
        raise RuntimeError("drop")

    failing = LatestFrameQueue[str](capacity=1, on_drop=fail)
    failing.put("old")
    with pytest.raises(RuntimeError, match="drop"):
        failing.put("new")

    assert failing.depth == 1
    assert failing.dropped_count == 1
    assert failing.get_latest() == "new"


def test_q11_get_latest_returns_highest_sequence_frame() -> None:
    queue = LatestFrameQueue[FramePacket]()
    for sequence_id in (1, 2, 3):
        queue.put(_packet(sequence_id))

    latest = queue.get_latest()

    assert latest is not None
    assert latest.sequence_id == 3


def test_q12_stress_conservation() -> None:
    recorded: list[int] = []
    record_lock = threading.Lock()

    def on_drop(count: int) -> None:
        with record_lock:
            recorded.append(count)

    queue = LatestFrameQueue[int](capacity=3, on_drop=on_drop)
    consumed = [0]
    max_depth = [0]
    producers_done = threading.Event()
    stop_sampler = threading.Event()

    def produce(offset: int) -> None:
        for value in range(offset, offset + 5_000):
            queue.put(value)

    def consume() -> None:
        while not producers_done.is_set():
            item = queue.get_latest(timeout=0.01)
            if item is not None:
                consumed[0] += 1
        while True:
            item = queue.get_latest(timeout=0)
            if item is None:
                break
            consumed[0] += 1

    def sample() -> None:
        while not stop_sampler.is_set():
            depth = queue.depth
            if depth > max_depth[0]:
                max_depth[0] = depth
            time.sleep(0.001)

    producers = [threading.Thread(target=produce, args=(index * 5_000,)) for index in range(4)]
    consumer = threading.Thread(target=consume)
    sampler = threading.Thread(target=sample)
    for producer in producers:
        producer.start()
    consumer.start()
    sampler.start()
    for producer in producers:
        _join(producer)
    producers_done.set()
    _join(consumer)
    stop_sampler.set()
    _join(sampler)
    assert queue.depth == 0
    queue.close()

    assert max_depth[0] <= 3
    assert consumed[0] + queue.dropped_count + queue.depth == 20_000
    assert queue.dropped_count == sum(recorded)


def test_q13_import_safety() -> None:
    tree = ast.parse(_QUEUES_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in _ALLOWED_MODULES
                assert not alias.name.startswith("faceswap")
                assert alias.name != "logging"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module in _ALLOWED_MODULES
            assert not module.startswith("faceswap")
            assert module != "logging"

    for statement in tree.body:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and _is_queue_constructor(node):
                raise AssertionError("module-level LatestFrameQueue call")


def _is_queue_constructor(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "LatestFrameQueue"
    if isinstance(func, ast.Attribute):
        return func.attr == "LatestFrameQueue"
    return False
