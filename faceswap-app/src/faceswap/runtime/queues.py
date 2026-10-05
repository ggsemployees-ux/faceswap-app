"""Bounded latest-frame queue per NFR-PERF-003, spec §10.2 items 15, 16 and 22, and ADR-005.

Item 15 accepts a frame and discards the oldest unprocessed frame when the queue is full.
Item 16 hands the consumer the newest eligible frame. Item 22 sends completed frames to the
application preview and virtual-camera latest-frame buffers. ADR-005 requires bounded queues
that drop stale frames. Spec §17 frames_dropped counts stale frames intentionally discarded.

Each operation that discards one or more stale items reports that count through the optional
on_drop callback after the queue lock is released and the new state is committed (D-8.4).
"""

from __future__ import annotations

import collections
import math
import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

MAX_CAPACITY = 3
"""Maximum queue capacity from NFR-PERF-003."""

DEFAULT_CAPACITY = 3
"""Default queue capacity."""


class LatestFrameQueue(Generic[T]):  # noqa: UP046 — D-8.1 requires Generic[T] and TypeVar.
    """Thread-safe bounded queue that keeps the newest items and drops stale ones.

    One threading.Condition wrapping a threading.Lock guards every read and write.
    The class does not start any threads. Items are stored by reference and are not copied.
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        on_drop: Callable[[int], None] | None = None,
    ) -> None:
        """Create a queue that holds at most capacity items and reports drops via on_drop."""
        self._capacity = _require_capacity(capacity)
        self._on_drop = on_drop
        self._items: collections.deque[T] = collections.deque()
        self._dropped = 0
        self._closed = False
        self._condition = threading.Condition(threading.Lock())

    @property
    def capacity(self) -> int:
        """Maximum number of items the queue can hold."""
        with self._condition:
            return self._capacity

    @property
    def depth(self) -> int:
        """Number of items currently stored."""
        with self._condition:
            return len(self._items)

    @property
    def dropped_count(self) -> int:
        """Lifetime count of stale items intentionally discarded."""
        with self._condition:
            return self._dropped

    @property
    def closed(self) -> bool:
        """Whether close has been called."""
        with self._condition:
            return self._closed

    def __len__(self) -> int:
        """Return the current depth."""
        return self.depth

    def put(self, item: T) -> None:
        """Store item without blocking, dropping the oldest item when the queue is full.

        A closed queue ignores the item. That discard is not counted as a drop.
        """
        dropped = False
        with self._condition:
            if self._closed:
                return
            if len(self._items) >= self._capacity:
                self._items.popleft()
                self._dropped += 1
                dropped = True
            self._items.append(item)
            self._condition.notify(1)
        if dropped and self._on_drop is not None:
            self._on_drop(1)

    def get_latest(self, timeout: float | None = None) -> T | None:
        """Return the newest item and count every older item still queued as a drop.

        timeout None waits until an item arrives or the queue is closed. timeout 0 never
        waits. Returns None on timeout or when the queue is closed.
        """
        _validate_timeout(timeout)
        item, dropped = self._wait_and_take(timeout)
        if dropped > 0 and self._on_drop is not None:
            self._on_drop(dropped)
        return item

    def close(self) -> None:
        """Close the queue, discard queued items, and wake every waiting consumer.

        Items removed here are shutdown discards, not stale-frame drops. Further calls
        leave the queue closed and raise nothing.
        """
        with self._condition:
            self._closed = True
            self._items.clear()
            self._condition.notify_all()

    def _wait_and_take(self, timeout: float | None) -> tuple[T | None, int]:
        with self._condition:
            self._wait_for_item(timeout)
            if not self._items:
                return None, 0
            newest = self._items.pop()
            dropped = len(self._items)
            self._items.clear()
            self._dropped += dropped
            return newest, dropped

    def _wait_for_item(self, timeout: float | None) -> None:
        if self._items or self._closed or timeout == 0:
            return
        if timeout is None:
            while not self._items and not self._closed:
                self._condition.wait()
            return
        deadline = time.monotonic() + timeout
        while not self._items and not self._closed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._condition.wait(remaining)


def _require_capacity(capacity: object) -> int:
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an int")
    if not 1 <= capacity <= MAX_CAPACITY:
        raise ValueError("capacity must be >= 1 and <= MAX_CAPACITY")
    return capacity


def _validate_timeout(timeout: object) -> None:
    if timeout is None:
        return
    if isinstance(timeout, bool) or not isinstance(timeout, int | float):
        raise TypeError("timeout must be None or a finite int or float >= 0")
    if not math.isfinite(timeout):
        raise ValueError("timeout must be finite and >= 0")
    if timeout < 0:
        raise ValueError("timeout must be finite and >= 0")
