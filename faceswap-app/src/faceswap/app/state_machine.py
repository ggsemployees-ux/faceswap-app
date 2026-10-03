"""Application runtime state machine per spec §16.2.

Deterministic cleanup and recovery policy belong to the controller.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from faceswap.runtime.errors import AppError


class AppState(enum.StrEnum):
    """Runtime states from spec §16.2."""

    INIT = "init"
    ENV_CHECK = "env_check"
    MODEL_LOADING = "model_loading"
    READY = "ready"
    STARTING_CAMERA = "starting_camera"
    RUNNING = "running"
    SEARCHING_FACE = "searching_face"
    STOPPING = "stopping"
    ERROR = "error"


ALLOWED_TRANSITIONS: Mapping[AppState, frozenset[AppState]] = MappingProxyType(
    {
        AppState.INIT: frozenset({AppState.ENV_CHECK}),
        AppState.ENV_CHECK: frozenset({AppState.MODEL_LOADING}),
        AppState.MODEL_LOADING: frozenset({AppState.READY}),
        AppState.READY: frozenset({AppState.STARTING_CAMERA}),
        AppState.STARTING_CAMERA: frozenset({AppState.RUNNING}),
        AppState.RUNNING: frozenset({AppState.SEARCHING_FACE, AppState.STOPPING}),
        AppState.SEARCHING_FACE: frozenset({AppState.RUNNING, AppState.STOPPING}),
        AppState.STOPPING: frozenset({AppState.READY}),
        AppState.ERROR: frozenset({AppState.ENV_CHECK, AppState.READY}),
    }
)


class InvalidTransitionError(RuntimeError):
    """Illegal transition. The message names the source and target states."""

    source: AppState
    target: AppState

    def __init__(self, source: AppState, target: AppState) -> None:
        self.source = source
        self.target = target
        super().__init__(f"illegal transition from {source.value} to {target.value}")


@dataclass(frozen=True, slots=True)
class StateChange:
    """One committed transition, including the error stored after the change."""

    previous: AppState
    current: AppState
    error: AppError | None


class StateMachine:
    """Records and validates application state.

    The instance is not thread-safe. The controller serializes all calls.
    """

    def __init__(self) -> None:
        """Start in INIT with no error and no listeners."""
        self._state = AppState.INIT
        self._error: AppError | None = None
        self._listeners: list[Callable[[StateChange], None]] = []

    @property
    def state(self) -> AppState:
        """Current state."""
        return self._state

    @property
    def error(self) -> AppError | None:
        """AppError while in ERROR, otherwise None."""
        return self._error

    def can_transition_to(self, target: AppState) -> bool:
        """Return whether target is in the allowed table for the current state."""
        return target in ALLOWED_TRANSITIONS[self._state]

    def transition_to(self, target: AppState) -> None:
        """Move to target when the table allows it, then notify listeners."""
        source = self._state
        if target not in ALLOWED_TRANSITIONS[source]:
            raise InvalidTransitionError(source, target)
        self._state = target
        self._error = None
        self._notify(StateChange(previous=source, current=target, error=None))

    def fail(self, error: AppError) -> None:
        """Enter ERROR with error. Illegal when the machine is already in ERROR."""
        if not isinstance(error, AppError):
            raise TypeError("error must be an AppError")
        source = self._state
        if source is AppState.ERROR:
            raise InvalidTransitionError(source, AppState.ERROR)
        self._state = AppState.ERROR
        self._error = error
        self._notify(StateChange(previous=source, current=AppState.ERROR, error=error))

    def add_listener(self, callback: Callable[[StateChange], None]) -> Callable[[], None]:
        """Register callback and return a function that removes it."""
        self._listeners.append(callback)

        def unregister() -> None:
            try:
                self._listeners.remove(callback)
            except ValueError:
                return

        return unregister

    def _notify(self, change: StateChange) -> None:
        for listener in tuple(self._listeners):
            listener(change)
