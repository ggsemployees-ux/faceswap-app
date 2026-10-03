"""Unit tests for the application runtime state machine."""

import ast
import dataclasses
import itertools
import pathlib

import pytest

from faceswap.app.state_machine import (
    ALLOWED_TRANSITIONS,
    AppState,
    InvalidTransitionError,
    StateChange,
    StateMachine,
)
from faceswap.runtime.errors import AppError, ErrorCode, make_error

_STATE_MACHINE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "faceswap" / "app" / "state_machine.py"
)
_ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "dataclasses",
        "enum",
        "types",
        "collections.abc",
        "faceswap.runtime.errors",
    }
)
_EXPECTED_TRANSITIONS = {
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
_ROUTES: dict[AppState, tuple[AppState, ...]] = {
    AppState.INIT: (),
    AppState.ENV_CHECK: (AppState.ENV_CHECK,),
    AppState.MODEL_LOADING: (AppState.ENV_CHECK, AppState.MODEL_LOADING),
    AppState.READY: (AppState.ENV_CHECK, AppState.MODEL_LOADING, AppState.READY),
    AppState.STARTING_CAMERA: (
        AppState.ENV_CHECK,
        AppState.MODEL_LOADING,
        AppState.READY,
        AppState.STARTING_CAMERA,
    ),
    AppState.RUNNING: (
        AppState.ENV_CHECK,
        AppState.MODEL_LOADING,
        AppState.READY,
        AppState.STARTING_CAMERA,
        AppState.RUNNING,
    ),
    AppState.SEARCHING_FACE: (
        AppState.ENV_CHECK,
        AppState.MODEL_LOADING,
        AppState.READY,
        AppState.STARTING_CAMERA,
        AppState.RUNNING,
        AppState.SEARCHING_FACE,
    ),
    AppState.STOPPING: (
        AppState.ENV_CHECK,
        AppState.MODEL_LOADING,
        AppState.READY,
        AppState.STARTING_CAMERA,
        AppState.RUNNING,
        AppState.STOPPING,
    ),
}
_LEGAL_PAIRS = [
    (source, target) for source, targets in _EXPECTED_TRANSITIONS.items() for target in targets
]
_ILLEGAL_PAIRS = [
    (source, target)
    for source, target in itertools.product(AppState, AppState)
    if target not in _EXPECTED_TRANSITIONS[source]
]
_NON_ERROR_STATES = [state for state in AppState if state is not AppState.ERROR]


def drive_to(target: AppState, error: AppError | None = None) -> StateMachine:
    """Return a new machine moved to target using only legal calls."""
    machine = StateMachine()
    if target is AppState.ERROR:
        machine.fail(error if error is not None else make_error(ErrorCode.CAMERA_UNAVAILABLE))
        return machine
    for step in _ROUTES[target]:
        machine.transition_to(step)
    return machine


def test_app_state_members_are_exact() -> None:
    assert [(member.name, member.value) for member in AppState] == [
        ("INIT", "init"),
        ("ENV_CHECK", "env_check"),
        ("MODEL_LOADING", "model_loading"),
        ("READY", "ready"),
        ("STARTING_CAMERA", "starting_camera"),
        ("RUNNING", "running"),
        ("SEARCHING_FACE", "searching_face"),
        ("STOPPING", "stopping"),
        ("ERROR", "error"),
    ]


def test_allowed_transitions_match_the_table() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(AppState)
    assert dict(ALLOWED_TRANSITIONS) == _EXPECTED_TRANSITIONS


def test_allowed_transitions_are_read_only() -> None:
    with pytest.raises(TypeError):
        ALLOWED_TRANSITIONS[AppState.INIT] = frozenset()  # type: ignore[index]


def test_new_machine_starts_in_init() -> None:
    machine = StateMachine()

    assert machine.state is AppState.INIT
    assert machine.error is None


def test_happy_path_and_second_cycle() -> None:
    machine = StateMachine()
    sequence = (
        AppState.ENV_CHECK,
        AppState.MODEL_LOADING,
        AppState.READY,
        AppState.STARTING_CAMERA,
        AppState.RUNNING,
        AppState.SEARCHING_FACE,
        AppState.RUNNING,
        AppState.STOPPING,
        AppState.READY,
        AppState.STARTING_CAMERA,
        AppState.RUNNING,
        AppState.STOPPING,
        AppState.READY,
    )

    for target in sequence:
        machine.transition_to(target)
        assert machine.state is target
        assert machine.error is None


@pytest.mark.parametrize(("source", "target"), _LEGAL_PAIRS)
def test_legal_transitions_succeed(source: AppState, target: AppState) -> None:
    machine = drive_to(source)

    machine.transition_to(target)

    assert machine.state is target
    assert machine.error is None


@pytest.mark.parametrize(("source", "target"), _ILLEGAL_PAIRS)
def test_illegal_transitions_are_rejected(source: AppState, target: AppState) -> None:
    machine = drive_to(source)
    previous_error = machine.error
    calls: list[StateChange] = []
    machine.add_listener(calls.append)

    with pytest.raises(InvalidTransitionError) as caught:
        machine.transition_to(target)

    assert source.value in str(caught.value)
    assert target.value in str(caught.value)
    assert machine.state is source
    assert machine.error is previous_error
    assert calls == []


@pytest.mark.parametrize(("source", "target"), list(itertools.product(AppState, AppState)))
def test_can_transition_to_matches_the_table(source: AppState, target: AppState) -> None:
    machine = drive_to(source)

    assert machine.can_transition_to(target) is (target in ALLOWED_TRANSITIONS[source])


@pytest.mark.parametrize("source", _NON_ERROR_STATES)
def test_fail_enters_error_from_every_non_error_state(source: AppState) -> None:
    machine = drive_to(source)
    error = make_error(ErrorCode.GPU_OUT_OF_MEMORY, detail="oom")

    machine.fail(error)

    assert machine.state is AppState.ERROR
    assert machine.error is error


def test_fail_while_in_error_keeps_the_original_error() -> None:
    original = make_error(ErrorCode.CAMERA_UNAVAILABLE, detail="first")
    machine = drive_to(AppState.ERROR, error=original)

    with pytest.raises(InvalidTransitionError):
        machine.fail(make_error(ErrorCode.GPU_OUT_OF_MEMORY, detail="second"))

    assert machine.state is AppState.ERROR
    assert machine.error is original


@pytest.mark.parametrize("target", [AppState.ENV_CHECK, AppState.READY])
def test_leaving_error_clears_the_stored_error(target: AppState) -> None:
    machine = drive_to(AppState.ERROR)

    machine.transition_to(target)

    assert machine.state is target
    assert machine.error is None


def test_fail_rejects_a_non_app_error() -> None:
    machine = drive_to(AppState.ENV_CHECK)

    with pytest.raises(TypeError, match="AppError"):
        machine.fail("not an app error")  # type: ignore[arg-type]

    assert machine.state is AppState.ENV_CHECK
    assert machine.error is None


def test_listeners_are_called_in_order_and_can_unregister() -> None:
    machine = StateMachine()
    order: list[str] = []
    seen_a: list[StateChange] = []
    seen_b: list[StateChange] = []

    def listener_a(change: StateChange) -> None:
        order.append("a")
        seen_a.append(change)

    def listener_b(change: StateChange) -> None:
        order.append("b")
        seen_b.append(change)

    remove_a = machine.add_listener(listener_a)
    machine.add_listener(listener_b)
    machine.transition_to(AppState.ENV_CHECK)
    error = make_error(ErrorCode.NATIVE_COMPONENT_FAILURE, detail="native")
    machine.fail(error)

    expected = [
        StateChange(previous=AppState.INIT, current=AppState.ENV_CHECK, error=None),
        StateChange(previous=AppState.ENV_CHECK, current=AppState.ERROR, error=error),
    ]
    assert seen_a == expected
    assert seen_b == expected
    assert order == ["a", "b", "a", "b"]

    remove_a()
    machine.transition_to(AppState.READY)

    assert seen_a == expected
    assert seen_b[-1] == StateChange(
        previous=AppState.ERROR,
        current=AppState.READY,
        error=None,
    )
    remove_a()


def test_listener_sees_the_committed_state() -> None:
    machine = StateMachine()
    observed: list[tuple[AppState, AppError | None]] = []

    def listener(change: StateChange) -> None:
        observed.append((machine.state, machine.error))
        assert machine.state is change.current
        assert machine.error is change.error

    machine.add_listener(listener)
    error = make_error(ErrorCode.CUDA_PROVIDER_UNAVAILABLE)
    machine.transition_to(AppState.ENV_CHECK)
    machine.fail(error)

    assert observed == [(AppState.ENV_CHECK, None), (AppState.ERROR, error)]


def test_listener_exception_propagates_after_commit() -> None:
    machine = StateMachine()

    def listener(_change: StateChange) -> None:
        raise RuntimeError("listener failed")

    machine.add_listener(listener)

    with pytest.raises(RuntimeError, match="listener failed"):
        machine.transition_to(AppState.ENV_CHECK)

    assert machine.state is AppState.ENV_CHECK
    assert machine.error is None


def test_state_change_is_frozen() -> None:
    change = StateChange(previous=AppState.INIT, current=AppState.ENV_CHECK, error=None)
    field_name = "current"

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(change, field_name, AppState.READY)


def test_identical_sequences_produce_identical_changes() -> None:
    def record(machine: StateMachine) -> list[StateChange]:
        changes: list[StateChange] = []
        machine.add_listener(changes.append)
        machine.transition_to(AppState.ENV_CHECK)
        machine.transition_to(AppState.MODEL_LOADING)
        machine.fail(make_error(ErrorCode.MODEL_CHECKSUM_INVALID, detail="same"))
        machine.transition_to(AppState.READY)
        return changes

    assert record(StateMachine()) == record(StateMachine())


def test_state_machine_module_import_safety() -> None:
    tree = ast.parse(_STATE_MACHINE_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in _ALLOWED_IMPORTS
                if alias.name.startswith("faceswap"):
                    assert alias.name == "faceswap.runtime.errors"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module in _ALLOWED_IMPORTS
            assert module == "faceswap.runtime.errors" or not module.startswith("faceswap")

    for statement in tree.body:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and _is_state_machine_call(node):
                raise AssertionError("module-level StateMachine call")


def _is_state_machine_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "StateMachine"
    return isinstance(func, ast.Attribute) and func.attr == "StateMachine"
