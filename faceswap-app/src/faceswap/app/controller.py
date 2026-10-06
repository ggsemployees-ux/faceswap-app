"""Application controller for spec §16.2 and the Phase 1 exit gate.

The controller drives INIT through settings load, ENV_CHECK, and MODEL_LOADING to READY,
or to ERROR with an actionable AppError (FR-017). Persisted preferences are restored on
startup (FR-018). An empty model list is allowed in Phase 1; requiring specific model
roles is still pending the license decision (spec §32, D1).

Steps 5 and 6 are skipped. Transition and error logging is not implemented here:
listeners on state_machine observe every StateChange, and last_error plus
environment_report stay available for a later logging layer. MetricsSink is not used.
Phase 1 has no frame pipeline, and queue drops stay on LatestFrameQueue.on_drop.

This controller is not thread-safe and must be called from one thread. The UI calls
initialize and retry off the UI thread so the interface does not freeze.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from faceswap.app.config import SettingsLoadStatus, UserSettings, load_settings
from faceswap.app.state_machine import AppState, InvalidTransitionError, StateMachine
from faceswap.models.manager import ModelManager, VerifiedModel
from faceswap.runtime.device_profile import (
    EnvironmentReport,
    SystemInfo,
    collect_system_info,
    evaluate_environment,
)
from faceswap.runtime.errors import AppError, ErrorCode, FaceSwapError, make_error


class AppController:
    """Drives startup to READY or ERROR. Call it from one thread only."""

    def __init__(
        self,
        *,
        model_manager: ModelManager,
        manifest_paths: Sequence[Path] = (),
        settings_path: Path | None = None,
        state_machine: StateMachine | None = None,
        system_info_provider: Callable[[], SystemInfo] = collect_system_info,
        environment_evaluator: Callable[[SystemInfo], EnvironmentReport] = evaluate_environment,
    ) -> None:
        """Store collaborators. Does not read settings, the machine, or model files."""
        self._model_manager = model_manager
        self._manifest_paths = tuple(manifest_paths)
        self._settings_path = settings_path
        self._machine = StateMachine() if state_machine is None else state_machine
        self._system_info_provider = system_info_provider
        self._environment_evaluator = environment_evaluator
        self._environment_report: EnvironmentReport | None = None
        self._models: tuple[VerifiedModel, ...] = ()
        self._user_settings = UserSettings()
        self._settings_status: SettingsLoadStatus | None = None
        self._can_return_to_ready = False

    @property
    def state(self) -> AppState:
        """Current application state."""
        return self._machine.state

    @property
    def last_error(self) -> AppError | None:
        """AppError while the state is ERROR, otherwise None."""
        return self._machine.error

    @property
    def state_machine(self) -> StateMachine:
        """State machine listeners can subscribe to for every StateChange."""
        return self._machine

    @property
    def environment_report(self) -> EnvironmentReport | None:
        """Latest environment report, or None before a check completes."""
        return self._environment_report

    @property
    def models(self) -> tuple[VerifiedModel, ...]:
        """Models verified on the way to READY. Empty until that succeeds."""
        return self._models

    @property
    def user_settings(self) -> UserSettings:
        """Preferences loaded at initialize, or defaults when no file is used."""
        return self._user_settings

    @property
    def settings_status(self) -> SettingsLoadStatus | None:
        """How settings were loaded. None when settings_path was not provided."""
        return self._settings_status

    @property
    def can_return_to_ready(self) -> bool:
        """True after models were verified successfully earlier in this session."""
        return self._can_return_to_ready

    def initialize(self) -> AppState:
        """Load settings, then run ENV_CHECK and MODEL_LOADING. Only legal from INIT."""
        if self._machine.state is not AppState.INIT:
            raise InvalidTransitionError(self._machine.state, AppState.ENV_CHECK)
        self._load_settings()
        return self._check_phase()

    def retry(self) -> AppState:
        """Run ENV_CHECK and MODEL_LOADING again. Only legal from ERROR.

        Settings are not read again. Previously verified models are cleared until
        this attempt reaches READY.
        """
        if self._machine.state is not AppState.ERROR:
            raise InvalidTransitionError(self._machine.state, AppState.ENV_CHECK)
        self._models = ()
        self._environment_report = None
        self._can_return_to_ready = False
        return self._check_phase()

    def return_to_ready(self) -> AppState:
        """Return to READY after a later failure when models were already verified."""
        if self._machine.state is not AppState.ERROR or not self._can_return_to_ready:
            raise InvalidTransitionError(self.state, AppState.READY)
        self._machine.transition_to(AppState.READY)
        return self.state

    def report_failure(self, error: AppError) -> None:
        """Record a fatal subsystem failure. A second report while in ERROR is ignored."""
        if not isinstance(error, AppError):
            raise TypeError("error must be an AppError")
        if self._machine.state is AppState.ERROR:
            return
        self._machine.fail(error)

    def _load_settings(self) -> None:
        path = self._settings_path
        if path is None:
            return
        try:
            result = load_settings(path)
        except OSError:
            self._user_settings = UserSettings()
            self._settings_status = SettingsLoadStatus.INVALID
            return
        self._user_settings = result.settings
        self._settings_status = result.status

    def _check_phase(self) -> AppState:
        try:
            self._machine.transition_to(AppState.ENV_CHECK)
            report = self._environment_evaluator(self._system_info_provider())
            self._environment_report = report
            if not report.passed:
                detail = "; ".join(f"{check.name}: {check.message}" for check in report.failures)
                self._machine.fail(make_error(ErrorCode.ENVIRONMENT_UNSUPPORTED, detail))
                return self.state
        except Exception as exc:
            self._fail_internal("ENV_CHECK", exc)
            return self.state
        try:
            self._machine.transition_to(AppState.MODEL_LOADING)
            loaded = self._model_manager.verify_packages(self._manifest_paths)
        except FaceSwapError as exc:
            self._machine.fail(exc.error)
            return self.state
        except Exception as exc:
            self._fail_internal("MODEL_LOADING", exc)
            return self.state
        self._models = loaded
        self._can_return_to_ready = True
        self._machine.transition_to(AppState.READY)
        return self.state

    def _fail_internal(self, phase: str, exc: Exception) -> None:
        detail = f"{phase}: {type(exc).__name__}: {exc}"
        self._machine.fail(make_error(ErrorCode.INTERNAL_ERROR, detail))
