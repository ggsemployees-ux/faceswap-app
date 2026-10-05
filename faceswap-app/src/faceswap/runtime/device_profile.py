"""Non-GPU environment checks for spec §7, §8, and §16.2 ENV_CHECK.

Facts about this machine are collected once, then evaluated against the Windows 11 x64,
Python 3.12, memory, and CPU minimums. GPU, CUDA, and VRAM checks come in a later step
(D-9.8).
"""

from __future__ import annotations

import ctypes
import enum
import os
import platform
import sys
from dataclasses import dataclass

MIN_WINDOWS_BUILD = 22000
"""Minimum supported Windows build (Windows 11)."""

REQUIRED_PYTHON = (3, 12)
"""Required Python major and minor version."""

MIN_RAM_GB = 16
"""Minimum installed RAM, in gibibytes (1024**3 bytes each)."""

RECOMMENDED_RAM_GB = 32
"""Recommended installed RAM, in gibibytes."""

MIN_LOGICAL_CPUS = 8
"""Minimum logical processors for a 4-core/8-thread CPU."""


@dataclass(frozen=True, slots=True)
class SystemInfo:
    """Raw facts about a machine. None means that fact could not be read.

    platform is sys.platform. windows_build is the Windows build, or None when this is
    not Windows or the build is unknown. machine is platform.machine(). python_version
    is major, minor, micro. installed_ram_bytes is physically installed memory.
    logical_cpus is the logical processor count.
    """

    platform: str
    windows_build: int | None
    machine: str
    python_version: tuple[int, int, int]
    installed_ram_bytes: int | None
    logical_cpus: int | None


class CheckStatus(enum.StrEnum):
    """Outcome of one environment check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One user-facing environment check result."""

    name: str
    status: CheckStatus
    message: str


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """Immutable evaluation of a SystemInfo profile."""

    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """Return True when no check failed. Warnings and unknowns do not block."""
        return all(check.status != CheckStatus.FAIL for check in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        """Return the FAIL checks, in report order."""
        return tuple(check for check in self.checks if check.status == CheckStatus.FAIL)

    def get(self, name: str) -> CheckResult:
        """Return the check with this name."""
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)


def collect_system_info() -> SystemInfo:
    """Read facts about this machine using the standard library.

    The installed-memory probe never raises: a failed or non-Windows call yields None.
    """
    version_info = sys.version_info
    return SystemInfo(
        platform=sys.platform,
        windows_build=_windows_build(),
        machine=platform.machine(),
        python_version=(version_info[0], version_info[1], version_info[2]),
        installed_ram_bytes=_installed_ram_bytes(),
        logical_cpus=os.cpu_count(),
    )


def evaluate_environment(info: SystemInfo) -> EnvironmentReport:
    """Evaluate machine facts against the §7 and §8 minimums. Performs no I/O."""
    return EnvironmentReport(
        (
            _check_os(info),
            _check_windows_build(info),
            _check_architecture(info),
            _check_python(info),
            _check_ram(info),
            _check_cpu(info),
        )
    )


def _windows_build() -> int | None:
    if sys.platform != "win32":
        return None
    return int(sys.getwindowsversion().build)


def _installed_ram_bytes() -> int | None:
    if sys.platform != "win32":
        return None
    try:
        total_kilobytes = ctypes.c_ulonglong()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_memory = kernel32.GetPhysicallyInstalledSystemMemory
        get_memory.argtypes = (ctypes.POINTER(ctypes.c_ulonglong),)
        get_memory.restype = ctypes.c_int
        if not get_memory(ctypes.byref(total_kilobytes)):
            return None
        return int(total_kilobytes.value) * 1024
    except Exception:
        return None


def _check_os(info: SystemInfo) -> CheckResult:
    if info.platform == "win32":
        return CheckResult("os", CheckStatus.PASS, "This PC is running Windows.")
    return CheckResult(
        "os",
        CheckStatus.FAIL,
        "Windows is required. Install Windows 11 x64 (build 22000 or later).",
    )


def _check_windows_build(info: SystemInfo) -> CheckResult:
    build = info.windows_build
    if build is None:
        return CheckResult(
            "windows_build",
            CheckStatus.UNKNOWN,
            "The Windows build could not be read. Windows 11 (build 22000 or later) is required.",
        )
    if build >= MIN_WINDOWS_BUILD:
        return CheckResult(
            "windows_build",
            CheckStatus.PASS,
            f"Windows build {build} meets Windows 11 (build {MIN_WINDOWS_BUILD} or later).",
        )
    return CheckResult(
        "windows_build",
        CheckStatus.FAIL,
        (
            f"Windows build {build} is below {MIN_WINDOWS_BUILD}. "
            "Windows 11 (build 22000 or later) is required."
        ),
    )


def _check_architecture(info: SystemInfo) -> CheckResult:
    if info.machine.upper() in {"AMD64", "X86_64"}:
        return CheckResult(
            "architecture",
            CheckStatus.PASS,
            f"Processor architecture {info.machine} is 64-bit x64.",
        )
    return CheckResult(
        "architecture",
        CheckStatus.FAIL,
        f"Windows 11 x64 is required. This PC reports architecture {info.machine}.",
    )


def _check_python(info: SystemInfo) -> CheckResult:
    version = info.python_version
    shown = ".".join(str(part) for part in version)
    if version[:2] == REQUIRED_PYTHON:
        return CheckResult(
            "python",
            CheckStatus.PASS,
            f"Python {shown} matches the required Python 3.12 baseline.",
        )
    return CheckResult(
        "python",
        CheckStatus.FAIL,
        f"Python {shown} is not supported. Install Python 3.12.",
    )


def _check_ram(info: SystemInfo) -> CheckResult:
    installed = info.installed_ram_bytes
    if installed is None:
        return CheckResult(
            "ram",
            CheckStatus.UNKNOWN,
            (
                "Installed memory could not be read. "
                f"At least {MIN_RAM_GB} GB is required; {RECOMMENDED_RAM_GB} GB is recommended."
            ),
        )
    if installed >= MIN_RAM_GB * (1024**3):
        return CheckResult(
            "ram",
            CheckStatus.PASS,
            (
                f"Installed memory meets the {MIN_RAM_GB} GB minimum "
                f"({RECOMMENDED_RAM_GB} GB recommended)."
            ),
        )
    return CheckResult(
        "ram",
        CheckStatus.WARN,
        (
            f"Installed memory is below {MIN_RAM_GB} GB. "
            f"{RECOMMENDED_RAM_GB} GB is recommended; performance may be reduced."
        ),
    )


def _check_cpu(info: SystemInfo) -> CheckResult:
    cpus = info.logical_cpus
    if cpus is None:
        return CheckResult(
            "cpu",
            CheckStatus.UNKNOWN,
            (
                "The number of logical processors could not be read. "
                f"At least {MIN_LOGICAL_CPUS} logical processors are recommended."
            ),
        )
    if cpus >= MIN_LOGICAL_CPUS:
        return CheckResult(
            "cpu",
            CheckStatus.PASS,
            (
                f"This PC has {cpus} logical processors, "
                f"which meets the minimum of {MIN_LOGICAL_CPUS}."
            ),
        )
    return CheckResult(
        "cpu",
        CheckStatus.WARN,
        (
            f"This PC has {cpus} logical processors. At least {MIN_LOGICAL_CPUS} "
            "(4-core/8-thread) are recommended; performance may be reduced."
        ),
    )
