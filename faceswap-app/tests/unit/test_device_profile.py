"""Unit tests for non-GPU environment checks."""

import ast
import dataclasses
import pathlib
import sys

import pytest

from faceswap.runtime.device_profile import (
    MIN_LOGICAL_CPUS,
    MIN_RAM_GB,
    MIN_WINDOWS_BUILD,
    RECOMMENDED_RAM_GB,
    REQUIRED_PYTHON,
    CheckResult,
    CheckStatus,
    EnvironmentReport,
    SystemInfo,
    collect_system_info,
    evaluate_environment,
)

_PROFILE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "faceswap"
    / "runtime"
    / "device_profile.py"
)
_ALLOWED_MODULES = frozenset(
    {"__future__", "ctypes", "dataclasses", "enum", "os", "platform", "sys"}
)
_GB = 1024**3
_CHECK_NAMES = ("os", "windows_build", "architecture", "python", "ram", "cpu")


def _info(
    *,
    platform: str = "win32",
    windows_build: int | None = 22631,
    machine: str = "AMD64",
    python_version: tuple[int, int, int] = (3, 12, 10),
    installed_ram_bytes: int | None = 32 * _GB,
    logical_cpus: int | None = 16,
) -> SystemInfo:
    return SystemInfo(
        platform=platform,
        windows_build=windows_build,
        machine=machine,
        python_version=python_version,
        installed_ram_bytes=installed_ram_bytes,
        logical_cpus=logical_cpus,
    )


def test_e1_constants() -> None:
    assert MIN_WINDOWS_BUILD == 22000
    assert REQUIRED_PYTHON == (3, 12)
    assert MIN_RAM_GB == 16
    assert RECOMMENDED_RAM_GB == 32
    assert MIN_LOGICAL_CPUS == 8


def test_e2_good_profile_passes() -> None:
    report = evaluate_environment(_info())

    assert tuple(check.name for check in report.checks) == _CHECK_NAMES
    assert all(check.status == CheckStatus.PASS for check in report.checks)
    assert report.passed is True
    assert report.failures == ()


def test_e3_windows_build_threshold() -> None:
    old = evaluate_environment(_info(windows_build=19045))
    failed = old.get("windows_build")

    assert failed.status == CheckStatus.FAIL
    assert "22000" in failed.message
    assert "Windows 11" in failed.message
    assert old.passed is False
    assert old.failures == (failed,)

    exact = evaluate_environment(_info(windows_build=22000))
    assert exact.get("windows_build").status == CheckStatus.PASS


def test_e4_non_windows_platform() -> None:
    report = evaluate_environment(_info(platform="linux", windows_build=None))

    assert report.get("os").status == CheckStatus.FAIL
    assert report.get("windows_build").status == CheckStatus.UNKNOWN
    assert report.passed is False


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("AMD64", CheckStatus.PASS),
        ("amd64", CheckStatus.PASS),
        ("x86_64", CheckStatus.PASS),
        ("x86", CheckStatus.FAIL),
        ("ARM64", CheckStatus.FAIL),
    ],
)
def test_e5_architecture(machine: str, expected: CheckStatus) -> None:
    result = evaluate_environment(_info(machine=machine)).get("architecture")

    assert result.status == expected
    if expected == CheckStatus.FAIL:
        assert "Windows 11 x64 is required" in result.message


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((3, 12, 0), CheckStatus.PASS),
        ((3, 12, 10), CheckStatus.PASS),
        ((3, 11, 9), CheckStatus.FAIL),
        ((3, 13, 0), CheckStatus.FAIL),
    ],
)
def test_e6_python_version(version: tuple[int, int, int], expected: CheckStatus) -> None:
    result = evaluate_environment(_info(python_version=version)).get("python")

    assert result.status == expected


@pytest.mark.parametrize(
    ("ram", "expected"),
    [
        (8 * _GB, CheckStatus.WARN),
        (16 * _GB, CheckStatus.PASS),
        (32 * _GB, CheckStatus.PASS),
        (None, CheckStatus.UNKNOWN),
    ],
)
def test_e7_ram(ram: int | None, expected: CheckStatus) -> None:
    report = evaluate_environment(_info(installed_ram_bytes=ram))

    assert report.get("ram").status == expected
    if expected == CheckStatus.WARN:
        assert report.passed is True


@pytest.mark.parametrize(
    ("cpus", "expected"),
    [
        (4, CheckStatus.WARN),
        (8, CheckStatus.PASS),
        (None, CheckStatus.UNKNOWN),
    ],
)
def test_e8_cpu(cpus: int | None, expected: CheckStatus) -> None:
    report = evaluate_environment(_info(logical_cpus=cpus))

    assert report.get("cpu").status == expected
    if expected == CheckStatus.UNKNOWN:
        assert report.passed is True


def test_e9_messages_frozen_report_and_lookup() -> None:
    profiles = (
        _info(),
        _info(windows_build=19045),
        _info(platform="linux", windows_build=None),
        _info(machine="ARM64"),
        _info(python_version=(3, 11, 9)),
        _info(installed_ram_bytes=8 * _GB),
        _info(installed_ram_bytes=None),
        _info(logical_cpus=4),
        _info(logical_cpus=None),
    )
    for info in profiles:
        report = evaluate_environment(info)
        assert isinstance(report.checks, tuple)
        assert all(
            isinstance(check.message, str) and check.message.strip() for check in report.checks
        )

    info = _info()
    report = evaluate_environment(info)
    result = report.get("ram")
    assert isinstance(result, CheckResult)
    assert isinstance(info, SystemInfo)
    assert isinstance(report, EnvironmentReport)
    platform_name = "platform"
    message_name = "message"
    checks_name = "checks"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(info, platform_name, "linux")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, message_name, "changed")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(report, checks_name, ())
    assert report.get("ram").name == "ram"
    with pytest.raises(KeyError):
        report.get("gpu")


def test_e10_evaluate_environment_is_pure() -> None:
    info = _info()

    assert evaluate_environment(info) == evaluate_environment(info)


def test_e11_collect_system_info_on_this_machine() -> None:
    info = collect_system_info()

    assert isinstance(info, SystemInfo)
    assert info.platform == sys.platform
    assert info.python_version == tuple(sys.version_info[:3])
    if sys.platform == "win32":
        assert isinstance(info.windows_build, int)
        assert info.windows_build > 0
        assert info.installed_ram_bytes is None or info.installed_ram_bytes > 0
    assert info.logical_cpus is None or info.logical_cpus > 0


def test_e12_import_safety() -> None:
    tree = ast.parse(_PROFILE_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in _ALLOWED_MODULES
                assert not alias.name.startswith("faceswap")
                assert alias.name not in {"psutil", "logging"}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module in _ALLOWED_MODULES
            assert not module.startswith("faceswap")
            assert module not in {"psutil", "logging"}

    for statement in tree.body:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and _is_collect_call(node):
                raise AssertionError("module-level collect_system_info call")


def _is_collect_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "collect_system_info"
    if isinstance(func, ast.Attribute):
        return func.attr == "collect_system_info"
    return False
