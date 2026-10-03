"""Import smoke test for the faceswap package and its subpackages."""

import importlib

import pytest

_PACKAGES = (
    "faceswap",
    "faceswap.app",
    "faceswap.ui",
    "faceswap.capture",
    "faceswap.vision",
    "faceswap.inference",
    "faceswap.rendering",
    "faceswap.output",
    "faceswap.runtime",
    "faceswap.models",
)


@pytest.mark.parametrize("name", _PACKAGES)
def test_package_imports(name: str) -> None:
    importlib.import_module(name)
